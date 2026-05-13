import json
import sqlite3
import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from retrieve_evidence import get_evidence_text
from tqdm import tqdm

try:
    # pyrefly: ignore [missing-import]
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN')
except (ImportError, ModuleNotFoundError):
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")

login(token=hf_token)

model_id = "Qwen/Qwen3.5-2B"

# Left-padding is required for batched generation so all sequences
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto",
    quantization_config=quantization_config,
    trust_remote_code=True
)

model.generation_config.max_length = None
model.config.pad_token_id = tokenizer.pad_token_id

# ── DB ────────────────────────────────────────────────────────────────────────

def init_db(db_path="db/cot.db"):
    dir_name = os.path.dirname(db_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS reasoning_data (
            claim_id TEXT PRIMARY KEY,
            claim_text TEXT,
            label TEXT,
            reasoning TEXT
        )
    ''')
    conn.commit()
    return conn

# ── Prompt ────────────────────────────────────────────────────────────────────

def format_prompt(evidence, claim, label):
    few_shot = """You are a climate science fact-checker. Your task is to explain the logical connection between the Evidence and the Claim to justify the Label. Use the format: "Let's analyze step by step: [Reasoning] Therefore, the conclusion is: [Label]."
                    Evidence: 1. CO2 can be toxic to animals at 10,000 ppm. 2. Plants grow faster at 1,000 ppm CO2. 3. Higher CO2 affects plant growth favorably.
                    Claim: Higher CO2 concentrations actually help ecosystems support more plant and animal life.
                    Label: DISPUTED
                    Let's analyze step by step: The evidence confirms that CO2 promotes plant growth, which supports part of the claim. However, it also notes that extremely high concentrations are toxic to animal life. Since the claim makes a broad positive statement without accounting for these toxic thresholds, the claim is partially accurate but also potentially dangerous/misleading.
                    Therefore, the conclusion is: DISPUTED.

                    Evidence: 1. Human activity and GHG emissions are key factors in global temperature increases. 2. Warming is driven by human-caused thermal expansion and melting ice.
                    Claim: El Niño drove record highs in global temperatures suggesting rise may not be down to man-made emissions.
                    Label: REFUTES
                    Let's analyze step by step: While El Niño is a natural driver of temperature, the evidence explicitly states that human activity is the "key factor" in the pace of current temperature increases. The claim attempts to dismiss man-made emissions by pointing to a natural cause, which contradicts the "substantial evidence" mentioned in the text regarding human-caused warming.
                    Therefore, the conclusion is: REFUTES.

                    Evidence: 1. Reversals in polarity occurred around 1925, 1947, and 1977. 2. The PDO changed to a "cool" phase in a regime shift similar to the 1970s.
                    Claim: In 1946, PDO switched to a cool phase.
                    Label: SUPPORTS
                    Let's analyze step by step: The evidence mentions a major PDO reversal occurring around 1947 and explicitly describes a shift to a "cool" phase. The year 1946 is immediately adjacent to the 1947 reversal date cited. Given the context of regime shifts, the evidence provides sufficient support for the timing and nature of the phase change described in the claim.
                    Therefore, the conclusion is: SUPPORTS.

                    Evidence: {evidence}
                    Claim: {claim}
                    Label: {label}
                    Let's analyze step by step:"""
    return few_shot.format(evidence=evidence, claim=claim, label=label)

# ── Batched generation ────────────────────────────────────────────────────────

def generate_batch(batch_items: list[dict], max_new_tokens: int = 300) -> list[str]:
    """
    Tokenize each prompt individually (as-is), then left-pad into a single
    batch tensor and call model.generate() once for the whole batch.

    Returns a list of decoded reasoning strings, one per item.
    """
    prompts = [item["prompt"] for item in batch_items]

    # Single-item tokenization (no padding here, just get token ids)
    input_ids_list = [
        tokenizer(p, return_tensors="pt")["input_ids"][0]
        for p in prompts
    ]

    # Manual left-pad so all sequences end at the same position
    max_len = max(ids.shape[0] for ids in input_ids_list)
    padded_ids = []
    attention_masks = []
    for ids in input_ids_list:
        pad_len = max_len - ids.shape[0]
        padded = torch.cat([
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
            ids
        ])
        mask = torch.cat([
            torch.zeros(pad_len, dtype=torch.long),
            torch.ones(ids.shape[0], dtype=torch.long)
        ])
        padded_ids.append(padded)
        attention_masks.append(mask)

    input_ids = torch.stack(padded_ids).to(model.device)
    attention_mask = torch.stack(attention_masks).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    results = []
    for out in outputs:
        new_tokens = out[max_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(text)

    return results

# ── Main ──────────────────────────────────────────────────────────────────────

BATCH_SIZE = 4
MAX_NEW_TOKENS = 300

db_conn = init_db()
cursor = db_conn.cursor()

# Checkpoint / resume: skip already-processed claims
cursor.execute("SELECT claim_id FROM reasoning_data")
done_ids = {row[0] for row in cursor.fetchall()}
print(f"Resuming: {len(done_ids)} claims already in DB.")

with open('data/train-claims.json', 'r') as f:
    train_data = json.load(f)

all_claims_ready = []
for cid, data in train_data.items():
    if cid in done_ids:
        continue

    evidence_pieces = []
    for i, ev_id in enumerate(data['evidences'], start=1):
        ev_text = get_evidence_text(ev_id)
        if ev_text:
            evidence_pieces.append(f"{i}. {ev_text}")
        else:
            print(f"Warning: {ev_id} not found in DB, skipping.")

    if not evidence_pieces:
        print(f"Skipping {cid}: no evidence retrieved.")
        continue

    evidence_text = " ".join(evidence_pieces)
    claim_text = data['claim_text']
    label = data['claim_label']

    prompt = format_prompt(evidence_text, claim_text, label)
    all_claims_ready.append({
        "cid": cid,
        "prompt": prompt,
        "claim_text": claim_text,
        "label": label
    })

print(f"To generate: {len(all_claims_ready)} claims (batch_size={BATCH_SIZE})")

for batch_start in tqdm(range(0, len(all_claims_ready), BATCH_SIZE), desc="Generating CoT"):
    batch_items = all_claims_ready[batch_start: batch_start + BATCH_SIZE]

    reasonings = generate_batch(batch_items, max_new_tokens=MAX_NEW_TOKENS)

    for item, reasoning in zip(batch_items, reasonings):
        cursor.execute(
            "INSERT OR REPLACE INTO reasoning_data VALUES (?, ?, ?, ?)",
            (item["cid"], item["claim_text"], item["label"], reasoning)
        )

    # Commit after every batch so progress is always saved
    db_conn.commit()

db_conn.close()
print("Done! All CoT reasoning saved to DB.")