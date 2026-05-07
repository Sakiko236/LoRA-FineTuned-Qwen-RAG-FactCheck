import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.retrieve_rerank import RAGPipeline
from fine_tuning.retrieve_evidence import get_evidence_text

class ClaimVerifier:
    def __init__(self, base_model_id="Qwen/Qwen3.5-2B", lora_path=None):
        print("Loading Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print("Configuring 4-bit Quantization...")
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype
        )

        print(f"Loading Base Model: {base_model_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=compute_dtype,
            attn_implementation="sdpa"
        )

        if lora_path and os.path.exists(lora_path):
            print(f"Applying LoRA weights from {lora_path}...")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            print("LoRA loaded successfully!")
        else:
            print("No valid LoRA path provided. Using Base Model for inference.")
            
        self.rag_pipeline = None

    def fetch_evidence(self, claim_text, evidence_ids=None):
        evidence_texts = []
        
        if evidence_ids and len(evidence_ids) > 0:
            print(f"Using provided evidence IDs: {evidence_ids}")
            for eid in evidence_ids:
                text = get_evidence_text(eid)
                if text:
                    evidence_texts.append(f"[{eid}] {text}")
        else:
            print("No evidence IDs provided. Triggering RAG Pipeline...")
            if self.rag_pipeline is None:
                self.rag_pipeline = RAGPipeline()
            
            top_evidence = self.rag_pipeline.process_claim(claim_text, top_k_retrieve=20, top_k_rerank=5)
            for ev in top_evidence:
                evidence_texts.append(f"[{ev['id']}] {ev['text']}")
                
        return "\n".join(evidence_texts)

def predict(self, claims_texts, batch_evidence_ids, few_shot=False):
        self.tokenizer.padding_side = 'left'
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        text_prompts = []
        
        for claim_text, evidence_ids in zip(claims_texts, batch_evidence_ids):
            combined_evidence = self.fetch_evidence(claim_text, evidence_ids)
            
            if few_shot:
                messages = [
                    {"role": "system", "content": """You are a climate science fact-checker. Your task is to explain the logical connection between the Evidence and the Claim to justify the Label. Use the format: "Let's analyze step by step: [Reasoning] Therefore, the conclusion is: [Label]." conclude your response with one of the following four labels: 
                    [SUPPORTS, REFUTES, NOT_ENOUGH_INFO, DISPUTED]. Here are some examples:
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
                    Therefore, the conclusion is: SUPPORTS."""},
                    {"role": "user", "content": f"Evidence:\n{combined_evidence}\n\nClaim:\n{claim_text}"}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a climate science fact-checker. Your task is to explain the logical connection between the Evidence and the Claim to justify the Label."},
                    {"role": "user", "content": f"Evidence:\n{combined_evidence}\n\nClaim:\n{claim_text}"}
                ]

            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            text_prompts.append(prompt)

        inputs = self.tokenizer(
            text_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to(self.model.device)

        print(f"Generating predictions for a batch of {len(claims_texts)} claims...")
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id
            )

        input_length = inputs.input_ids.shape[1]
        clean_responses = []
        
        for output in outputs:
            generated_tokens = output[input_length:]
            full_response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            clean_response = full_response
            for word in ["Evidence:", "Claim:", "Label:"]:
                if word in clean_response:
                    clean_response = clean_response.split(word)[0]
            
            clean_responses.append(clean_response.strip())

        return clean_responses

if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen3.5-2B"
    LORA_PATH = "model/qwen-cot-lora-final"
    
    verifier = ClaimVerifier(base_model_id=BASE_MODEL, lora_path=LORA_PATH)
    
    test_claim_1 = "[South Australia] has the most expensive electricity in the world."
    test_evidences_1 = ["evidence-67732", "evidence-572512"]
    print("\n>>> TEST CASE 1: With Evidence IDs <<<")
    verifier.predict(test_claim_1, evidence_ids=test_evidences_1)
    
    test_claim_2 = "Higher CO2 concentrations actually help ecosystems support more plant and animal life by increasing plant growth speed."
    print("\n>>> TEST CASE 2: No Evidence IDs (Trigger RAG) <<<")
    verifier.predict(test_claim_2, evidence_ids=None)