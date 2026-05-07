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

    def predict(self, claim_text, evidence_ids=None):
        print("-" * 50)
        print(f"Claim: {claim_text}")
        
        combined_evidence = self.fetch_evidence(claim_text, evidence_ids)
        print(f"\nExtracted Evidence:\n{combined_evidence}\n")

        messages = [
            {"role": "system", "content": "You are a factual verification assistant. Think step-by-step to classify the claim based on the evidence."},
            {"role": "user", "content": f"Evidence:\n{combined_evidence}\n\nClaim:\n{claim_text}"}
        ]

        text_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer([text_prompt], return_tensors="pt").to(self.model.device)

        print("Generating prediction...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

        input_length = inputs.input_ids.shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        print(f"\nModel Output:\n{response}")
        print("-" * 50)
        return response

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
    verifier.predict(test_claim_2)