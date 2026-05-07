import os
import sys
import json
import re
from datetime import datetime
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from infer.inference import ClaimVerifier

def extract_label(text):
    text_upper = text.upper()

    matches = re.finditer(r'\b(SUPPORTS|REFUTES|NOT_ENOUGH_INFO|DISPUTED)\b', text_upper)
    
    found_labels = [m.group(0) for m in matches]
    
    if found_labels:
        return found_labels[-1]
    else:
        return "UNKNOWN_FORMAT"

def evaluate_dev_set(base_model_id, lora_path, dev_file_path, output_dir, batch_size=2):
    print("Initializing Verifier...")
    verifier = ClaimVerifier(base_model_id=base_model_id, lora_path=lora_path)
    
    print(f"Loading Dev Set from {dev_file_path}...")
    with open(dev_file_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
        
    results = []
    correct_count = 0
    total_count = len(dev_data)
    
    print(f"Starting Batch Evaluation on {total_count} claims (Batch Size: {batch_size})...")
    
    items = list(dev_data.items()) if isinstance(dev_data, dict) else dev_data
    
    for i in tqdm(range(0, total_count, batch_size), desc="Evaluating in Batches"):
        batch_items = items[i:i + batch_size]
        
        batch_claim_ids = []
        batch_claim_texts = []
        batch_true_labels = []
        batch_evidences = []
        
        for item in batch_items:
            if isinstance(dev_data, dict):
                claim_id = item[0]
                claim_info = item[1]
            else:
                claim_info = item
                claim_id = claim_info.get('id', 'unknown_id')
                
            batch_claim_ids.append(claim_id)
            batch_claim_texts.append(claim_info.get('claim_text', ''))
            batch_true_labels.append(claim_info.get('claim_label', ''))
            batch_evidences.append(claim_info.get('evidences', []))
        
        batch_model_outputs = verifier.predict(
            batch_claim_texts, 
            evidence_ids=batch_evidences, 
            few_shot=True if lora_path is None else False
        )

        for j in range(len(batch_model_outputs)):
            model_output = batch_model_outputs[j]
            true_label = batch_true_labels[j]
            claim_id = batch_claim_ids[j]
            claim_text = batch_claim_texts[j]
            
            predicted_label = extract_label(model_output)
            
            is_correct = (predicted_label == true_label)
            if is_correct:
                correct_count += 1
                
            results.append({
                "claim_id": claim_id,
                "claim_text": claim_text,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "is_correct": is_correct,
                "model_raw_output": model_output
            })
        
    accuracy = correct_count / total_count if total_count > 0 else 0
    print("\n" + "="*50)
    print(f"Evaluation Complete!")
    print(f"Total Claims : {total_count}")
    print(f"Correct      : {correct_count}")
    print(f"Accuracy     : {accuracy:.4%} ({accuracy:.4f})")
    print("="*50)
    
    final_output = {
        "metadata": {
            "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "base_model": base_model_id,
            "lora_version": lora_path if lora_path else "None (Base Model)",
            "total_samples": total_count,
            "batch_size": batch_size,
            "accuracy": accuracy
        },
        "predictions": results
    }
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_name = os.path.basename(lora_path) if lora_path else "base"
    output_filename = os.path.join(output_dir, f"eval_dev_{version_name}_{timestamp}.json")
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
        
    print(f"Detailed results saved to: {output_filename}")


if __name__ == "__main__":
    BASE_MODEL = "Qwen/Qwen3.5-2B"
    # "model/qwen-cot-lora-final" or None
    LORA_PATH = "model/qwen-cot-lora-final" 
    
    DEV_FILE = "data/dev-claims.json"
    OUTPUT_DIR = "infer/results"
    
    BATCH_SIZE = 2
    
    # evaluate_dev_set(
    #     base_model_id=BASE_MODEL,
    #     lora_path=LORA_PATH,
    #     dev_file_path=DEV_FILE,
    #     output_dir=OUTPUT_DIR,
    #     batch_size=BATCH_SIZE
    # )

    evaluate_dev_set(
        base_model_id=BASE_MODEL,
        lora_path=None,
        dev_file_path=DEV_FILE,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE
    )