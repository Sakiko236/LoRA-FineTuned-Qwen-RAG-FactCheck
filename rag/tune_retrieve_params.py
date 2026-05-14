import json
import sys
import os
import itertools
from pathlib import Path

DATA_PATH = Path("data/dev-claims.json")

# Search best parameters for evidence retrieval
PARAM_GRID = {
    "top_k_retrieve": [5],
    "threshold":      [2],
}

def compute_f1(retrieved_ids: list[str], gold_ids: list[str]) -> tuple[float, float, float]:
    retrieved_set = set(retrieved_ids)
    gold_set      = set(gold_ids)

    if not retrieved_set:
        precision = 0.0
    else:
        precision = len(retrieved_set & gold_set) / len(retrieved_set)

    if not gold_set:
        recall = 1.0       
    else:
        recall = len(retrieved_set & gold_set) / len(gold_set)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate(pipeline, claims: dict, top_k_retrieve: int, threshold: float) -> dict:
    precisions, recalls, f1s = [], [], []

    for claim_id, claim_data in claims.items():
        claim_text = claim_data["claim_text"]
        gold_ids   = claim_data.get("evidences", [])

        docs = pipeline.process_claim(
            claim_text,
            top_k_retrieve=top_k_retrieve,
            threshold=threshold,
        )
        retrieved_ids = [doc["id"] for doc in docs]

        p, r, f = compute_f1(retrieved_ids, gold_ids)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    n = len(claims)
    return {
        "macro_precision": sum(precisions) / n,
        "macro_recall":    sum(recalls)    / n,
        "macro_f1":        sum(f1s)        / n,
    }


def main():
    # Loading data
    if not DATA_PATH.exists():
        sys.exit(f"[ERROR] can't find {DATA_PATH}")

    print(f"[INFO] Loading data: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        claims = json.load(f)
    print(f"[INFO] Total {len(claims)} claims.\n")

    # Initialize RAG Pipeline
    sys.path.insert(0, str(Path(__file__).parent))
    from retrieve_rerank import RAGPipeline

    print("[INFO]Initializing RAGPipeline (loading models & index, may take a few minutes)")
    pipeline = RAGPipeline()
    print("[INFO] initialization completed. \n")

    # Generating all hyperparameter combinations
    param_names  = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combos       = list(itertools.product(*param_values))
    total        = len(combos)
    print(f"[INFO] begin search {total} combinations…\n")

    best_f1     = -1.0
    best_params = {}
    best_metrics = {}
    results     = []

    for i, combo in enumerate(combos, 1):
        params = dict(zip(param_names, combo))
        print(
            f"[{i:>3}/{total}] top_k={params['top_k_retrieve']:>3}  "
            f"threshold={params['threshold']:.2f}  ",
            end="",
            flush=True,
        )

        metrics = evaluate(
            pipeline,
            claims,
            top_k_retrieve=params["top_k_retrieve"],
            threshold=params["threshold"],
        )

        f1 = metrics["macro_f1"]
        print(
            f"P={metrics['macro_precision']:.4f}  "
            f"R={metrics['macro_recall']:.4f}  "
            f"F1={f1:.4f}"
        )

        results.append({**params, **metrics})

        if f1 > best_f1:
            best_f1      = f1
            best_params  = params.copy()
            best_metrics = metrics.copy()

    print("\n" + "=" * 70)
    print("best parameters:")
    for k, v in best_params.items():
        print(f"  {k:>15} = {v}")
    print(f"\n  Macro-Precision = {best_metrics['macro_precision']:.4f}")
    print(f"  Macro-Recall    = {best_metrics['macro_recall']:.4f}")
    print(f"  Macro-F1        = {best_metrics['macro_f1']:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
