import json
import sys
import os
import itertools
from pathlib import Path

DATA_PATH = Path("data/dev-claims.json")

# Search best parameters for evidence retrieval
PARAM_GRID = {
    "top_k_retrieve": [20, 30, 50],
    "threshold":      [0.0, 1.0, 2.0, 3.0, 4.0],
    "max_results":    [2, 3, 5, 8],
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


def evaluate(pipeline, claims: dict, top_k_retrieve: int, threshold: float, max_results: int) -> dict:
    precisions, recalls, f1s = [], [], []

    for claim_id, claim_data in claims.items():
        claim_text = claim_data["claim_text"]
        gold_ids   = claim_data.get("evidences", [])

        docs = pipeline.process_claim(
            claim_text,
            top_k_dense=top_k_retrieve,
            top_k_sparse=top_k_retrieve//2,
            threshold=threshold,
            max_results=max_results,
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
            f"threshold={params['threshold']:.1f}  max_results={params['max_results']:>2}  ",
            end="",
            flush=True,
        )

        metrics = evaluate(
            pipeline,
            claims,
            top_k_retrieve=params["top_k_retrieve"],
            threshold=params["threshold"],
            max_results=params["max_results"],
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

    # Saving all results to JSON
    out_path = Path("tune_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_params":  best_params,
                "best_metrics": best_metrics,
                "all_results":  sorted(results, key=lambda x: x["macro_f1"], reverse=True),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[INFO] all results saved to: {out_path}")


if __name__ == "__main__":
    main()
