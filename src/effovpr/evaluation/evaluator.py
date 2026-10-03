from __future__ import annotations

from typing import List

from .metrics import mean_reciprocal_rank, per_class_recall_at_k, recall_at_k


def evaluate_label_retrieval(query_labels: List[str], ranked_labels: List[List[str]], ks: List[int] = [1, 5, 10]):
    results = {}
    for k in ks:
        results[f"Recall@{k}"] = recall_at_k(query_labels, ranked_labels, k=k)
        per_class = per_class_recall_at_k(query_labels, ranked_labels, k=k)
        results[f"macro_Recall@{k}"] = float(sum(per_class.values()) / len(per_class)) if per_class else 0.0
        results[f"per_class_Recall@{k}"] = per_class
    results["MRR"] = mean_reciprocal_rank(query_labels, ranked_labels)
    return results
