from __future__ import annotations

import numpy as np


def recall_at_k(labels: list[str], preds: list[str], k: int = 1):
    if len(labels) != len(preds):
        raise ValueError("labels and preds length mismatch")
    correct = 0
    for gt, topk in zip(labels, preds):
        topk_list = topk[:k]
        if gt in topk_list:
            correct += 1
    return correct / len(labels) if len(labels) else 0.0


def mean_reciprocal_rank(labels: list[str], preds: list[str]):
    if len(labels) != len(preds):
        raise ValueError("labels and preds length mismatch")
    rrs = []
    for gt, ranked in zip(labels, preds):
        for idx, item in enumerate(ranked, start=1):
            if item == gt:
                rrs.append(1.0 / idx)
                break
        else:
            rrs.append(0.0)
    return float(np.mean(rrs)) if rrs else 0.0


def per_class_recall_at_k(labels: list[str], preds: list[str], k: int = 1):
    if len(labels) != len(preds):
        raise ValueError("labels and preds length mismatch")
    grouped: dict[str, list[bool]] = {}
    for label, ranked in zip(labels, preds):
        grouped.setdefault(str(label), []).append(str(label) in [str(item) for item in ranked[:k]])
    return {label: float(np.mean(values)) for label, values in sorted(grouped.items())}
