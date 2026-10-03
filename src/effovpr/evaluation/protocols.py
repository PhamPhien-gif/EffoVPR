from __future__ import annotations

from typing import List


def label_based_correctness(query_label: str, gallery_labels: List[str]):
    return any(label == query_label for label in gallery_labels)
