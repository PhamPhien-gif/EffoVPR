from __future__ import annotations

import torch


def select_attention_keypoints(scores: torch.Tensor, threshold: float = 0.05):
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    mask = scores > threshold
    selected = scores[mask]
    return mask, selected


def summarize_keypoint_selection(scores: torch.Tensor):
    if scores.ndim != 1:
        scores = scores.reshape(-1)
    selected = scores[scores > 0.05]
    return {
        "average_selected_patches": float(selected.mean().item()) if selected.numel() > 0 else 0.0,
        "minimum_selected_patches": int(selected.numel() if selected.numel() > 0 else 0),
        "maximum_selected_patches": int(selected.numel()),
        "std_selected_patches": float(selected.std(unbiased=False).item()) if selected.numel() > 1 else 0.0,
    }
