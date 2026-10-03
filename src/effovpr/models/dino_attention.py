from __future__ import annotations

import math

import torch


def resolve_layer_index(layer, total_layers: int) -> int:
    if isinstance(layer, str):
        layer = layer.strip().lower()
        if layer.startswith("n-"):
            offset = int(layer.split("-")[-1])
            idx = total_layers - offset
        elif layer == "n":
            idx = total_layers - 1
        elif layer.startswith("n") and layer[1:].isdigit():
            idx = int(layer[1:]) - 1
        else:
            raise ValueError(f"Unsupported layer token: {layer}")
    else:
        idx = int(layer)
    if idx < 0:
        idx = total_layers + idx
    if idx < 0 or idx >= total_layers:
        raise ValueError(f"Layer index {idx} is out of range for total layers {total_layers}.")
    return idx


def parse_tokens(hidden_tokens: torch.Tensor):
    if hidden_tokens.ndim != 3:
        raise ValueError(f"Expected [batch, tokens, dim], got {tuple(hidden_tokens.shape)}")
    batch_size = hidden_tokens.size(0)
    cls_token = hidden_tokens[:, 0:1, :]
    register_tokens = hidden_tokens[:, 1:5, :] if hidden_tokens.size(1) > 5 else hidden_tokens[:, 1:1, :]
    patch_tokens = hidden_tokens[:, 5:, :]
    return cls_token, register_tokens, patch_tokens


def attention_patch_scores(q_patch: torch.Tensor, k_cls: torch.Tensor, temperature: float | None = None):
    if q_patch.ndim != 2:
        q_patch = q_patch.reshape(q_patch.size(0), -1)
    if temperature is None:
        temperature = math.sqrt(q_patch.size(-1))
    scores = (q_patch @ k_cls.reshape(-1)) / temperature
    return torch.softmax(scores, dim=-1)


def assert_patch_count(patch_tokens: torch.Tensor, image_resolution: int | None = None):
    patch_count = patch_tokens.shape[1]
    grid_h = int(round(math.sqrt(patch_count)))
    grid_w = patch_count // grid_h if grid_h > 0 else 0
    if image_resolution is not None:
        expected = (image_resolution // 14) ** 2
        if patch_count != expected:
            raise AssertionError(f"Expected patch count {expected} for resolution {image_resolution}, got {patch_count}")
    else:
        if grid_h * grid_w != patch_count:
            raise AssertionError(f"patch count mismatch: {patch_count} != {grid_h} * {grid_w}")
