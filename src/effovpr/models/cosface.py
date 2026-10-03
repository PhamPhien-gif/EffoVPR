from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CosFace(nn.Module):
    def __init__(self, in_features: int, out_features: int, scale: float = 30.0, margin: float = 0.4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, targets: torch.Tensor):
        if targets is None:
            raise ValueError("CosFace requires target class indices")
        if embeddings.dim() != 2:
            embeddings = embeddings.view(embeddings.size(0), -1)
        if embeddings.size(1) != self.in_features:
            raise ValueError(f"Expected embedding dimension {self.in_features}, got {embeddings.size(1)}")
        if targets.ndim != 1 or targets.size(0) != embeddings.size(0):
            raise ValueError("targets must be a 1D tensor with one class index per embedding")
        if targets.dtype != torch.long:
            targets = targets.long()
        if targets.numel() and (targets.min() < 0 or targets.max() >= self.out_features):
            raise ValueError(f"targets must be in [0, {self.out_features})")
        norm_embeddings = F.normalize(embeddings, p=2, dim=1)
        norm_weight = F.normalize(self.weight, p=2, dim=1)
        cosine = norm_embeddings @ norm_weight.T
        target_mask = F.one_hot(targets, num_classes=self.out_features).to(dtype=cosine.dtype)
        logits = (cosine - self.margin * target_mask) * self.scale
        loss = F.cross_entropy(logits, targets)
        return logits, loss
