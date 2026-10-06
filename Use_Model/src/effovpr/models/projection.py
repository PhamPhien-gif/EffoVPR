from __future__ import annotations

import torch
import torch.nn as nn


class GlobalProjection(nn.Module):
    def __init__(self, input_dim: int = 1024, output_dim: int = 1024):
        super().__init__()
        if output_dim not in (128, 256, 1024):
            raise ValueError("Supported global dimensions are 128, 256, and 1024.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)
