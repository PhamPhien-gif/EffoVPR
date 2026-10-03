from __future__ import annotations

import torch


def local_descriptors_from_v(v_patch: torch.Tensor):
    return v_patch


def compute_local_similarity(query_desc: torch.Tensor, candidate_desc: torch.Tensor):
    q = query_desc / (query_desc.norm(dim=-1, keepdim=True) + 1e-8)
    c = candidate_desc / (candidate_desc.norm(dim=-1, keepdim=True) + 1e-8)
    return q @ c.T
