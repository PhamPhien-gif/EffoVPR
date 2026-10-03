from __future__ import annotations

import torch


def mutual_nearest_neighbor_matches(query_local: torch.Tensor, candidate_local: torch.Tensor, threshold: float = 0.65):
    query_local = torch.as_tensor(query_local, dtype=torch.float32)
    candidate_local = torch.as_tensor(candidate_local, dtype=torch.float32)
    if query_local.ndim != 2 or candidate_local.ndim != 2:
        raise ValueError("Query and candidate local descriptors must be 2D tensors.")
    q_norm = query_local / (query_local.norm(dim=1, keepdim=True) + 1e-8)
    c_norm = candidate_local / (candidate_local.norm(dim=1, keepdim=True) + 1e-8)
    similarity = q_norm @ c_norm.T
    if similarity.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long)
    query_to_candidate = similarity.argmax(dim=1)
    candidate_to_query = similarity.argmax(dim=0)
    n_queries = query_local.size(0)
    query_ids = torch.arange(n_queries, device=similarity.device)
    mutual = candidate_to_query[query_to_candidate] == query_ids
    valid_scores = similarity[query_ids, query_to_candidate][mutual]
    keep = valid_scores > threshold
    retained = query_ids[mutual][keep]
    matched_candidates = query_to_candidate[mutual][keep]
    # return a pair list (query_index, candidate_index)
    if retained.numel() == 0:
        return torch.empty((0, 2), dtype=torch.long, device=similarity.device)
    return torch.stack([retained, matched_candidates], dim=1).long()
