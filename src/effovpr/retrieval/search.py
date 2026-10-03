from __future__ import annotations

import numpy as np

from .index import search_faiss


def global_retrieval(index, query_features: np.ndarray, top_k: int = 10):
    scores, ids = search_faiss(index, query_features, top_k=top_k)
    return scores, ids
