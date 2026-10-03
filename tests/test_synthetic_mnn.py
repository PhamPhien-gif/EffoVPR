import numpy as np

from src.effovpr.reranking.mnn import mutual_nearest_neighbor_matches


def test_synthetic_mutual_match():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    matches = mutual_nearest_neighbor_matches(q, c, threshold=0.65)
    assert matches.shape[0] == 2


def test_synthetic_threshold_eliminates_pair():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    c = np.array([[0.8, 0.6]], dtype=np.float32)
    matches = mutual_nearest_neighbor_matches(q, c, threshold=0.95)
    assert matches.shape[0] == 0


def test_synthetic_one_way_nearest_neighbor_not_mutual():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    c = np.array([[1.0, 0.0], [0.9, 0.1]], dtype=np.float32)
    matches = mutual_nearest_neighbor_matches(q, c, threshold=0.5)
    assert matches.shape[0] == 1
