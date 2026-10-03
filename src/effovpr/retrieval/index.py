from __future__ import annotations

import json
import os

import faiss
import numpy as np
import pandas as pd


def build_faiss_index(features: np.ndarray, index_type: str = "IndexFlatIP"):
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got {features.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("Feature matrix must be non-empty")
    if not np.isfinite(features).all():
        raise ValueError("Feature matrix contains non-finite values")
    if index_type == "IndexFlatIP":
        index = faiss.IndexFlatIP(features.shape[1])
    else:
        raise ValueError(f"Unsupported index type: {index_type}")
    normalized = features.astype(np.float32, copy=False)
    norms = np.linalg.norm(normalized, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("Features must be L2-normalized before IndexFlatIP insertion")
    index.add(normalized)
    return index


def search_faiss(index, query_features: np.ndarray, top_k: int = 10):
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if query_features.ndim != 2 or query_features.shape[1] != index.d:
        raise ValueError(f"Query features must have shape [N, {index.d}]")
    if not np.isfinite(query_features).all():
        raise ValueError("Query features contain non-finite values")
    query_features = query_features.astype(np.float32, copy=False)
    norms = np.linalg.norm(query_features, axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("Query features must be L2-normalized")
    distances, indices = index.search(query_features, min(top_k, index.ntotal))
    return distances, indices


def save_index_artifacts(index_path: str, features_path: str, metadata_path: str, config_path: str, index, features, metadata, config):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    if index.ntotal != len(metadata) or index.ntotal != len(features):
        raise ValueError(f"Index, feature, and metadata counts differ: {index.ntotal}, {len(features)}, {len(metadata)}")
    faiss.write_index(index, index_path)
    np.save(features_path, features)
    metadata.to_parquet(metadata_path, index=False)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def load_index_artifacts(
    index_path: str,
    features_path: str,
    metadata_path: str,
    config_path: str,
    expected_config: dict | None = None,
):
    paths = tuple(map(os.fspath, (index_path, features_path, metadata_path, config_path)))
    if not all(os.path.isfile(path) for path in paths):
        return None

    index_path, features_path, metadata_path, config_path = paths
    index = faiss.read_index(index_path)
    features = np.load(features_path, allow_pickle=False)
    metadata = pd.read_parquet(metadata_path)
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    if not isinstance(config, dict):
        raise ValueError("Index configuration must be a JSON object")
    if expected_config is not None:
        mismatches = {
            key: (config.get(key), expected_value)
            for key, expected_value in expected_config.items()
            if config.get(key) != expected_value
        }
        if mismatches:
            raise ValueError(f"Saved index configuration does not match expected configuration: {mismatches}")

    if features.ndim != 2:
        raise ValueError(f"Saved gallery features must be 2D, got {features.shape}")
    if index.ntotal != len(features) or index.ntotal != len(metadata):
        raise ValueError(
            "Saved index, feature, and metadata counts differ: "
            f"{index.ntotal}, {len(features)}, {len(metadata)}"
        )
    if features.shape[1] != index.d:
        raise ValueError(
            f"Saved feature dimension {features.shape[1]} does not match index dimension {index.d}"
        )
    if not np.isfinite(features).all():
        raise ValueError("Saved gallery features contain non-finite values")
    norms = np.linalg.norm(features.astype(np.float32, copy=False), axis=1)
    if np.any(np.abs(norms - 1.0) > 1e-3):
        raise ValueError("Saved gallery features are not L2-normalized")
    if config.get("count") != index.ntotal:
        raise ValueError("Saved index count does not match its configuration")
    if config.get("feature_dimension") != index.d:
        raise ValueError("Saved index dimension does not match its configuration")
    if config.get("index_type") != "IndexFlatIP":
        raise ValueError(f"Unsupported saved index type: {config.get('index_type')!r}")

    return index, features, metadata, config
