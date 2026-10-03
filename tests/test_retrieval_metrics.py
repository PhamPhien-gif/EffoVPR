from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.effovpr.evaluation.evaluator import evaluate_label_retrieval
from src.effovpr.retrieval.index import (
    build_faiss_index,
    load_index_artifacts,
    save_index_artifacts,
    search_faiss,
)


def test_faiss_index_search_requires_normalized_features():
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = build_faiss_index(gallery)
    scores, indices = search_faiss(index, np.asarray([[1.0, 0.0]], dtype=np.float32), top_k=5)
    assert indices.tolist() == [[0, 1]]
    assert scores.shape == (1, 2)
    with pytest.raises(ValueError, match="L2-normalized"):
        build_faiss_index(np.asarray([[2.0, 0.0]], dtype=np.float32))


def test_faiss_artifacts_validate_metadata_alignment(tmp_path):
    index = build_faiss_index(np.asarray([[1.0, 0.0]], dtype=np.float32))
    metadata = pd.DataFrame([{"image_id": "a"}])
    save_index_artifacts(
        str(tmp_path / "gallery.index"),
        str(tmp_path / "gallery_features.npy"),
        str(tmp_path / "gallery_metadata.parquet"),
        str(tmp_path / "index_config.json"),
        index,
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        metadata,
        {},
    )
    assert (tmp_path / "gallery.index").is_file()
    with pytest.raises(ValueError, match="counts differ"):
        save_index_artifacts(
            str(tmp_path / "bad" / "gallery.index"),
            str(tmp_path / "bad" / "gallery_features.npy"),
            str(tmp_path / "bad" / "gallery_metadata.parquet"),
            str(tmp_path / "bad" / "index_config.json"),
            index,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            metadata.iloc[:0],
            {},
        )


def test_saved_index_load_checks_its_metadata(tmp_path):
    features = np.asarray([[1.0, 0.0]], dtype=np.float32)
    metadata = pd.DataFrame([{"image_id": "a", "label": "A"}])
    config = {
        "index_type": "IndexFlatIP",
        "normalized": True,
        "count": 1,
        "feature_dimension": 2,
        "checkpoint_identifier": "checkpoint.pt",
        "input_resolution": 504,
        "model_identifier": "dino",
    }
    paths = [tmp_path / name for name in ("gallery.index", "gallery_features.npy", "gallery_metadata.parquet", "index_config.json")]
    save_index_artifacts(*(str(path) for path in paths), build_faiss_index(features), features, metadata, config)
    loaded = load_index_artifacts(*(str(path) for path in paths), expected_config=config)
    assert loaded is not None
    assert loaded[0].ntotal == 1
    with pytest.raises(ValueError, match="configuration"):
        load_index_artifacts(*(str(path) for path in paths), expected_config={**config, "input_resolution": 322})


def test_metrics_report_micro_macro_and_per_class():
    result = evaluate_label_retrieval(
        ["a", "a", "b"],
        [["a"], ["b"], ["b"]],
        ks=[1],
    )
    assert result["Recall@1"] == pytest.approx(2 / 3)
    assert result["macro_Recall@1"] == pytest.approx(0.75)
    assert result["per_class_Recall@1"] == {"a": 0.5, "b": 1.0}
