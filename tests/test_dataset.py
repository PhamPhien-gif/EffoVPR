from __future__ import annotations

import pytest
from PIL import Image

from src.effovpr.data.audit import compute_dataset_audit, save_dataset_audit
from src.effovpr.data.split import deterministic_label_split
from src.effovpr.data.dataset import build_label_mapping


def test_label_mapping_and_split():
    rows = [
        {"id": "a1", "image": None, "label": "A"},
        {"id": "a2", "image": None, "label": "A"},
        {"id": "a3", "image": None, "label": "A"},
        {"id": "b1", "image": None, "label": "B"},
        {"id": "b2", "image": None, "label": "B"},
        {"id": "c1", "image": None, "label": "C"},
    ]
    split = deterministic_label_split(rows, seed=42)
    assert set(split) == {"train", "gallery", "val_queries", "test_queries"}
    assert sum(len(split[k]) for k in split) == len(rows)
    mapping = build_label_mapping(rows)
    assert mapping["A"] == 0
    assert mapping["B"] == 1
    assert mapping["C"] == 2


def test_split_is_deterministic_and_duplicate_images_stay_together():
    shared = Image.new("RGB", (4, 4), color="red")
    rows = [
        {"id": f"a{index}", "image": Image.new("RGB", (4, 4), color=(index, 0, 0)), "label": "A"}
        for index in range(1, 12)
    ]
    rows.extend([
        {"id": "dupe-1", "image": shared.copy(), "label": "B"},
        {"id": "dupe-2", "image": shared.copy(), "label": "B"},
    ])
    first = deterministic_label_split(rows, seed=19)
    second = deterministic_label_split(rows, seed=19)
    for name in first:
        assert first[name].equals(second[name])
    assignments = {}
    for name, frame in first.items():
        for image_id in frame["id"]:
            assignments[image_id] = name
    assert assignments["dupe-1"] == assignments["dupe-2"]


def test_split_rejects_duplicate_ids():
    rows = [
        {"id": "same", "image": None, "label": "A"},
        {"id": "same", "image": None, "label": "A"},
    ]
    with pytest.raises(ValueError, match="Duplicate sample IDs"):
        deterministic_label_split(rows)


def test_audit_counts_duplicate_images_and_serializes_resolution_stats(tmp_path):
    image = Image.new("RGB", (3, 5), color="blue")
    rows = [
        {"id": "x", "image": image.copy(), "label": "A"},
        {"id": "y", "image": image.copy(), "label": "A"},
        {"id": "z", "image": None, "label": "B"},
    ]
    stats = compute_dataset_audit(rows)
    assert stats["duplicate_image_count"] == 1
    assert stats["invalid_image_count"] == 1
    assert stats["image_resolution_statistics"] == {"3x5": 2}
    save_dataset_audit(stats, str(tmp_path))
    assert (tmp_path / "dataset_audit.json").is_file()
    assert (tmp_path / "dataset_audit.csv").is_file()
