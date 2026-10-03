from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.effovpr.data.dataset import build_label_mapping
from scripts.common import ensure_dataset_audit, get_dataset, get_splits, load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    args = parser.parse_args()
    dataset = get_dataset(load_config(args.config))
    stats = ensure_dataset_audit(dataset)
    frames = get_splits(dataset, args.output_dir, seed=args.seed)
    mapping_path = Path("artifacts/class_mapping.json")
    mapping_path.parent.mkdir(parents=True, exist_ok=True)
    mapping_path.write_text(
        json.dumps(build_label_mapping(dataset), indent=2, ensure_ascii=True, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(
        {"audit": stats, "split_counts": {key: len(value) for key, value in frames.items()}},
        ensure_ascii=True,
    ))


if __name__ == "__main__":
    main()
