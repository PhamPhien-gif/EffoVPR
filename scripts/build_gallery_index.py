from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.effovpr.retrieval.index import build_faiss_index, save_index_artifacts
from scripts.common import (
    build_model,
    checkpoint_identifier,
    get_dataset,
    get_splits,
    load_config,
    load_or_extract_global_features,
    records_for_split,
    set_deterministic_seed,
    write_run_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index over the gallery split.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    config = load_config(args.config)
    set_deterministic_seed(42)
    dataset = get_dataset(config)
    splits = get_splits(dataset, "artifacts/splits")
    gallery = records_for_split(dataset, splits["gallery"])
    if not gallery:
        raise ValueError("Gallery split is empty")
    checkpoint_path = args.checkpoint if Path(args.checkpoint).is_file() else None
    if checkpoint_path is None:
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    model = build_model(config, checkpoint_path=checkpoint_path)
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    resolution = int(config["inference"]["resolution"])
    features = load_or_extract_global_features(
        model,
        gallery,
        resolution,
        batch_size,
        split="gallery",
        checkpoint_path=checkpoint_path,
        config=config,
    )
    index = build_faiss_index(features, config["retrieval"]["index_type"])
    metadata = pd.DataFrame(
        [
            {
                "image_id": record["id"],
                "label": record["label"],
                "dataset_index": record["dataset_index"],
                "global_feature_dimension": int(features.shape[1]),
                "checkpoint_identifier": checkpoint_identifier(checkpoint_path),
                "input_resolution": resolution,
                "model_identifier": config["model"]["backbone"],
            }
            for record in gallery
        ]
    )
    output = Path("artifacts/index")
    save_index_artifacts(
        str(output / "gallery.index"),
        str(output / "gallery_features.npy"),
        str(output / "gallery_metadata.parquet"),
        str(output / "index_config.json"),
        index,
        features,
        metadata,
        {
            "index_type": config["retrieval"]["index_type"],
            "normalized": True,
            "count": int(index.ntotal),
            "feature_dimension": int(features.shape[1]),
            "checkpoint_identifier": checkpoint_identifier(checkpoint_path),
            "input_resolution": resolution,
            "model_identifier": config["model"]["backbone"],
        },
    )
    write_run_metadata("artifacts/run_metadata.json", config, checkpoint_path, 42)
    print(f"Saved {index.ntotal} gallery vectors ({features.shape[1]}D) to {output}")


if __name__ == "__main__":
    main()
