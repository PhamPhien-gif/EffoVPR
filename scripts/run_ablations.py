from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Run configured EffoVPR evaluation ablations.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--checkpoint", default="artifacts/checkpoints/best.pt")
    parser.add_argument("--compact-checkpoint", action="append", default=[], metavar="DIM=PATH")
    parser.add_argument("--stages", nargs="+", choices=(
        "compact", "rerank_k", "layer", "facet", "t1", "t2", "resolution",
    ), default=("rerank_k", "layer", "facet", "t1", "t2", "resolution", "compact"))
    parser.add_argument("--split", choices=("val_queries", "test_queries"), default="val_queries")
    args = parser.parse_args()

    compact_checkpoints = {}
    for entry in args.compact_checkpoint:
        dimension, separator, checkpoint = entry.partition("=")
        if not separator or dimension not in {"128", "256", "1024"} or not checkpoint:
            raise ValueError(f"Invalid --compact-checkpoint {entry!r}; expected DIM=PATH")
        compact_checkpoints[int(dimension)] = checkpoint
    if "compact" not in args.stages and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    specs = {
        "rerank_k": ("rerank_k", [("top-k", value) for value in (5, 10, 25, 50, 100)]),
        "layer": ("layer", [("layer", value) for value in ("n-5", "n-4", "n-3", "n-2", "n-1", "n")]),
        "facet": ("facet", [("facet", value) for value in ("Q", "K", "V")]),
        "t1": ("t1", [("t1", value) for value in (0.00, 0.05, 0.10)]),
        "t2": ("t2", [("t2", value) for value in (0.50, 0.65, 0.80)]),
        "resolution": ("resolution", [("resolution", value) for value in (224, 322, 504)]),
    }
    output_root = Path("artifacts/evaluation/ablations")
    output_root.mkdir(parents=True, exist_ok=True)

    def evaluate(
        suffix: str,
        overrides: list[tuple[str, object]],
        checkpoint: str,
        dimension: int | None = None,
        mode: str = "effovpr-r",
    ):
        command = [
            sys.executable, "-m", "scripts.evaluate",
            "--config", args.config,
            "--checkpoint", checkpoint,
            "--mode", mode,
            "--split", args.split,
            "--output-suffix", f"ablation_{suffix}",
        ]
        if dimension is not None:
            command.extend(("--global-dim", str(dimension)))
        for option, value in overrides:
            command.extend((f"--{option}", str(value)))
        subprocess.run(command, check=True)
        result_path = Path("artifacts/evaluation") / f"effovpr-r_ablation_{suffix}" / "results.json"
        if mode != "effovpr-r":
            result_path = Path("artifacts/evaluation") / f"{mode}_ablation_{suffix}" / "results.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        return {
            "status": "COMPLETED",
            "checkpoint": checkpoint,
            **overrides_to_dict(overrides),
            **result["metrics"],
            "gallery_count": result["gallery_count"],
            "query_count": result["query_count"],
            "feature_dimension": result["global_feature_dimension"],
            "feature_storage_bytes": result["gallery_feature_storage_bytes"],
            "global_extraction_and_search_seconds": result["global_extraction_and_search_seconds"],
            "reranking_seconds": result["reranking_seconds"],
        }

    def overrides_to_dict(overrides: list[tuple[str, object]]) -> dict[str, object]:
        return {key.replace("-", "_"): value for key, value in overrides}

    for stage in args.stages:
        if stage == "compact":
            rows = []
            for dimension in (1024, 256, 128):
                checkpoint = compact_checkpoints.get(dimension)
                if checkpoint is None and dimension == 1024:
                    checkpoint = args.checkpoint
                if checkpoint is None or not Path(checkpoint).is_file():
                    rows.append({
                        "status": "NOT EXECUTED",
                        "dimension": dimension,
                        "reason": f"No trained checkpoint supplied for {dimension}D",
                    })
                    continue
                row = evaluate(
                    f"compact_{dimension}",
                    [],
                    checkpoint,
                    dimension=dimension,
                    mode="effovpr-g",
                )
                row["dimension"] = dimension
                row["feature_storage_bytes"] = int(row["gallery_count"] * dimension * 4)
                rows.append(row)
            pd.DataFrame(rows).to_csv(output_root / "compact_feature.csv", index=False)
            continue

        result_rows = []
        for option, value in specs[stage][1]:
            result_rows.append(evaluate(f"{stage}_{option}_{value}", [(option, value)], args.checkpoint))
        pd.DataFrame(result_rows).to_csv(output_root / f"{stage}.csv", index=False)
        print(f"Saved {stage} ablation to {output_root / f'{stage}.csv'}")


if __name__ == "__main__":
    main()
