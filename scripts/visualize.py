from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import torch

from src.effovpr.models.dino_attention import attention_patch_scores
from src.effovpr.reranking.mnn import mutual_nearest_neighbor_matches
from scripts.common import build_model, get_dataset, image_batch, load_config, set_deterministic_seed


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize retrieval results and optional EffoVPR local matches.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--checkpoint", default="artifacts/model")
    parser.add_argument("--mode", choices=("dino-zs", "effovpr-zs", "effovpr-g", "effovpr-r"), default="effovpr-r")
    parser.add_argument("--results", default="")
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")
    config = load_config(args.config)
    zero_shot = args.mode in {"dino-zs", "effovpr-zs"}
    local_reranking = args.mode in {"effovpr-zs", "effovpr-r"}
    checkpoint = None if zero_shot else args.checkpoint
    if checkpoint and not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    results_path = Path(args.results) if args.results else Path("artifacts/evaluation") / args.mode / "query_predictions.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Evaluation predictions do not exist: {results_path}")
    set_deterministic_seed(42)
    dataset = get_dataset(config)
    model = build_model(config, class_count=len({str(sample["label"]) for sample in dataset}), checkpoint_path=checkpoint, zero_shot=zero_shot)
    model.eval()
    id_to_index = {str(image_id): index for index, image_id in enumerate(dataset["id"])}

    def get_record(image_id: str):
        index = id_to_index.get(str(image_id))
        return dataset[index] if index is not None else None

    predictions = pd.read_csv(results_path).head(args.limit)
    output_dir = Path("artifacts/visualizations") / args.mode
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = int(config["inference"]["resolution"])
    layer = config["reranking"]["zero_shot_layer"] if zero_shot else config["reranking"]["layer"]
    facet = config["reranking"]["facet"].lower()
    threshold1 = float(config["reranking"]["T1"])
    threshold2 = float(config["reranking"]["T2"])

    with torch.inference_mode():
        for row in predictions.itertuples(index=False):
            query = get_record(str(row.query_id))
            if query is None:
                raise ValueError(f"Query id {row.query_id!r} is not in the loaded dataset")
            candidate_ids = ast.literal_eval(row.top_10_ids) if isinstance(row.top_10_ids, str) else []
            candidates = [record for image_id in candidate_ids if (record := get_record(str(image_id))) is not None]
            if not candidates:
                continue
            name = safe_name(str(row.query_id))
            if local_reranking:
                selected_features = []
                selected_grids = []
                score_maps = []
                for record in [query, candidates[0]]:
                    pixels = image_batch(model, [record["image"]], resolution)
                    qkv = model.extract_qkv(pixels, layer=layer)
                    scores = attention_patch_scores(qkv["q_patch"][0], qkv["k"][0, 0])
                    mask = scores > threshold1
                    selected_features.append(qkv[f"{facet}_patch"][0, mask].float())
                    selected_grids.append(mask.nonzero(as_tuple=False).flatten().cpu().numpy())
                    grid_size = resolution // model.backbone.patch_size
                    score_maps.append(scores.reshape(grid_size, grid_size).cpu().numpy())
                matches = mutual_nearest_neighbor_matches(selected_features[0], selected_features[1], threshold=threshold2)
                match_pairs = matches.cpu().numpy()

                figure = plt.figure(figsize=(18, 10), constrained_layout=True)
                grid = figure.add_gridspec(2, 10, height_ratios=(3, 1))
                query_axis = figure.add_subplot(grid[0, :5])
                candidate_axis = figure.add_subplot(grid[0, 5:])
                query_image = query["image"].convert("RGB").resize((resolution, resolution))
                candidate_image = candidates[0]["image"].convert("RGB").resize((resolution, resolution))
                query_axis.imshow(query_image)
                candidate_axis.imshow(candidate_image)
                patch_size = model.backbone.patch_size
                for axis, patch_ids, image_width in (
                    (query_axis, selected_grids[0], resolution),
                    (candidate_axis, selected_grids[1], resolution),
                ):
                    for patch_id in patch_ids:
                        x = int(patch_id % (resolution // patch_size)) * patch_size
                        y = int(patch_id // (resolution // patch_size)) * patch_size
                        axis.add_patch(Rectangle((x, y), patch_size, patch_size, fill=False, edgecolor="yellow", linewidth=0.8))
                    axis.set_xlim(0, image_width)
                    axis.set_ylim(resolution, 0)
                    axis.axis("off")
                query_axis.set_title(f"Query: {row.query_id} | GT={row.query_label}")
                candidate_axis.set_title(
                    f"Top-1: {candidates[0]['id']} | pred={candidates[0]['label']} | MNN={len(match_pairs)}"
                )
                grid_size = resolution // patch_size
                for query_patch, candidate_patch in match_pairs[:100]:
                    qx = (int(selected_grids[0][query_patch]) % grid_size + 0.5) * patch_size
                    qy = (int(selected_grids[0][query_patch]) // grid_size + 0.5) * patch_size
                    cx = (int(selected_grids[1][candidate_patch]) % grid_size + 0.5) * patch_size
                    cy = (int(selected_grids[1][candidate_patch]) // grid_size + 0.5) * patch_size
                    figure.add_artist(plt.Line2D(
                        [qx / resolution * 0.5, 0.5 + cx / resolution * 0.5],
                        [0.25 + qy / resolution * 0.58, 0.25 + cy / resolution * 0.58],
                        transform=figure.transFigure,
                        color="lime",
                        alpha=0.45,
                        linewidth=0.6,
                    ))
                for column, candidate in enumerate(candidates[:10]):
                    axis = figure.add_subplot(grid[1, column])
                    axis.imshow(candidate["image"].convert("RGB"))
                    axis.set_title(str(candidate["label"]), fontsize=8)
                    axis.axis("off")
                figure.savefig(output_dir / f"retrieval_{name}.png", dpi=120)
                plt.close(figure)

                heatmap, heat_axis = plt.subplots(figsize=(6, 5))
                heat_axis.imshow(score_maps[0], cmap="magma")
                heat_axis.set_title(f"Attention heatmap: {row.query_id}")
                heat_axis.axis("off")
                heatmap.savefig(output_dir / f"attention_{name}.png", dpi=150)
                plt.close(heatmap)
            else:
                figure = plt.figure(figsize=(18, 12), constrained_layout=True)
                grid = figure.add_gridspec(3, 5)
                query_axis = figure.add_subplot(grid[0, :2])
                query_axis.imshow(query["image"].convert("RGB"))
                query_axis.set_title(f"Query | GT={row.query_label}")
                query_axis.axis("off")
                candidate_axes = [
                    figure.add_subplot(grid[row_index, column_index])
                    for row_index in range(1, 3)
                    for column_index in range(5)
                ]
                for candidate_rank, (axis, candidate) in enumerate(zip(candidate_axes, candidates[:10]), start=1):
                    axis.imshow(candidate["image"].convert("RGB"))
                    correctness = "correct" if str(candidate["label"]) == str(row.query_label) else "wrong"
                    axis.set_title(f"#{candidate_rank} {correctness}\n{candidate['label']}", fontsize=8)
                    axis.axis("off")
                figure.suptitle(f"{args.mode}: query {row.query_id}")
                figure.savefig(output_dir / f"retrieval_{name}.png", dpi=120)
                plt.close(figure)
    print(f"Saved visualizations for {len(predictions)} queries to {output_dir}")


if __name__ == "__main__":
    main()
