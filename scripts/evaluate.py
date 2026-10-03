from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.effovpr.evaluation.evaluator import evaluate_label_retrieval
from src.effovpr.retrieval.index import build_faiss_index, load_index_artifacts, search_faiss
from src.effovpr.reranking.mnn import mutual_nearest_neighbor_matches
from scripts.common import (
    build_model,
    checkpoint_identifier,
    extract_local_descriptors,
    get_dataset,
    get_splits,
    load_config,
    load_or_extract_global_features,
    records_for_split,
    set_deterministic_seed,
    write_run_metadata,
)


MODES = ("dino-zs", "effovpr-zs", "effovpr-g", "effovpr-r")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate VN_Attractions label-based retrieval.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=MODES, default="effovpr-r")
    parser.add_argument("--split", choices=("val_queries", "test_queries"), default="test_queries")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--layer")
    parser.add_argument("--facet", choices=("Q", "K", "V"))
    parser.add_argument("--t1", type=float)
    parser.add_argument("--t2", type=float)
    parser.add_argument("--resolution", type=int)
    parser.add_argument("--global-dim", type=int)
    parser.add_argument("--output-suffix", default="")
    parser.add_argument("--limit", type=int, help="Limit gallery and query counts for a smoke/limited experiment")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.global_dim is not None:
        if args.global_dim not in {128, 256, 1024}:
            raise ValueError("--global-dim must be one of 128, 256, or 1024")
        config["model"]["global_dim"] = args.global_dim
    if args.resolution is not None:
        if args.resolution <= 0 or args.resolution % 14:
            raise ValueError("--resolution must be a positive multiple of the backbone patch size (14)")
        config["inference"]["resolution"] = args.resolution
    if args.layer is not None:
        config["reranking"]["layer"] = args.layer
        config["reranking"]["zero_shot_layer"] = args.layer
    if args.facet is not None:
        config["reranking"]["facet"] = args.facet
    if args.t1 is not None:
        config["reranking"]["T1"] = args.t1
    if args.t2 is not None:
        config["reranking"]["T2"] = args.t2
    if args.top_k is not None:
        config["reranking"]["top_k"] = args.top_k
    set_deterministic_seed(42)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    zero_shot = args.mode in {"dino-zs", "effovpr-zs"}
    rerank = args.mode in {"effovpr-zs", "effovpr-r"}
    if not zero_shot and not args.checkpoint:
        raise ValueError(f"{args.mode} requires --checkpoint")
    if args.checkpoint and not Path(args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    dataset = get_dataset(config)
    splits = get_splits(dataset, "artifacts/splits")
    if args.limit is not None:
        gallery_frame = splits["gallery"].head(args.limit)
        gallery_labels = set(gallery_frame["label"].astype(str))
        query_frame = splits[args.split]
        positive_queries = query_frame[query_frame["label"].astype(str).isin(gallery_labels)]
        query_frame = (positive_queries if not positive_queries.empty else query_frame).head(args.limit)
    else:
        gallery_frame = splits["gallery"]
        query_frame = splits[args.split]
    gallery = records_for_split(dataset, gallery_frame)
    queries = records_for_split(dataset, query_frame)
    if not gallery or not queries:
        raise ValueError("Evaluation requires non-empty gallery and query splits")
    class_count = len({str(sample["label"]) for sample in dataset})
    model = build_model(
        config,
        class_count=class_count,
        checkpoint_path=args.checkpoint,
        zero_shot=zero_shot,
    )
    resolution = int(config["inference"]["resolution"])
    batch_size = args.batch_size or int(config["training"]["batch_size"])
    started = time.perf_counter()
    gallery_features = load_or_extract_global_features(
        model, gallery, resolution, batch_size, "gallery", args.checkpoint, config, zero_shot=zero_shot
    )
    query_features = load_or_extract_global_features(
        model, queries, resolution, batch_size, args.split, args.checkpoint, config, zero_shot=zero_shot
    )
    expected_index_config = {
        "index_type": config["retrieval"]["index_type"],
        "normalized": True,
        "count": len(gallery),
        "feature_dimension": int(gallery_features.shape[1]),
        "checkpoint_identifier": checkpoint_identifier(args.checkpoint),
        "input_resolution": resolution,
        "model_identifier": config["model"]["backbone"],
    }
    loaded_index = None
    if not zero_shot and args.limit is None:
        loaded_index = load_index_artifacts(
            "artifacts/index/gallery.index",
            "artifacts/index/gallery_features.npy",
            "artifacts/index/gallery_metadata.parquet",
            "artifacts/index/index_config.json",
            expected_index_config,
        )
    if loaded_index is None:
        index = build_faiss_index(gallery_features, config["retrieval"]["index_type"])
        index_source = "features cache (in-memory FAISS)"
    else:
        index, indexed_features, index_metadata, _ = loaded_index
        if index_metadata["image_id"].astype(str).tolist() != [record["id"] for record in gallery]:
            raise ValueError("Gallery metadata order does not match the deterministic gallery split")
        if index_metadata["label"].astype(str).tolist() != [record["label"] for record in gallery]:
            raise ValueError("Gallery metadata labels do not match the deterministic gallery split")
        if not np.allclose(indexed_features, gallery_features, rtol=1e-5, atol=1e-6):
            raise ValueError("Saved gallery index features differ from the active feature cache")
        index_source = "artifacts/index/gallery.index"
    requested_k = args.top_k or int(config["reranking"]["top_k"])
    if requested_k <= 0:
        raise ValueError("--top-k must be positive")
    candidate_k = min(max(requested_k, 100, 10), len(gallery))
    global_scores, global_indices = search_faiss(index, query_features, top_k=candidate_k)
    feature_extraction_and_search_seconds = time.perf_counter() - started

    reranked_indices: list[list[int]] = []
    reranked_scores: list[list[float]] = []
    rerank_seconds = 0.0
    if rerank:
        rerank_config = config["reranking"]
        local_layer = rerank_config["zero_shot_layer"] if zero_shot else rerank_config["layer"]
        facet = str(rerank_config["facet"]).upper()
        t1 = float(rerank_config["T1"])
        t2 = float(rerank_config["T2"])
        for query_index, query in enumerate(queries):
            rerank_start = time.perf_counter()
            query_local = extract_local_descriptors(
                model,
                query["image"],
                resolution,
                local_layer,
                facet,
                t1,
            )
            scored = []
            for original_rank, candidate_index in enumerate(global_indices[query_index][:requested_k]):
                candidate_index = int(candidate_index)
                if candidate_index < 0:
                    continue
                candidate_local = extract_local_descriptors(
                    model,
                    gallery[candidate_index]["image"],
                    resolution,
                    local_layer,
                    facet,
                    t1,
                )
                matches = mutual_nearest_neighbor_matches(query_local, candidate_local, threshold=t2)
                scored.append((candidate_index, float(len(matches)), original_rank, float(global_scores[query_index, original_rank])))
            scored.sort(key=lambda item: (-item[1], item[2]))
            reranked = [(item[0], item[1]) for item in scored]
            reranked_ids = {item[0] for item in scored}
            for original_rank, candidate_index in enumerate(global_indices[query_index]):
                candidate_index = int(candidate_index)
                if candidate_index >= 0 and candidate_index not in reranked_ids:
                    reranked.append((candidate_index, 0.0))
            reranked_indices.append([item[0] for item in reranked])
            reranked_scores.append([item[1] for item in reranked])
            rerank_seconds += time.perf_counter() - rerank_start
    else:
        reranked_indices = [
            [int(candidate) for candidate in row if candidate >= 0]
            for row in global_indices
        ]
        reranked_scores = [
            [float(score) for score, candidate in zip(score_row, id_row) if candidate >= 0]
            for score_row, id_row in zip(global_scores, global_indices)
        ]

    ranked_labels = [[gallery[index]["label"] for index in row] for row in reranked_indices]
    global_ranked_labels = [
        [gallery[int(index)]["label"] for index in row if index >= 0]
        for row in global_indices
    ]
    metrics = evaluate_label_retrieval(
        [query["label"] for query in queries],
        ranked_labels,
    )
    global_metrics = evaluate_label_retrieval(
        [query["label"] for query in queries],
        global_ranked_labels,
    )
    query_predictions = []
    failures = []
    for i, query in enumerate(queries):
        global_row = [int(candidate) for candidate in global_indices[i] if candidate >= 0]
        final_row = reranked_indices[i]
        global_correct_rank = next((rank for rank, idx in enumerate(global_row, 1) if gallery[idx]["label"] == query["label"]), None)
        final_correct_rank = next((rank for rank, idx in enumerate(final_row, 1) if gallery[idx]["label"] == query["label"]), None)
        global_top = global_row[0] if global_row else None
        final_top = final_row[0] if final_row else None
        row = {
            "query_id": query["id"],
            "query_label": query["label"],
            "global_rank_1_id": gallery[global_top]["id"] if global_top is not None else "",
            "global_rank_1_label": gallery[global_top]["label"] if global_top is not None else "",
            "global_rank_1_score": float(global_scores[i, 0]) if len(global_row) else None,
            "reranked_rank_1_id": gallery[final_top]["id"] if final_top is not None else "",
            "reranked_rank_1_label": gallery[final_top]["label"] if final_top is not None else "",
            "reranked_rank_1_score": reranked_scores[i][0] if reranked_scores[i] else None,
            "top_5_ids": [gallery[idx]["id"] for idx in final_row[:5]],
            "top_5_labels": [gallery[idx]["label"] for idx in final_row[:5]],
            "top_10_ids": [gallery[idx]["id"] for idx in final_row[:10]],
            "top_10_labels": [gallery[idx]["label"] for idx in final_row[:10]],
            "is_correct_at_1": bool(final_correct_rank == 1),
            "is_correct_at_5": bool(final_correct_rank is not None and final_correct_rank <= 5),
            "is_correct_at_10": bool(final_correct_rank is not None and final_correct_rank <= 10),
            "global_correct_rank": global_correct_rank,
            "reranked_correct_rank": final_correct_rank,
        }
        query_predictions.append(row)
        if global_correct_rank != 1 or final_correct_rank != 1:
            if global_correct_rank != 1 and final_correct_rank == 1:
                failure_category = "global wrong -> reranker correct"
            elif global_correct_rank == 1 and final_correct_rank != 1:
                failure_category = "global correct -> reranker wrong"
            elif (final_correct_rank is not None and final_correct_rank <= 10) or (
                global_correct_rank is not None and global_correct_rank <= 10
            ):
                failure_category = "correct result exists in top-10 but not top-1"
            elif (final_correct_rank is not None and final_correct_rank <= 100) or (
                global_correct_rank is not None and global_correct_rank <= 100
            ):
                failure_category = "correct result exists in top-100 but not top-10"
            else:
                failure_category = "correct result absent from top-100"
            failures.append({
                "query_id": query["id"],
                "query_label": query["label"],
                "global_prediction": row["global_rank_1_label"],
                "reranked_prediction": row["reranked_rank_1_label"],
                "global_correct_rank": global_correct_rank,
                "reranked_correct_rank": final_correct_rank,
                "failure_category": failure_category,
                "global_top_candidates": global_row[:10],
                "mnn_top_scores": reranked_scores[i][:10],
            })

    output_name = args.mode + (f"_{args.output_suffix}" if args.output_suffix else "")
    output = Path("artifacts/evaluation") / output_name
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "LIMITED EXPERIMENT" if args.split == "val_queries" or args.limit is not None else "FULL EXPERIMENT",
        "sample_limit": args.limit,
        "evaluation_protocol": "VN_Attractions Label-Based Retrieval",
        "mode": args.mode,
        "resolution": resolution,
        "reranking": {
            "top_k": requested_k,
            "layer": config["reranking"]["zero_shot_layer"] if zero_shot else config["reranking"]["layer"],
            "facet": config["reranking"]["facet"],
            "T1": config["reranking"]["T1"],
            "T2": config["reranking"]["T2"],
        },
        "query_split": args.split,
        "gallery_count": len(gallery),
        "query_count": len(queries),
        "metrics": metrics,
        "global_metrics": global_metrics,
        "global_feature_dimension": int(gallery_features.shape[1]),
        "retrieval_index": index_source,
        "gallery_feature_storage_bytes": int(gallery_features.nbytes),
        "global_extraction_and_search_seconds": feature_extraction_and_search_seconds,
        "reranking_seconds": rerank_seconds,
    }
    (output / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(query_predictions).to_csv(output / "query_predictions.csv", index=False)
    failure_dir = Path("artifacts/failure_analysis") / output_name
    failure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(failures).to_csv(failure_dir / "failure_cases.csv", index=False)
    write_run_metadata("artifacts/run_metadata.json", config, args.checkpoint, 42)
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
