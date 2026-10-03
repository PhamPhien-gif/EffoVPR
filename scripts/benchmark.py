from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.effovpr.models.dino_attention import attention_patch_scores
from src.effovpr.retrieval.index import build_faiss_index, search_faiss
from src.effovpr.reranking.mnn import mutual_nearest_neighbor_matches
from scripts.common import (
    build_model,
    extract_global_features,
    extract_local_descriptors,
    get_dataset,
    get_splits,
    image_batch,
    load_config,
    records_for_split,
    set_deterministic_seed,
    write_run_metadata,
)


def summarize(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "p95_ms": float(np.percentile(samples, 95)) * 1000,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure EffoVPR inference stages on real dataset images.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--mode", choices=("dino-zs", "effovpr-zs", "effovpr-g", "effovpr-r"), default="dino-zs")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if args.runs < 1 or args.warmup < 0 or args.top_k < 1:
        raise ValueError("--runs and --top-k must be positive; --warmup cannot be negative")
    if args.mode in {"effovpr-g", "effovpr-r"} and not args.checkpoint:
        raise ValueError(f"{args.mode} requires --checkpoint")

    config = load_config(args.config)
    set_deterministic_seed(42)
    dataset = get_dataset(config)
    splits = get_splits(dataset, "artifacts/splits")
    gallery = records_for_split(dataset, splits["gallery"])
    queries = records_for_split(dataset, splits["test_queries"])
    if not gallery or not queries:
        raise ValueError("Benchmark requires non-empty gallery and test-query splits")
    zero_shot = args.mode in {"dino-zs", "effovpr-zs"}
    model = build_model(
        config,
        class_count=len({str(sample["label"]) for sample in dataset}),
        checkpoint_path=args.checkpoint,
        zero_shot=zero_shot,
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    query = queries[0]
    candidates = gallery[: min(args.top_k, len(gallery))]
    resolution = int(config["inference"]["resolution"])
    rerank_layer = config["reranking"]["zero_shot_layer"] if zero_shot else config["reranking"]["layer"]
    facet = config["reranking"]["facet"]
    device = next(model.parameters()).device

    def measure(operation):
        for _ in range(args.warmup):
            operation()
        timings = []
        for _ in range(args.runs):
            start = time.perf_counter()
            operation()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            timings.append(time.perf_counter() - start)
        return summarize(timings)

    preprocessed = image_batch(model, [query["image"]], resolution)
    global_gallery_records = gallery[: min(100, len(gallery))]
    gallery_features = extract_global_features(model, global_gallery_records, resolution, batch_size=1, zero_shot=zero_shot)
    query_feature = extract_global_features(model, [query], resolution, batch_size=1, zero_shot=zero_shot)
    index = build_faiss_index(gallery_features, config["retrieval"]["index_type"])
    qkv = model.extract_qkv(preprocessed, layer=rerank_layer)
    attention_scores = attention_patch_scores(qkv["q_patch"][0], qkv["k"][0, 0])
    selected_mask = attention_scores > float(config["reranking"]["T1"])
    local_query = qkv[f"{facet.lower()}_patch"][0, selected_mask].float()
    local_candidate = extract_local_descriptors(
        model,
        candidates[0]["image"],
        resolution,
        rerank_layer,
        facet,
        float(config["reranking"]["T1"]),
    )

    measurements = {}
    measurements["preprocessing"] = measure(lambda: image_batch(model, [query["image"]], resolution))
    measurements["backbone_forward"] = measure(lambda: model.backbone(preprocessed))
    measurements["global_feature_extraction"] = measure(lambda: extract_global_features(model, [query], resolution, 1, zero_shot=zero_shot))
    measurements["faiss_search"] = measure(lambda: search_faiss(index, query_feature, top_k=min(args.top_k, index.ntotal)))
    measurements["qkv_extraction"] = measure(lambda: model.extract_qkv(preprocessed, layer=rerank_layer))
    measurements["local_patch_selection"] = measure(
        lambda: attention_scores > float(config["reranking"]["T1"])
    )
    measurements["mnn"] = measure(
        lambda: mutual_nearest_neighbor_matches(local_query, local_candidate, float(config["reranking"]["T2"]))
    )

    def rerank_records(query_record, candidate_records):
        query_desc = extract_local_descriptors(
            model, query_record["image"], resolution, rerank_layer, facet, float(config["reranking"]["T1"])
        )
        counts = []
        for candidate in candidate_records:
            candidate_desc = extract_local_descriptors(
                model, candidate["image"], resolution, rerank_layer, facet, float(config["reranking"]["T1"])
            )
            counts.append(len(mutual_nearest_neighbor_matches(query_desc, candidate_desc, float(config["reranking"]["T2"]))))
        return counts

    measurements["reranking"] = measure(lambda: rerank_records(query, candidates))

    def end_to_end():
        query_vector = extract_global_features(model, [query], resolution, 1, zero_shot=zero_shot)
        scores, ids = search_faiss(index, query_vector, top_k=min(args.top_k, index.ntotal))
        if args.mode in {"effovpr-zs", "effovpr-r"}:
            candidate_records = [global_gallery_records[int(index_id)] for index_id in ids[0] if index_id >= 0]
            local_scores = rerank_records(query, candidate_records)
            return scores, ids, sorted(zip(candidate_records, local_scores), key=lambda pair: -pair[1])
        return scores, ids

    measurements["end_to_end"] = measure(end_to_end)
    model_memory = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    feature_bytes = int(gallery_features.nbytes)
    result = {
        "status": "LIMITED EXPERIMENT",
        "mode": args.mode,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda,
        "pytorch_version": torch.__version__,
        "python_version": platform.python_version(),
        "batch_size": 1,
        "resolution": resolution,
        "checkpoint": args.checkpoint or "pretrained-zero-shot",
        "runs": args.runs,
        "warmup": args.warmup,
        "gallery_vectors_benchmarked": int(index.ntotal),
        "measurements": measurements,
        "memory": {
            "model_parameter_bytes": int(model_memory),
            "gallery_feature_bytes": feature_bytes,
            "faiss_index_estimated_bytes": feature_bytes,
            "peak_inference_vram_bytes": int(torch.cuda.max_memory_allocated(device)) if torch.cuda.is_available() else None,
            "local_descriptor_bytes_query": int(local_query.numel() * local_query.element_size()),
            "theoretical_gallery_float32_bytes": int(len(gallery) * model.global_dim * 4),
        },
    }
    output = Path("artifacts/benchmarks")
    output.mkdir(parents=True, exist_ok=True)
    (output / "benchmark.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"stage": stage, **values} for stage, values in measurements.items()]
    ).to_csv(output / "benchmark.csv", index=False)
    write_run_metadata("artifacts/run_metadata.json", config, args.checkpoint, 42)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
