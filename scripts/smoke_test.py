from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from src.effovpr.data.dataset import load_vn_attractions
from src.effovpr.evaluation.evaluator import evaluate_label_retrieval
from src.effovpr.models.effovpr import EffoVPR
from src.effovpr.models.dino_attention import attention_patch_scores
from src.effovpr.retrieval.index import build_faiss_index, search_faiss
from src.effovpr.reranking.mnn import mutual_nearest_neighbor_matches
from src.effovpr.utils.checkpoint import load_checkpoint, save_checkpoint
from src.effovpr.utils.device import get_device
from scripts.common import image_batch, load_config, set_deterministic_seed


def run_smoke_test(
    config_path: str = "configs/effovpr_vn_attractions.yaml",
    cache_dir: str | None = None,
    resolution: int = 224,
) -> dict:
    if resolution <= 0 or resolution % 14:
        raise ValueError("Smoke-test resolution must be a positive multiple of 14")
    config = load_config(config_path)
    if cache_dir:
        config.setdefault("dataset", {})["cache_dir"] = str(Path(cache_dir) / "datasets")
        config["model"]["cache_dir"] = str(Path(cache_dir) / "models")
    set_deterministic_seed(42)
    dataset = load_vn_attractions(
        split="train[:2]",
        cache_dir=config.get("dataset", {}).get("cache_dir"),
    )
    if not len(dataset):
        raise RuntimeError("The VN_Attractions smoke-test split is empty")
    class_count = max(2, len({str(item["label"]) for item in dataset}))
    model = EffoVPR(
        backbone_name=config["model"]["backbone"],
        global_dim=128,
        num_classes=class_count,
        fine_tune_last_n_layers=1,
        with_registers=bool(config["model"].get("with_registers", True)),
    ).to(get_device())
    images = [item["image"].convert("RGB") for item in dataset if item["image"] is not None]
    if not images:
        raise RuntimeError("No valid dataset images were available for smoke test")
    training_pixels = image_batch(model, images, 224)
    model.train()
    labels = torch.arange(len(images), dtype=torch.long, device=get_device())
    global_features, logits, loss = model(training_pixels, labels=labels)
    loss.backward()
    if global_features.shape != (len(images), 128) or logits.shape != (len(images), class_count):
        raise AssertionError("Model output dimensions do not match the smoke-test contract")

    model.eval()
    with torch.inference_mode():
        pixels = image_batch(model, images, resolution)
        qkv = model.extract_qkv(pixels, layer="n-1")
        backbone_output = model.backbone(pixels)
        final_block = model.backbone.encoder_layers()[-1]
        expected_query = final_block.attention.q_proj(final_block.norm1(backbone_output.hidden_states[-2]))
        if not torch.allclose(qkv["q"], expected_query, rtol=1e-5, atol=1e-6):
            raise AssertionError("Extracted Q does not match the model's normalized internal Q projection")
        if qkv["q_patch"].shape[1] != (resolution // model.backbone.patch_size) ** 2:
            raise AssertionError(f"Unexpected patch token count: {qkv['q_patch'].shape}")
        if qkv["v_patch"].shape[-1] != model.backbone.hidden_dim:
            raise AssertionError("V descriptor dimension must match the backbone hidden dimension")
        features = model.extract_global(pixels)
        gallery_features = features[1:] if len(images) > 1 else features
        index = build_faiss_index(gallery_features.float().cpu().numpy())
        query_features = features[:1].float().cpu().numpy()
        scores, indices = search_faiss(index, query_features, top_k=1)
        local_descriptors = []
        for image_index in range(len(images)):
            attention_scores = attention_patch_scores(qkv["q_patch"][image_index], qkv["k"][image_index, 0])
            local_descriptors.append(qkv["v_patch"][image_index][attention_scores > 0.05])
        local_matches = mutual_nearest_neighbor_matches(
            local_descriptors[0],
            local_descriptors[1] if len(local_descriptors) > 1 else local_descriptors[0],
            threshold=0.65,
        )
        gallery_offset = 1 if len(images) > 1 else 0
        retrieved_label = dataset[gallery_offset]["label"]
        metrics = evaluate_label_retrieval(
            [str(dataset[0]["label"])],
            [[str(retrieved_label)]],
        )
    with tempfile.TemporaryDirectory(prefix="effovpr-smoke-") as temporary_dir:
        checkpoint_path = Path(temporary_dir) / "smoke.pt"
        save_checkpoint(str(checkpoint_path), {"model_state_dict": model.state_dict(), "epoch": 0})
        checkpoint = load_checkpoint(str(checkpoint_path))
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    result = {
        "status": "SMOKE TEST",
        "dataset_samples": len(dataset),
        "inference_resolution": resolution,
        "global_shape": list(global_features.shape),
        "qkv_shape": list(qkv["v"].shape),
        "patch_count": int(qkv["v_patch"].shape[1]),
        "faiss_index_size": int(index.ntotal),
        "faiss_top1_score": float(scores[0, 0]),
        "faiss_top1_index": int(indices[0, 0]),
        "local_mnn_match_count": int(local_matches.shape[0]),
        "qkv_matches_internal_projection": True,
        "label_retrieval_metrics": metrics,
        "checkpoint_round_trip": True,
        "training_loss": float(loss.detach().item()),
    }
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resolution", type=int, default=224)
    args = parser.parse_args()
    run_smoke_test(args.config, args.cache_dir, args.resolution)


if __name__ == "__main__":
    main()
