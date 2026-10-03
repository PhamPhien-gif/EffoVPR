from __future__ import annotations

import json
import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from src.effovpr.data.dataset import load_vn_attractions
from src.effovpr.data.audit import compute_dataset_audit, save_dataset_audit
from src.effovpr.data.split import deterministic_label_split, save_split_frames, validate_split_frames
from src.effovpr.models.dino_attention import attention_patch_scores
from src.effovpr.models.effovpr import EffoVPR
from src.effovpr.utils.checkpoint import load_checkpoint
from src.effovpr.utils.device import get_device
from src.effovpr.utils.seed import set_seed


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration at {path} must contain a YAML mapping")
    return config


def get_dataset(config: dict[str, Any]):
    dataset_config = config.get("dataset", {})
    return load_vn_attractions(
        split=dataset_config.get("source_split", "train"),
        cache_dir=dataset_config.get("cache_dir"),
    )


def ensure_dataset_audit(dataset, output_dir: str | Path = "artifacts") -> dict[str, Any]:
    audit_path = Path(output_dir) / "dataset_audit.json"
    ids = [str(image_id) for image_id in dataset["id"]]
    id_digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()
    if audit_path.is_file():
        existing = json.loads(audit_path.read_text(encoding="utf-8"))
        if existing.get("sample_count") == len(dataset) and existing.get("id_sha256") == id_digest:
            return existing
    stats = compute_dataset_audit(dataset)
    save_dataset_audit(stats, str(output_dir))
    return stats


def get_splits(dataset, split_dir: str | Path, seed: int = 42) -> dict[str, pd.DataFrame]:
    split_path = Path(split_dir)
    names = ("train", "gallery", "val_queries", "test_queries")
    paths = {name: split_path / f"{name}.csv" for name in names}
    manifest_path = split_path / "split_config.json"
    dataset_identity = hashlib.sha256("\n".join(map(str, dataset["id"])).encode("utf-8")).hexdigest()
    expected_manifest = {
        "seed": seed,
        "sample_count": len(dataset),
        "id_sha256": dataset_identity,
        "ratios": {"train": 0.7, "gallery": 0.1, "val_queries": 0.1, "test_queries": 0.1},
    }
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_matches = bool(
        manifest is not None
        and all(manifest.get(key) == value for key, value in expected_manifest.items())
        and manifest.get("split_csv_sha256")
        == {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in paths.items()
            if path.is_file()
        }
    )
    if not all(path.is_file() for path in paths.values()) or not manifest_matches:
        frames = deterministic_label_split(dataset, seed=seed)
        save_split_frames(frames, split_path)
        manifest = {
            **expected_manifest,
            "split_csv_sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in paths.items()
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    else:
        frames = {name: pd.read_csv(path, dtype={"id": str, "label": str}) for name, path in paths.items()}
        validate_split_frames(frames)
        for name, frame in frames.items():
            if not {"id", "label", "dataset_index", "split"}.issubset(frame.columns):
                raise ValueError(f"Split file {paths[name]} is missing required columns")
            if not frame.empty and (frame["dataset_index"].min() < 0 or frame["dataset_index"].max() >= len(dataset)):
                raise ValueError(f"Split file {paths[name]} contains out-of-range dataset indices")
    return frames


def records_for_split_metadata(dataset, frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    for row in frame.itertuples(index=False):
        sample = dataset[int(row.dataset_index)]
        if str(sample.get("id", "")) != str(row.id) or str(sample.get("label", "")) != str(row.label):
            raise ValueError(f"Split metadata does not match dataset row {row.dataset_index}")
        records.append({
            "id": str(row.id),
            "label": str(row.label),
            "dataset_index": int(row.dataset_index),
        })
    return records


def records_for_split(dataset, frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = records_for_split_metadata(dataset, frame)
    for record in records:
        record["image"] = dataset[record["dataset_index"]]["image"]
    return records


def build_model(config: dict[str, Any], class_count: int | None = None, checkpoint_path: str | None = None, zero_shot: bool = False) -> EffoVPR:
    model_config = config["model"]
    model = EffoVPR(
        backbone_name=model_config["backbone"],
        global_dim=int(model_config.get("global_dim", 1024)),
        num_classes=None if zero_shot else class_count,
        fine_tune_last_n_layers=0 if zero_shot else int(model_config.get("fine_tune_last_n_layers", 5)),
        with_registers=bool(model_config.get("with_registers", True)),
        cosface_scale=float(model_config.get("cosface_scale", 30.0)),
        cosface_margin=float(model_config.get("cosface_margin", 0.4)),
        cache_dir=model_config.get("cache_dir"),
    )
    if checkpoint_path:
        checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
        state = checkpoint.get("model_state_dict", checkpoint.get("model_state"))
        if state is None:
            raise ValueError(f"Checkpoint {checkpoint_path} does not contain model weights")
        model.load_state_dict(state, strict=True)
    return model.to(get_device())


def image_batch(model: EffoVPR, images: list[Image.Image], resolution: int) -> torch.Tensor:
    if resolution <= 0 or resolution % model.backbone.patch_size:
        raise ValueError(f"Resolution must be a positive multiple of patch size {model.backbone.patch_size}")
    resized_images = [
        image.convert("RGB").resize(
            (resolution, resolution),
            resample=Image.Resampling.BICUBIC,
        )
        for image in images
    ]
    processed = model.backbone.image_processor(
        images=resized_images,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )
    return processed["pixel_values"].to(next(model.parameters()).device)


@torch.inference_mode()
def extract_global_features(
    model: EffoVPR,
    records: list[dict[str, Any]],
    resolution: int,
    batch_size: int,
    zero_shot: bool = False,
) -> np.ndarray:
    features: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        if any(record.get("image") is None for record in batch):
            raise ValueError("Missing image while extracting global features")
        pixels = image_batch(model, [record["image"] for record in batch], resolution)
        if zero_shot:
            output = model.backbone(pixels)
            descriptor = output.last_hidden_state[:, 0, :]
            descriptor = torch.nn.functional.normalize(descriptor, p=2, dim=-1)
        else:
            descriptor = model.extract_global(pixels)
        features.append(descriptor.float().cpu().numpy())
    if not features:
        return np.empty((0, model.backbone.hidden_dim if zero_shot else model.global_dim), dtype=np.float32)
    return np.concatenate(features).astype(np.float32, copy=False)


def load_or_extract_global_features(
    model: EffoVPR,
    records: list[dict[str, Any]],
    resolution: int,
    batch_size: int,
    split: str,
    checkpoint_path: str | None,
    config: dict[str, Any],
    zero_shot: bool = False,
) -> np.ndarray:
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else None
    checkpoint_stat = checkpoint.stat() if checkpoint and checkpoint.is_file() else None
    metadata = {
        "checkpoint": str(checkpoint) if checkpoint else "pretrained-zero-shot",
        "checkpoint_size": checkpoint_stat.st_size if checkpoint_stat else None,
        "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns if checkpoint_stat else None,
        "model_identifier": config["model"]["backbone"],
        "input_resolution": int(resolution),
        "feature_dimension": model.backbone.hidden_dim if zero_shot else model.global_dim,
        "local_feature_layer": config["reranking"]["zero_shot_layer"] if zero_shot else config["reranking"]["layer"],
        "facet": config["reranking"]["facet"],
        "T1": config["reranking"]["T1"],
        "T2": config["reranking"]["T2"],
        "split": split,
        "record_ids_sha256": hashlib.sha256(
            "\n".join(str(record["id"]) for record in records).encode("utf-8")
        ).hexdigest(),
        "zero_shot": bool(zero_shot),
        "model_revision": getattr(model.backbone.model.config, "_commit_hash", None),
    }
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    cache_dir = Path("artifacts/features/global")
    cache_dir.mkdir(parents=True, exist_ok=True)
    features_path = cache_dir / f"{split}_{key}.npy"
    metadata_path = cache_dir / f"{split}_{key}.json"
    if features_path.is_file() and metadata_path.is_file():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if cached_metadata != metadata:
            raise ValueError(f"Global feature cache metadata mismatch: {metadata_path}")
        features = np.load(features_path, allow_pickle=False)
        expected_shape = (len(records), int(metadata["feature_dimension"]))
        if features.shape != expected_shape or not np.isfinite(features).all():
            raise ValueError(f"Global feature cache has invalid shape or values: {features_path}")
        if len(features) and not np.allclose(np.linalg.norm(features, axis=1), 1.0, atol=1e-3):
            raise ValueError(f"Global feature cache contains non-normalized vectors: {features_path}")
        return features.astype(np.float32, copy=False)
    features = extract_global_features(model, records, resolution, batch_size, zero_shot=zero_shot)
    if features.shape != (len(records), int(metadata["feature_dimension"])):
        raise ValueError(f"Feature extractor returned unexpected shape {features.shape}")
    np.save(features_path, features)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return features


@torch.inference_mode()
def extract_local_descriptors(
    model: EffoVPR,
    image: Image.Image,
    resolution: int,
    layer: int | str,
    facet: str,
    threshold: float,
) -> torch.Tensor:
    facet_key = facet.lower()
    if facet_key not in {"q", "k", "v"}:
        raise ValueError("facet must be one of Q, K, or V")
    pixels = image_batch(model, [image], resolution)
    qkv = model.extract_qkv(pixels, layer=layer)
    scores = attention_patch_scores(qkv["q_patch"][0], qkv["k"][0, 0])
    keep = scores > threshold
    return qkv[f"{facet_key}_patch"][0, keep].float()


def checkpoint_identifier(path: str | None) -> str:
    return str(Path(path).resolve()) if path else "pretrained-zero-shot"


def write_run_metadata(path: str | Path, config: dict[str, Any], checkpoint: str | None, seed: int) -> dict[str, Any]:
    device = get_device()
    workspace = Path(__file__).resolve().parents[1]
    git_result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    metadata = {
        "seed": seed,
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "dataset": config.get("dataset", {}).get("identifier", "PhamPhien/VN_Attractions"),
        "dataset_revision": None,
        "backbone": config["model"]["backbone"],
        "checkpoint": checkpoint_identifier(checkpoint),
        "input_resolution": config.get("inference", {}).get("resolution"),
        "training_resolution": config.get("training", {}).get("training_resolution"),
        "global_dimension": config.get("model", {}).get("global_dim"),
        "rerank_k": config.get("reranking", {}).get("top_k"),
        "local_feature_layer": config.get("reranking", {}).get("layer"),
        "facet": config.get("reranking", {}).get("facet"),
        "T1": config.get("reranking", {}).get("T1"),
        "T2": config.get("reranking", {}).get("T2"),
        "batch_size": config.get("training", {}).get("batch_size"),
        "git_commit": git_result.stdout.strip() if git_result.returncode == 0 else None,
        "configuration": config,
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def set_deterministic_seed(seed: int) -> None:
    set_seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
