from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.effovpr.models.dino_attention import attention_patch_scores
from src.effovpr.reranking.mnn import mutual_nearest_neighbor_matches
from src.effovpr.utils.model_artifact import load_model_artifact, prepare_image

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "Models"
DATASET_ID = "PhamPhien/VN_Attractions"
DATASET_SPLIT = "train"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
SIMILARITY_SOFTMAX_TEMPERATURE = 0.05

st.set_page_config(page_title="EffoVPR G/R Comparison", page_icon="??", layout="wide")
st.title("EffoVPR_G vs EffoVPR_R")
st.caption("Run the same query images through both trained models and compare place predictions and retrieval results.")


def find_named_artifacts() -> dict[str, Path]:
    found: dict[str, Path] = {}
    if not MODELS_DIR.is_dir():
        return found
    for config_path in MODELS_DIR.rglob("config.json"):
        candidate = config_path.parent
        name = candidate.name.upper()
        if name not in {"EFFOVPR_G", "EFFOVPR_R"}:
            continue
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if config.get("architecture") == "EffoVPR" and (candidate / "model.safetensors").is_file():
            found[name] = candidate
    return found


@st.cache_resource(show_spinner="Loading EffoVPR model...")
def cached_model(model_dir: str, device: str, weight_size: int, weight_mtime: int):
    return load_model_artifact(model_dir, device)


@st.cache_data(show_spinner=False)
def cached_gallery_bundle(index_dir: str, feature_size: int, feature_mtime: int):
    folder = Path(index_dir)
    index_config = json.loads((folder / "index_config.json").read_text(encoding="utf-8"))
    features = np.load(folder / "gallery_features.npy", allow_pickle=False).astype(np.float32)
    metadata = pd.read_parquet(folder / "gallery_metadata.parquet")
    return features, metadata, index_config


@st.cache_resource(show_spinner="Loading Hugging Face gallery images...")
def cached_hf_gallery(dataset_id: str, split: str):
    from datasets import load_dataset
    return load_dataset(dataset_id, split=split)


def get_index_dir(model_dir: Path) -> Path | None:
    direct = model_dir / "index"
    candidates = [direct, *(path.parent for path in model_dir.rglob("gallery_features.npy"))]
    for folder in candidates:
        if all((folder / file).is_file() for file in (
            "gallery_features.npy", "gallery_metadata.parquet", "index_config.json"
        )):
            return folder
    return None


def load_and_validate_index(model_dir: Path, artifact: dict):
    index_dir = get_index_dir(model_dir)
    if index_dir is None:
        raise FileNotFoundError(f"No gallery bundle under {model_dir / 'index'}")
    stat = (index_dir / "gallery_features.npy").stat()
    features, metadata, config = cached_gallery_bundle(str(index_dir), stat.st_size, stat.st_mtime_ns)
    expected = {
        "feature_dimension": int(artifact["model"]["global_dim"]),
        "input_resolution": int(artifact["inference"]["resolution"]),
        "model_identifier": artifact["model"]["backbone_name"],
        "count": len(metadata),
        "normalized": True,
        "index_type": "IndexFlatIP",
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"Index config mismatch: {mismatches}")
    required = {"image_id", "label", "dataset_index"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Gallery metadata missing columns: {sorted(missing)}")
    if features.shape != (len(metadata), expected["feature_dimension"]):
        raise ValueError(f"Invalid gallery feature shape: {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError("Gallery features contain non-finite values")
    if np.any(np.abs(np.linalg.norm(features, axis=1) - 1.0) > 1e-3):
        raise ValueError("Gallery features are not L2-normalized")
    return features, metadata, config


@torch.inference_mode()
def global_embedding(model, image: Image.Image, resolution: int) -> torch.Tensor:
    return model.extract_global(prepare_image(model, image, resolution)).float()


@torch.inference_mode()
def predict_label(model, artifact: dict, embedding: torch.Tensor) -> tuple[str, float]:
    mapping = artifact.get("class_mapping") or {}
    if model.cosface is None or not mapping:
        return "Classifier label unavailable", float("nan")
    id_to_label = {int(class_id): label for label, class_id in mapping.items()}
    weights = F.normalize(model.cosface.weight.float(), dim=1)
    scores = F.normalize(embedding.float(), dim=1) @ weights.T
    class_id = int(scores.argmax(dim=1).item())
    return id_to_label.get(class_id, f"class_{class_id}"), float(scores[0, class_id].item())


def relative_probabilities(similarities: np.ndarray) -> np.ndarray:
    """Convert ranking scores into relative percentages within the supplied candidate set."""
    logits = np.asarray(similarities, dtype=np.float64) / SIMILARITY_SOFTMAX_TEMPERATURE
    logits -= logits.max()
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum()


def rank_labels(similarities: np.ndarray, metadata: pd.DataFrame) -> list[dict]:
    """Rank labels by the mean similarity of their three strongest gallery images."""
    scores = pd.DataFrame({
        "label": metadata["label"].astype(str).to_numpy(),
        "cosine_similarity": similarities.astype(np.float64),
        "image_id": metadata["image_id"].astype(str).to_numpy(),
        "dataset_index": metadata["dataset_index"].to_numpy(dtype=np.int64),
    })
    top_images = scores.sort_values("cosine_similarity", ascending=False).groupby("label", sort=False).head(3)
    label_scores = top_images.groupby("label", sort=False)["cosine_similarity"].mean()
    representatives = scores.loc[scores.groupby("label")["cosine_similarity"].idxmax()].set_index("label")
    label_scores = label_scores.sort_values(ascending=False)
    probabilities = relative_probabilities(label_scores.to_numpy())
    return [
        {
            "label": label,
            "aggregated_similarity": float(score),
            "relative_probability_pct": float(probabilities[rank] * 100),
            "representative_image_id": str(representatives.loc[label, "image_id"]),
            "representative_similarity": float(representatives.loc[label, "cosine_similarity"]),
            "dataset_index": int(representatives.loc[label, "dataset_index"]),
        }
        for rank, (label, score) in enumerate(label_scores.items())
    ]


@torch.inference_mode()
def local_descriptors(model, image: Image.Image, resolution: int, rerank: dict) -> torch.Tensor:
    qkv = model.extract_qkv(prepare_image(model, image, resolution), layer=rerank["layer"])
    scores = attention_patch_scores(qkv["q_patch"][0], qkv["k"][0, 0])
    selected = scores > float(rerank["T1"])
    return qkv[f"{str(rerank['facet']).lower()}_patch"][0, selected].float()


def read_queries(mode: str, local_selection: list[str], uploads) -> list[tuple[str, Image.Image]]:
    items = []
    if mode == "Images folder":
        for relative in local_selection:
            try:
                with Image.open(ROOT / relative) as image:
                    items.append((relative, image.convert("RGB")))
            except Exception as exc:
                st.warning(f"Could not read {relative}: {exc}")
    else:
        for uploaded in uploads or []:
            try:
                uploaded.seek(0)
                with Image.open(uploaded) as image:
                    items.append((uploaded.name, image.convert("RGB")))
            except Exception as exc:
                st.warning(f"Could not read {uploaded.name}: {exc}")
    return items


def gallery_matches_dataset(metadata: pd.DataFrame, dataset) -> bool:
    info = dataset.select_columns(["id", "label"])
    source_ids = np.asarray(info["id"], dtype=str)
    source_labels = np.asarray(info["label"], dtype=str)
    row_ids = metadata["dataset_index"].to_numpy(dtype=np.int64)
    if np.any(row_ids < 0) or np.any(row_ids >= len(info)):
        return False
    return (
        np.array_equal(source_ids[row_ids], metadata["image_id"].astype(str).to_numpy())
        and np.array_equal(source_labels[row_ids], metadata["label"].astype(str).to_numpy())
    )


artifacts = find_named_artifacts()
missing_models = {"EFFOVPR_G", "EFFOVPR_R"}.difference(artifacts)
if missing_models:
    st.error(f"Missing model artifact(s) under {MODELS_DIR}: {', '.join(sorted(missing_models))}")
    st.stop()

with st.sidebar:
    st.header("Comparison settings")
    source_mode = st.radio("Query image source", ("Images folder", "Upload images"))
    local_selection = []
    uploads = []
    if source_mode == "Images folder":
        local_options = sorted(
            str(path.relative_to(ROOT))
            for path in (ROOT / "Images").rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ) if (ROOT / "Images").is_dir() else []
        if local_options:
            local_selection = st.multiselect(
                "Choose query images", local_options, default=local_options[: min(2, len(local_options))]
            )
        else:
            st.info("No supported images found in Images/. Select Upload images instead.")
    else:
        uploads = st.file_uploader(
            "Upload query images", type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )
    use_mnn = st.checkbox("Rerank each Top 10 with local MNN", value=False)
    display_image_top_n = st.selectbox("Similar images to display", (1, 5, 10), index=2)
    display_label_top_n = st.selectbox("Likely locations to display", (1, 5, 10), index=2)
    show_thumbnails = st.checkbox("Load gallery thumbnails from Hugging Face", value=True)
    run = st.button("Compare EffoVPR_G and EffoVPR_R", type="primary", use_container_width=True)

if not run:
    st.info("Choose query images and run the comparison. Both models will process the same images.")
    st.stop()

queries = read_queries(source_mode, local_selection, uploads)
if not queries:
    st.warning("Select at least one readable query image.")
    st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}
index_bundles = {}
for name in ("EFFOVPR_G", "EFFOVPR_R"):
    model_dir = artifacts[name]
    weight_path = model_dir / "model.safetensors"
    try:
        model, artifact = cached_model(
            str(model_dir), device, weight_path.stat().st_size, weight_path.stat().st_mtime_ns
        )
        loaded_models[name] = (model, artifact)
        index_bundles[name] = load_and_validate_index(model_dir, artifact)
    except Exception as exc:
        st.error(f"Could not load {name}: {exc}")
        st.stop()

st.caption(f"Inference device: {device.upper()}. Both models use their own gallery feature vectors and metadata.")

hf_gallery = None
if show_thumbnails or use_mnn:
    try:
        with st.spinner("Loading VN_Attractions gallery..."):
            hf_gallery = cached_hf_gallery(DATASET_ID, DATASET_SPLIT)
        if not all(gallery_matches_dataset(index_bundles[name][1], hf_gallery) for name in loaded_models):
            st.warning("One or both gallery metadata files do not match the current Hugging Face dataset. IDs/scores remain available, but thumbnails and MNN are disabled.")
            hf_gallery = None
    except Exception as exc:
        st.warning(f"Could not load the Hugging Face gallery: {exc}. Labels, IDs, and scores will still be compared.")
        hf_gallery = None

for query_name, query_image in queries:
    st.divider()
    st.subheader(query_name)
    st.image(query_image, width=260)
    model_results = {}
    for name in ("EFFOVPR_G", "EFFOVPR_R"):
        model, artifact = loaded_models[name]
        features, metadata, _ = index_bundles[name]
        resolution = int(artifact["inference"]["resolution"])
        rerank_config = artifact["reranking"]
        embedding = global_embedding(model, query_image, resolution)
        label, class_score = predict_label(model, artifact, embedding)
        similarities = features @ embedding[0].detach().cpu().numpy().astype(np.float32)
        label_candidates = rank_labels(similarities, metadata)
        top_k = min(10, len(similarities))
        indices = np.argpartition(-similarities, top_k - 1)[:top_k]
        indices = indices[np.argsort(-similarities[indices])]
        candidates = []
        query_local = local_descriptors(model, query_image, resolution, rerank_config) if use_mnn and hf_gallery is not None else None
        for global_rank, index in enumerate(indices, start=1):
            row = metadata.iloc[int(index)]
            item = {
                "final_rank": global_rank,
                "image_id": str(row["image_id"]),
                "label": str(row["label"]),
                "cosine_similarity": float(similarities[index]),
                "global_rank": global_rank,
                "mnn_matches": None,
                "dataset_index": int(row["dataset_index"]),
            }
            if use_mnn and hf_gallery is not None:
                candidate_image = hf_gallery[item["dataset_index"]]["image"].convert("RGB")
                candidate_local = local_descriptors(model, candidate_image, resolution, rerank_config)
                matches = mutual_nearest_neighbor_matches(
                    query_local, candidate_local, threshold=float(rerank_config["T2"])
                )
                item["mnn_matches"] = int(matches.shape[0])
            candidates.append(item)
        if use_mnn and hf_gallery is not None:
            candidates.sort(key=lambda item: (-item["mnn_matches"], item["global_rank"]))
            for rank, item in enumerate(candidates, start=1):
                item["final_rank"] = rank
        model_results[name] = {
            "label": label,
            "score": class_score,
            "candidates": candidates,
            "label_candidates": label_candidates,
        }

    left, right = st.columns(2, gap="large")
    for column, name in zip((left, right), ("EFFOVPR_G", "EFFOVPR_R")):
        result = model_results[name]
        with column:
            st.markdown(f"### {name}")
            st.caption("CosFace classifier prediction")
            st.success(result["label"])
            if np.isfinite(result["score"]):
                st.caption(f"Classifier cosine score: {result['score']:.4f} (not a probability)")
            st.subheader("Most similar images")
            st.caption(
                f"Top {min(display_image_top_n, len(result['candidates']))} individual gallery images, ranked by cosine similarity"
                + (" and reranked by MNN within the Top 10." if use_mnn and hf_gallery is not None else ".")
            )
            displayed_images = result["candidates"][:display_image_top_n]
            image_table = pd.DataFrame([
                {
                    "rank": item["final_rank"],
                    "image_id": item["image_id"],
                    "label": item["label"],
                    "cosine_similarity": item["cosine_similarity"],
                    "mnn_matches": item["mnn_matches"],
                }
                for item in displayed_images
            ])
            st.dataframe(image_table, hide_index=True, use_container_width=True)

            st.subheader("Most likely locations")
            st.caption(
                "Labels are ranked by the mean cosine similarity of their best 3 gallery images. "
                f"Softmax (temperature {SIMILARITY_SOFTMAX_TEMPERATURE:.2f}) gives relative percentages across all gallery labels; these are not calibrated probabilities."
            )
            displayed_labels = result["label_candidates"][:display_label_top_n]
            label_table = pd.DataFrame([
                {
                    "rank": rank,
                    "label": item["label"],
                    "relative_confidence_%": item["relative_probability_pct"],
                    "mean_top3_similarity": item["aggregated_similarity"],
                    "best_image_id": item["representative_image_id"],
                    "best_image_similarity": item["representative_similarity"],
                }
                for rank, item in enumerate(displayed_labels, start=1)
            ])
            st.dataframe(label_table, hide_index=True, use_container_width=True)
            if hf_gallery is not None and show_thumbnails:
                st.caption("Gallery examples for the displayed similar images")
                for start in range(0, len(displayed_images), 2):
                    image_columns = column.columns(2)
                    for image_column, item in zip(image_columns, displayed_images[start : start + 2]):
                        image = hf_gallery[item["dataset_index"]]["image"].convert("RGB")
                        caption = f"#{item['final_rank']} {item['label']}\nID: {item['image_id']}\ncos: {item['cosine_similarity']:.3f}"
                        if item["mnn_matches"] is not None:
                            caption += f" | MNN: {item['mnn_matches']}"
                        image_column.image(image, caption=caption, use_container_width=True)
                st.caption("Best matching gallery examples for the displayed locations")
                for start in range(0, len(displayed_labels), 2):
                    image_columns = column.columns(2)
                    for image_column, item in zip(image_columns, displayed_labels[start : start + 2]):
                        image = hf_gallery[item["dataset_index"]]["image"].convert("RGB")
                        caption = (
                            f"{item['label']} · {item['relative_probability_pct']:.1f}%"
                            f"\nBest image cos: {item['representative_similarity']:.3f}"
                        )
                        image_column.image(image, caption=caption, use_container_width=True)
