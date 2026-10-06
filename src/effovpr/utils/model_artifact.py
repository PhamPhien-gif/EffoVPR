"""Complete, offline EffoVPR distribution artifacts; checkpoints remain training-only."""
from __future__ import annotations

import json
from pathlib import Path

import torch
import transformers
from PIL import Image
from safetensors.torch import load_file, save_file

from ..models.effovpr import EffoVPR
from ..models.dino_attention import attention_patch_scores
from ..reranking.mnn import mutual_nearest_neighbor_matches


def load_model_artifact(model_dir, device="cpu"):
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if config["format_version"] != 1 or config["architecture"] != "EffoVPR":
        raise ValueError("Unsupported EffoVPR artifact format")
    model = EffoVPR(**config["model"], backbone_config=config["backbone_config"],
                    processor_config=config["processor_config"])
    if model.backbone.hidden_dim != config["hidden_dim"] or model.backbone.patch_size != config["patch_size"]:
        raise ValueError("Artifact architecture metadata mismatch")
    model.load_state_dict(load_file(str(model_dir / "model.safetensors")), strict=True)
    return model.to(device).eval(), config


def prepare_image(model, image, resolution):
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("RGB")
    if resolution <= 0 or resolution % model.backbone.patch_size:
        raise ValueError("Resolution must be a positive multiple of patch size")
    image = image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
    return model.backbone.image_processor(images=[image], do_resize=False,
        do_center_crop=False, return_tensors="pt")["pixel_values"].to(next(model.parameters()).device)


@torch.inference_mode()
def verify_model_artifact(original, model_dir, image):
    restored, config = load_model_artifact(model_dir, next(original.parameters()).device)
    original.eval()
    pixels = prepare_image(original, image, config["inference"]["resolution"])
    restored_pixels = prepare_image(restored, image, config["inference"]["resolution"])
    torch.testing.assert_close(pixels, restored_pixels, rtol=0, atol=0)
    first = original.extract_global(pixels)
    second = restored.extract_global(restored_pixels)
    assert first.shape == second.shape == (1, config["model"]["global_dim"])
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)
    report = {"global_shape": list(second.shape), "max_absolute_difference": float((first-second).abs().max()),
              "cosine_similarity": float(torch.nn.functional.cosine_similarity(first, second).item())}
    settings = config["reranking"]
    a = original.extract_qkv(pixels, settings["layer"])
    b = restored.extract_qkv(restored_pixels, settings["layer"])
    for key in ("q", "k", "v", "q_patch", "k_patch", "v_patch"):
        torch.testing.assert_close(a[key], b[key], rtol=1e-5, atol=1e-6)
    scores_a = attention_patch_scores(a["q_patch"][0], a["k"][0, 0])
    scores_b = attention_patch_scores(b["q_patch"][0], b["k"][0, 0])
    mask_a, mask_b = scores_a > settings["T1"], scores_b > settings["T1"]
    assert torch.equal(mask_a, mask_b)
    facet = settings["facet"].lower() + "_patch"
    torch.testing.assert_close(a[facet][0, mask_a], b[facet][0, mask_b], rtol=1e-5, atol=1e-6)
    candidate = a[facet][0]
    matches_a = mutual_nearest_neighbor_matches(a[facet][0, mask_a], candidate, settings["T2"])
    matches_b = mutual_nearest_neighbor_matches(b[facet][0, mask_b], candidate, settings["T2"])
    assert torch.equal(matches_a, matches_b)
    report.update(local_patch_shape=list(b[facet].shape), selected_local_count=int(mask_b.sum()),
                  local_max_absolute_difference=float((a[facet]-b[facet]).abs().max()), strict_load=True,
                  mnn_matches_equal=True)
    return report


def export_best_model(model, checkpoint_path, output_dir, configuration, verification_image):
    """Load the full best state, export every parameter/buffer, and verify reload."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Export requires a full best model_state_dict; trainable-only checkpoints are not supported")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "format_version": 1, "architecture": "EffoVPR",
        "model": {"backbone_name": model.backbone.model_name, "global_dim": model.global_dim,
            "num_classes": model.num_classes, "fine_tune_last_n_layers": model.fine_tune_last_n_layers,
            "with_registers": model.backbone.with_registers,
            "cosface_scale": model.cosface.scale if model.cosface else 30.0,
            "cosface_margin": model.cosface.margin if model.cosface else 0.4},
        "backbone_config": model.backbone.model.config.to_dict(),
        "processor_config": model.backbone.image_processor.to_dict(),
        "hidden_dim": model.backbone.hidden_dim, "patch_size": model.backbone.patch_size,
        "inference": configuration["inference"], "reranking": configuration["reranking"],
        "class_mapping": checkpoint.get("class_mapping"),
        "source": {"checkpoint": Path(checkpoint_path).name,
                   "selection_metric": "minimum_mean_training_loss",
                   "best_training_loss": checkpoint.get("best_training_loss")},
        "transformers_version": transformers.__version__,
    }
    # clone also breaks shared storage, while retaining every state_dict entry.
    state = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in checkpoint["model_state_dict"].items()}
    save_file(state, str(output_dir / "model.safetensors"), metadata={"format": "pt"})
    del state, checkpoint
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    report = verify_model_artifact(model, output_dir, verification_image)
    (output_dir / "README.md").write_text(
        "# EffoVPR\n\nFull best-checkpoint weights, including frozen backbone, projection and classifier.\n"
        "Architecture lives in the project Python source. No pretrained download is needed.\n"
        f"Use transformers {transformers.__version__}, PyTorch and safetensors.\n\n"
        "```python\nfrom src.effovpr.utils.model_artifact import load_model_artifact, prepare_image\n"
        "import torch\nmodel, config = load_model_artifact('artifacts/model')\n"
        "pixels = prepare_image(model, '/path/query.jpg', config['inference']['resolution'])\n"
        "with torch.inference_mode():\n    embedding = model.extract_global(pixels)\n"
        "    qkv = model.extract_qkv(pixels, config['reranking']['layer'])\n```\n\n"
        "Use_Model.ipynb accepts query/gallery file paths, retrieves Top 10 with global cosine similarity, "
        "then reranks those candidates by local MNN count (ties retain global rank).\n"
        "Download config.json and model.safetensors as separate files; do not zip the model.\n\n"
        "Verification against best checkpoint:\n```json\n" + json.dumps(report, indent=2) + "\n```\n",
        encoding="utf-8")
    print(json.dumps({"model_dir": str(output_dir), "safetensors_exists": (output_dir / "model.safetensors").is_file(),
                      "verification": report}, indent=2))
    return report
