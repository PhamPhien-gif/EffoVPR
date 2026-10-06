import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from PIL import Image
from transformers import Dinov2WithRegistersConfig, ViTImageProcessor

from src.effovpr.models.effovpr import EffoVPR
from src.effovpr.utils.model_artifact import export_best_model, load_model_artifact


def test_complete_offline_roundtrip(tmp_path):
    torch.manual_seed(42)
    model = EffoVPR(backbone_name="fixture-dinov2", global_dim=128, num_classes=3,
        fine_tune_last_n_layers=1,
        backbone_config=Dinov2WithRegistersConfig(hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, image_size=28, patch_size=14,
            num_register_tokens=4).to_dict(), processor_config=ViTImageProcessor().to_dict())
    best_state = {k: v.clone() for k, v in model.state_dict().items()}
    checkpoint = tmp_path / "best.pth"
    torch.save({"model_state_dict": best_state, "best_metric": 0.5}, checkpoint)
    # Simulate last epoch diverging from best; export must restore best, not last.
    with torch.no_grad():
        model.global_projection.proj.weight.add_(1)
    image = tmp_path / "query.png"
    Image.new("RGB", (42, 35), (20, 100, 180)).save(image)
    config = {"inference": {"resolution": 28}, "reranking": {
        "layer": "n-1", "facet": "V", "T1": 0.05, "T2": 0.65, "top_k": 10}}
    output = tmp_path / "model"
    with patch("transformers.AutoModel.from_pretrained", side_effect=AssertionError("network forbidden")), \
         patch("transformers.AutoImageProcessor.from_pretrained", side_effect=AssertionError("network forbidden")):
        report = export_best_model(model, checkpoint, output, config, image)
        (output / "README.md").unlink()
        restored, _ = load_model_artifact(output)
    assert set(p.name for p in output.iterdir()) == {"config.json", "model.safetensors"}
    for name, value in restored.state_dict().items():
        assert torch.equal(value, best_state[name]), name
    assert report["max_absolute_difference"] == 0
    assert report["local_max_absolute_difference"] == 0
    assert report["selected_local_count"] > 0
    assert report["global_shape"] == [1, 128]


def test_reject_partial_checkpoint(tmp_path):
    checkpoint = tmp_path / "last.pth"
    torch.save({"trainable_state_dict": {}}, checkpoint)
    with pytest.raises(ValueError, match="full best"):
        export_best_model(None, checkpoint, tmp_path / "model", {}, None)
