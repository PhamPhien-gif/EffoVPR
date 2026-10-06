"""Offline smoke verification with a SMALL RANDOM fixture, never trained G/R weights."""
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import Dinov2WithRegistersConfig, ViTImageProcessor

from src.effovpr.models.effovpr import EffoVPR
from src.effovpr.utils.model_artifact import export_best_model


def main():
    import matplotlib
    matplotlib.use("Agg")
    torch.manual_seed(42)
    root = Path("artifacts/verification_fixture").resolve()
    root.mkdir(parents=True, exist_ok=True)
    query = root / "query.png"
    Image.new("RGB", (42, 35), (20, 100, 180)).save(query)
    gallery = root / "gallery"
    gallery.mkdir(exist_ok=True)
    for i in range(10):
        Image.new("RGB", (42, 35), (20 + i * 10, 100, 180)).save(gallery / f"{i}.png")
    model = EffoVPR(backbone_name="random-test-fixture", global_dim=128, num_classes=3,
        fine_tune_last_n_layers=1,
        backbone_config=Dinov2WithRegistersConfig(hidden_size=32, num_hidden_layers=2,
            num_attention_heads=4, intermediate_size=64, image_size=28, patch_size=14,
            num_register_tokens=4).to_dict(), processor_config=ViTImageProcessor().to_dict())
    checkpoint = root / "best.pth"
    torch.save({"model_state_dict": model.state_dict()}, checkpoint)
    config = {"inference": {"resolution": 28}, "reranking": {
        "layer": "n-1", "facet": "V", "T1": 0.05, "T2": 0.65, "top_k": 10}}
    report = export_best_model(model, checkpoint, root / "model", config, query)
    # Execute the actual Use_Model inference cells against filesystem inputs.
    namespace = {"Path": Path, "json": json, "WORK_DIR": root,
        "MODEL_DIR": str(root / "model"), "QUERY_IMAGE_PATH": str(query),
        "GALLERY_SOURCE": "folder", "GALLERY_DIR": str(gallery),
        "GLOBAL_BATCH_SIZE": 4, "RERANK_TOP_K": 10, "SEED": 42,
        "display": lambda value: None}
    notebook = json.loads(Path("Use_Model.ipynb").read_text(encoding="utf-8"))
    started = False
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if source.startswith("import torch\nimport torch.nn.functional"):
            started = True
        if started and cell["cell_type"] == "code":
            exec(compile(source, "Use_Model.ipynb", "exec"), namespace)
    assert len(namespace["global_candidates"]) == len(namespace["reranked_candidates"]) == 10
    report.update(fixture_only=True, notebook_top10_verified=True, query_image=str(query))
    (root / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    for file in sorted((root / "model").iterdir()):
        print(file, file.stat().st_size)


if __name__ == "__main__":
    main()
