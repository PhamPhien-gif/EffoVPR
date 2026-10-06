"""Export an existing full best checkpoint without retraining."""
import argparse

from scripts.common import build_model, load_config
from src.effovpr.utils.checkpoint import load_checkpoint
from src.effovpr.utils.model_artifact import export_best_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--query-image", required=True)
    parser.add_argument("--output-dir", default="artifacts/model")
    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint = load_checkpoint(args.checkpoint)
    if "model_state_dict" not in checkpoint:
        raise ValueError("A full best model_state_dict is required")
    classifier = checkpoint["model_state_dict"].get("cosface.weight")
    class_count = classifier.shape[0] if classifier is not None else None
    del checkpoint
    model = build_model(config, class_count=class_count)
    export_best_model(model, args.checkpoint, args.output_dir, config, args.query_image)


if __name__ == "__main__":
    main()
