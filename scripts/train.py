from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.effovpr.data.dataset import build_label_mapping
from src.effovpr.data.audit import compute_dataset_audit, save_dataset_audit
from src.effovpr.utils.checkpoint import load_checkpoint, save_checkpoint
from src.effovpr.utils.model_artifact import export_best_model
from src.effovpr.utils.device import get_device
from scripts.common import (
    build_model,
    get_dataset,
    get_splits,
    image_batch,
    load_config,
    records_for_split,
    set_deterministic_seed,
    write_run_metadata,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EffoVPR using VN_Attractions labels.")
    parser.add_argument("--config", default="configs/effovpr_vn_attractions.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--global-dim", type=int)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", default="")
    parser.add_argument("--checkpoint-every-steps", type=int, default=100)
    args = parser.parse_args()

    if args.checkpoint_every_steps < 1:
        raise ValueError("--checkpoint-every-steps must be positive")
    set_deterministic_seed(args.seed)
    config = load_config(args.config)
    if args.global_dim is not None:
        if args.global_dim not in {128, 256, 1024}:
            raise ValueError("--global-dim must be one of 128, 256, or 1024")
        config["model"]["global_dim"] = args.global_dim
    if args.cache_dir:
        config.setdefault("dataset", {})["cache_dir"] = str(Path(args.cache_dir) / "datasets")
        config["model"]["cache_dir"] = str(Path(args.cache_dir) / "models")
    dataset = get_dataset(config)
    audit = compute_dataset_audit(dataset)
    save_dataset_audit(audit, "artifacts")
    splits = get_splits(dataset, "artifacts/splits", seed=args.seed)
    label_mapping = build_label_mapping(dataset)
    train_records = records_for_split(dataset, splits["train"])
    if not train_records:
        raise ValueError("Training requires a non-empty training split")

    model = build_model(config, class_count=len(label_mapping))
    device = get_device()
    training = config["training"]
    batch_size = int(training["batch_size"])
    epochs = int(args.epochs or training["epochs"])
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    optimizer = AdamW(
        [
            {
                "params": [parameter for parameter in model.backbone.parameters() if parameter.requires_grad],
                "lr": float(training["backbone_lr"]),
            },
            {
                "params": list(model.global_projection.parameters()) + list(model.cosface.parameters()),
                "lr": float(training["classifier_lr"]),
            },
        ],
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    resolution = int(training["training_resolution"])
    history = []
    best_training_loss = float("inf")
    start_epoch = 1
    resume_batch = 0
    resume_order = None
    resume_losses: list[float] = []
    global_step = 0
    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(exist_ok=True)
    (Path("artifacts") / "class_mapping.json").write_text(
        json.dumps(label_mapping, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    write_run_metadata("artifacts/run_metadata.json", config, None, args.seed)

    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        resume_checkpoint = load_checkpoint(str(resume_path), map_location="cpu")
        required_state = ("optimizer_state_dict", "scheduler_state_dict", "epoch")
        missing_state = [key for key in required_state if key not in resume_checkpoint]
        if missing_state:
            raise ValueError(f"Resume checkpoint is missing required state: {missing_state}")
        if not {"model_state_dict", "trainable_state_dict"}.intersection(resume_checkpoint):
            raise ValueError("Resume checkpoint does not contain model weights")
        if resume_checkpoint.get("configuration") != config:
            raise ValueError("Resume checkpoint configuration does not match the current training configuration")
        if int(resume_checkpoint.get("random_seed", args.seed)) != args.seed:
            raise ValueError("Resume checkpoint random seed does not match --seed")
        if resume_checkpoint.get("class_mapping") != label_mapping:
            raise ValueError("Resume checkpoint class mapping does not match the current dataset")
        if "model_state_dict" in resume_checkpoint:
            model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        elif "trainable_state_dict" in resume_checkpoint:
            model_state = model.state_dict()
            expected_trainable = {
                name for name, parameter in model.named_parameters() if parameter.requires_grad
            }
            saved_trainable = set(resume_checkpoint["trainable_state_dict"])
            if saved_trainable != expected_trainable:
                raise ValueError("Resume checkpoint trainable parameters do not match this model configuration")
            model_state.update(resume_checkpoint["trainable_state_dict"])
            model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        history = resume_checkpoint.get("history", [])
        best_training_loss = float(resume_checkpoint.get("best_training_loss", float("inf")))
        start_epoch = int(resume_checkpoint["epoch"])
        resume_batch = int(resume_checkpoint.get("next_batch", 0))
        resume_order = resume_checkpoint.get("epoch_order")
        resume_losses = list(resume_checkpoint.get("epoch_losses", []))
        global_step = int(resume_checkpoint.get("global_step", 0))
        if start_epoch < 1 or start_epoch > epochs + 1:
            raise ValueError(f"Resume checkpoint epoch {start_epoch} is outside the configured 1..{epochs} range")
        if resume_order is not None:
            resume_order = torch.as_tensor(resume_order, dtype=torch.long).tolist()
            if len(resume_order) != len(train_records):
                raise ValueError("Resume checkpoint epoch order does not match the current training split")
            if resume_batch < 0 or resume_batch > (len(train_records) + batch_size - 1) // batch_size:
                raise ValueError("Resume checkpoint next_batch is outside the current training epoch")
        if "torch_rng_state" in resume_checkpoint:
            torch.set_rng_state(resume_checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state" in resume_checkpoint:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_state"])
        print(f"Resuming training at epoch {start_epoch}, batch {resume_batch}, global step {global_step}")

    def checkpoint_state(
        epoch: int,
        next_batch: int,
        epoch_order: list[int] | None,
        epoch_losses: list[float],
        best_loss: float,
        include_frozen_backbone: bool,
    ) -> dict:
        state = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "next_batch": next_batch,
            "epoch_order": torch.tensor(epoch_order, dtype=torch.long) if epoch_order is not None else None,
            "epoch_losses": epoch_losses,
            "global_step": global_step,
            "best_training_loss": best_loss,
            "configuration": config,
            "class_mapping": label_mapping,
            "random_seed": args.seed,
            "dataset": "PhamPhien/VN_Attractions",
            "history": history,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        }
        if include_frozen_backbone:
            state["model_state_dict"] = model.state_dict()
        else:
            state["trainable_state_dict"] = {
                name: parameter.detach().cpu()
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            }
        return state

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        continuing_epoch = epoch == start_epoch and resume_order is not None
        order = resume_order if continuing_epoch else torch.randperm(len(train_records)).tolist()
        losses = resume_losses.copy() if continuing_epoch else []
        first_batch = resume_batch if continuing_epoch else 0
        batch_starts = range(first_batch * batch_size, len(order), batch_size)
        total_batches = (len(order) + batch_size - 1) // batch_size
        progress = tqdm(
            batch_starts,
            total=total_batches,
            desc=f"epoch {epoch}/{epochs}",
            initial=first_batch,
        )
        for batch_number, start in enumerate(progress, start=first_batch):
            batch_indices = order[start : start + batch_size]
            records = [train_records[index] for index in batch_indices]
            pixels = image_batch(model, [record["image"] for record in records], resolution)
            labels = torch.tensor([label_mapping[record["label"]] for record in records], dtype=torch.long, device=device)
            _, _, loss = model(pixels, labels=labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            global_step += 1
            losses.append(float(loss.detach().item()))
            progress.set_postfix(loss=f"{losses[-1]:.4f}")
            if global_step % args.checkpoint_every_steps == 0:
                save_checkpoint(
                    str(checkpoint_dir / "last.pth"),
                    checkpoint_state(epoch, batch_number + 1, order, losses, best_training_loss, False),
                )
        scheduler.step()
        epoch_loss = sum(losses) / len(losses) if losses else float("nan")
        entry = {
            "epoch": epoch,
            "training_loss": epoch_loss,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(entry)
        next_epoch = epoch + 1
        latest_checkpoint = checkpoint_state(
            next_epoch,
            0,
            None,
            [],
            best_training_loss,
            False,
        )
        if epoch_loss < best_training_loss:
            best_training_loss = epoch_loss
            checkpoint = checkpoint_state(
                next_epoch, 0, None, [], best_training_loss, True
            )
            save_checkpoint(str(checkpoint_dir / "best.pth"), checkpoint)
        latest_checkpoint["best_training_loss"] = best_training_loss
        save_checkpoint(str(checkpoint_dir / "last.pth"), latest_checkpoint)
        print(json.dumps(entry, ensure_ascii=False))
        (Path("artifacts") / "training_history.json").write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        resume_batch = 0
        resume_order = None
        resume_losses = []
    best_checkpoint = checkpoint_dir / "best.pth"
    if not best_checkpoint.is_file():
        raise FileNotFoundError(
            f"No best checkpoint is available at {best_checkpoint}. "
            "Train at least one epoch before exporting the model."
        )
    verification_image = dataset[int(splits["val_queries"].iloc[0]["dataset_index"])]["image"]
    export_best_model(model, best_checkpoint, "artifacts/model", config, verification_image)
    print(f"Training complete; best training loss={best_training_loss:.6f}")


if __name__ == "__main__":
    main()
