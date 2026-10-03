from __future__ import annotations

import os
import tempfile
from typing import Any, Dict
from pathlib import Path

import torch


def save_checkpoint(path: str, state: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    try:
        torch.save(state, temporary_path)
        os.replace(temporary_path, destination)
    except Exception:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise


def load_checkpoint(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=True)
