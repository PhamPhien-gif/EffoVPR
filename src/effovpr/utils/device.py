from __future__ import annotations

import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def maybe_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def nullcontext():
    from contextlib import nullcontext as _nullcontext
    return _nullcontext()
