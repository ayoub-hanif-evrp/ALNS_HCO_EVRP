"""Reproducibility helpers."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = torch.cuda.is_available()


def resolve_device(device: str = "auto") -> torch.device:
    """Resolve a config device string into a torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "Config requests GPU (device=cuda) but PyTorch cannot use CUDA.\n"
            "You likely have the CPU-only PyTorch wheel installed.\n"
            "Fix (pick your CUDA version from https://pytorch.org/get-started/locally/):\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "Then verify CUDA with: python -c \"import torch; print(torch.cuda.is_available())\""
        )
    return torch.device(device)


def describe_device(device: torch.device | str) -> str:
    """Human-readable device string for logs (includes GPU name when on CUDA)."""
    dev = torch.device(device)
    if dev.type != "cuda":
        return "cpu"
    idx = dev.index if dev.index is not None else torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    mem_gb = torch.cuda.get_device_properties(idx).total_memory / (1024**3)
    return f"cuda — {name} ({mem_gb:.1f} GB)"
