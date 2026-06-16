#!/usr/bin/env python
"""Verify PyTorch can see your GPU before starting long Hanif training runs."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.utils.seed import describe_device, resolve_device


def main() -> None:
    print(f"PyTorch version : {torch.__version__}")
    print(f"CUDA built-in   : {torch.version.cuda or 'none (CPU-only wheel)'}")
    print(f"CUDA available  : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem = props.total_memory / (1024**3)
            print(f"  GPU {i}: {props.name} ({mem:.1f} GB)")
        dev = resolve_device("cuda")
        print(f"Training device : {describe_device(dev)}")
        # Quick tensor op smoke test
        x = torch.randn(1024, 1024, device=dev)
        y = x @ x
        print(f"Matmul smoke OK : {tuple(y.shape)} on {y.device}")
        return
    print()
    print("GPU not visible to PyTorch. Training will fall back to CPU (~1 min/step on Hanif).")
    print("Install a CUDA build, for example:")
    print("  pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126")
    print("See https://pytorch.org/get-started/locally/ for your driver/CUDA version.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
