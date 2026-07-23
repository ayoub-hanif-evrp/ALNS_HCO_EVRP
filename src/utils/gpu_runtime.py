"""CUDA runtime helpers — AMP, compile, cudnn / TF32 tuning."""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn

LOGGER = logging.getLogger(__name__)


def configure_cuda_runtime(
    *,
    cudnn_benchmark: bool = True,
    allow_tf32: bool = True,
) -> None:
    """Enable GPU-friendly cudnn / TF32 heuristics when CUDA is available."""
    if not torch.cuda.is_available():
        return
    if cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    if allow_tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass
    LOGGER.info(
        "CUDA runtime ready | device=%s cudnn_benchmark=%s tf32=%s",
        torch.cuda.get_device_name(0),
        cudnn_benchmark,
        allow_tf32,
    )


def limit_cpu_threads_for_worker(n: int = 1) -> None:
    """Pin BLAS/OpenMP threads so ProcessPool workers do not oversubscribe cores."""
    n = max(1, int(n))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[key] = str(n)
    try:
        torch.set_num_threads(n)
    except Exception:
        pass


def maybe_compile_module(module: nn.Module, *, enabled: bool = True) -> nn.Module:
    """Wrap ``module`` with ``torch.compile`` when supported on CUDA."""
    if not enabled or not torch.cuda.is_available():
        return module
    if not hasattr(torch, "compile"):
        LOGGER.info("torch.compile unavailable — skipping model compile")
        return module
    try:
        compiled = torch.compile(module, mode="reduce-overhead")
        LOGGER.info("torch.compile enabled (%s)", type(module).__name__)
        return compiled
    except Exception as exc:
        LOGGER.warning("torch.compile failed for %s: %s", type(module).__name__, exc)
        return module


def autocast_ctx(device: torch.device | str, *, enabled: bool = True):
    """Context manager for mixed-precision forward passes."""
    dev = torch.device(device) if isinstance(device, str) else device
    use = bool(enabled and dev.type == "cuda")
    return torch.autocast(device_type=dev.type, enabled=use, dtype=torch.float16)
