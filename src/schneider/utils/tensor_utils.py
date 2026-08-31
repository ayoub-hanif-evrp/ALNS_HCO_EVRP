"""Tensor helpers shared across modules."""
from __future__ import annotations

import torch

NEG_INF = -1e9


def safe_mask_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mask in float32 so AMP half precision does not overflow on ``NEG_INF``."""
    return logits.float().masked_fill(~mask, NEG_INF)


def masked_log_softmax(logits: torch.Tensor, feasible_mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax that assigns ~zero probability to infeasible entries.

    Args:
        logits: raw scores, shape [..., N].
        feasible_mask: bool tensor, True where selectable, shape [..., N].
    """
    masked = logits.masked_fill(~feasible_mask, NEG_INF)
    return torch.log_softmax(masked, dim=-1)


def masked_softmax(logits: torch.Tensor, feasible_mask: torch.Tensor) -> torch.Tensor:
    """Softmax over feasible entries only."""
    masked = logits.masked_fill(~feasible_mask, NEG_INF)
    return torch.softmax(masked, dim=-1)
