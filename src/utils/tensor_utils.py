"""Tensor helpers shared across modules."""
from __future__ import annotations

import torch

NEG_INF = -1e9


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
