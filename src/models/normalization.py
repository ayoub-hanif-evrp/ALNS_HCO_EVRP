"""Fixed/running normalizers for continuous neural input features."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureNormalizer(nn.Module):
    """Affine normalization: (x - mean) / (std + eps). Binary flags are excluded upstream."""

    def __init__(self, feature_dim: int, eps: float = 1e-6):
        super().__init__()
        self.feature_dim = feature_dim
        self.eps = eps
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("std", torch.ones(feature_dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.long))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.feature_dim:
            raise ValueError(f"expected last dim {self.feature_dim}, got {x.shape[-1]}")
        return (x - self.mean) / (self.std + self.eps)

    @torch.no_grad()
    def update_running(self, batch: torch.Tensor, momentum: float = 0.01) -> None:
        """Optional online update — disabled by default during RL training."""
        if batch.numel() == 0:
            return
        flat = batch.reshape(-1, self.feature_dim)
        b_mean = flat.mean(dim=0)
        b_std = flat.std(dim=0, unbiased=False).clamp_min(self.eps)
        if int(self.count.item()) == 0:
            self.mean.copy_(b_mean)
            self.std.copy_(b_std)
        else:
            self.mean.mul_(1 - momentum).add_(b_mean, alpha=momentum)
            self.std.mul_(1 - momentum).add_(b_std, alpha=momentum)
        self.count += 1

    def state_dict_for_checkpoint(self) -> dict:
        return {"mean": self.mean.detach().cpu(), "std": self.std.detach().cpu(), "count": self.count.detach().cpu()}

    def load_from_checkpoint(self, state: dict) -> None:
        self.mean.copy_(state["mean"])
        self.std.copy_(state["std"])
        if "count" in state:
            self.count.copy_(state["count"])
