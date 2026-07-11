"""Reusable residual MLP blocks for actor/critic encoders."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResidualMLP(nn.Module):
    """Linear → activation → LayerNorm → Linear with residual projection."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        *,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_cls = nn.GELU if activation.lower() == "gelu" else nn.SiLU
        self.proj_in = nn.Linear(in_dim, hidden_dim)
        self.act = act_cls()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = (
            nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dropout(self.proj_out(self.norm(self.act(self.proj_in(x)))))
        return h + self.residual(x)


class FusionMLP(nn.Module):
    """Fuse a concatenated context vector into embedding dimension d."""

    def __init__(self, in_dim: int, out_dim: int, *, dropout: float = 0.0):
        super().__init__()
        hidden = max(out_dim, in_dim // 2)
        self.net = ResidualMLP(in_dim, hidden, out_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
