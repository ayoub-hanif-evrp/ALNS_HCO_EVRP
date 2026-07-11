"""Encoder for the dynamic state of the currently-acting EV (Section 5)."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.feature_specs import ACTING_EV_FEATURE_DIM
from src.models.mlp_blocks import ResidualMLP


class EVStateEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, ev_feature_dim: int = ACTING_EV_FEATURE_DIM):
        super().__init__()
        self.net = ResidualMLP(
            ev_feature_dim + embedding_dim,
            embedding_dim,
            embedding_dim,
        )

    def forward(self, ev_features: torch.Tensor, current_node_embedding: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ev_features, current_node_embedding], dim=-1)
        return self.net(x)
