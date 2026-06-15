"""Encoder for the dynamic state of the currently-acting EV."""
from __future__ import annotations

import torch
import torch.nn as nn

# Length of EVRPTWEnv.ev_dynamic_features.
EV_DYNAMIC_FEATURE_DIM = 8


class EVStateEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, ev_feature_dim: int = EV_DYNAMIC_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(ev_feature_dim + embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, ev_features: torch.Tensor, current_node_embedding: torch.Tensor) -> torch.Tensor:
        """Combine dynamic EV features with the embedding of its current node.

        Args:
            ev_features: [..., ev_feature_dim].
            current_node_embedding: [..., d] embedding of the EV's current node.

        Returns:
            ev_context: [..., d].
        """
        x = torch.cat([ev_features, current_node_embedding], dim=-1)
        return self.net(x)
