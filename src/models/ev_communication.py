"""EV communication module.

Lets the acting EV perceive the state of all other agents (which EVs are still
active, which have high battery, which are near remaining customers, how many
are already used). Each EV becomes a token; a small Transformer mixes them, and
the non-acting EV embeddings are pooled into ``other_ev_context``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Length of EVRPTWEnv.ev_token_features.
EV_TOKEN_FEATURE_DIM = 6


class EVCommunication(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        token_feature_dim: int = EV_TOKEN_FEATURE_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_proj = nn.Sequential(
            nn.Linear(token_feature_dim + embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        ev_token_features: torch.Tensor,
        ev_current_node_embeddings: torch.Tensor,
        acting_idx: int,
        finished_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Produce other-EV context for the acting EV.

        Args:
            ev_token_features: [m, token_feature_dim].
            ev_current_node_embeddings: [m, d] embedding of each EV's current node.
            acting_idx: index of the acting EV in [0, m).
            finished_mask: [m] bool, True for finished EVs (excluded from pooling).

        Returns:
            other_ev_context: [d].
        """
        tokens = torch.cat([ev_token_features, ev_current_node_embeddings], dim=-1)
        tokens = self.token_proj(tokens).unsqueeze(0)  # [1, m, d]
        encoded = self.encoder(tokens).squeeze(0)  # [m, d]

        m = encoded.shape[0]
        if m == 1:
            return torch.zeros(self.embedding_dim, device=encoded.device, dtype=encoded.dtype)

        keep = torch.ones(m, dtype=torch.bool, device=encoded.device)
        keep[acting_idx] = False
        if finished_mask is not None:
            keep = keep & (~finished_mask)

        if keep.any():
            return encoded[keep].mean(dim=0)
        # All others finished: fall back to mean of all non-acting tokens.
        others = torch.ones(m, dtype=torch.bool, device=encoded.device)
        others[acting_idx] = False
        return encoded[others].mean(dim=0)
