"""Static Transformer encoder over all graph nodes.

The encoder sees the complete graph (depot, customers, charging stations) once.
Visited / served / reserved nodes are NOT removed; feasibility is handled later
by masks in the decoder. Embeddings are computed once per instance per rollout.
"""
from __future__ import annotations

import torch
import torch.nn as nn

# Number of static per-node input features (see NODE_FEATURE_COLUMNS).
NODE_FEATURE_DIM = 9


class NodeEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(
        self,
        node_features: torch.Tensor,
        node_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode node features.

        Args:
            node_features: [B, N, F] or [N, F].
            node_padding_mask: [B, N] bool, True at padded positions (optional).

        Returns:
            Node embeddings of shape [B, N, d] (or [N, d] if input was 2D).
        """
        squeeze = node_features.dim() == 2
        if squeeze:
            node_features = node_features.unsqueeze(0)
            if node_padding_mask is not None and node_padding_mask.dim() == 1:
                node_padding_mask = node_padding_mask.unsqueeze(0)

        h = self.input_proj(node_features)
        h = self.encoder(h, src_key_padding_mask=node_padding_mask)

        if squeeze:
            h = h.squeeze(0)
        return h
