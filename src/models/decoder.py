"""Decoder: global context, pointer head over nodes, and charging-portion head."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.env.tw_diagnostics import DYNAMIC_ACTION_FEAT_DIM

# Length of EVRPTWEnv.global_features.
GLOBAL_FEATURE_DIM = 9


class GlobalContext(nn.Module):
    def __init__(self, embedding_dim: int = 128, global_feature_dim: int = GLOBAL_FEATURE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_feature_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.net(global_features)


class PointerDecoder(nn.Module):
    """Scores all nodes via scaled dot-product attention and a charging head."""

    def __init__(self, embedding_dim: int = 128, num_charging_portions: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.context_proj = nn.Sequential(
            nn.Linear(3 * embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.wk = nn.Linear(embedding_dim, embedding_dim, bias=False)

        # Charging-portion head: context + station embedding + 2 scalars.
        self.charge_head = nn.Sequential(
            nn.Linear(2 * embedding_dim + 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, num_charging_portions),
        )
        self.depot_mode_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 2),
        )
        self.dynamic_proj = nn.Linear(DYNAMIC_ACTION_FEAT_DIM, 1)

    def dynamic_score_bias(self, dynamic_feats: torch.Tensor) -> torch.Tensor:
        """Per-node temporal feasibility bias [N]."""
        return self.dynamic_proj(dynamic_feats).squeeze(-1)

    def decoder_context(
        self,
        ev_context: torch.Tensor,
        other_ev_context: torch.Tensor,
        global_context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([ev_context, other_ev_context, global_context], dim=-1)
        return self.context_proj(x)

    def node_scores(self, decoder_context: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute raw (unmasked) pointer scores for all nodes.

        Args:
            decoder_context: [d].
            node_embeddings: [N, d].

        Returns:
            scores: [N].
        """
        q = self.wq(decoder_context)  # [d]
        k = self.wk(node_embeddings)  # [N, d]
        scores = (k @ q) / math.sqrt(self.embedding_dim)
        return scores

    def charging_logits(
        self,
        decoder_context: torch.Tensor,
        station_embedding: torch.Tensor,
        battery_norm: float,
        time_norm: float,
    ) -> torch.Tensor:
        device = decoder_context.device
        scalars = torch.tensor([battery_norm, time_norm], device=device, dtype=decoder_context.dtype)
        x = torch.cat([decoder_context, station_embedding, scalars], dim=-1)
        return self.charge_head(x)

    def depot_mode_logits(self, decoder_context: torch.Tensor) -> torch.Tensor:
        """Logits for depot sub-mode: [recharge, finish]."""
        return self.depot_mode_head(decoder_context)
