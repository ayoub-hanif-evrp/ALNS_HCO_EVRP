"""Centralized value and action-value critics for actor-critic training."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.models.decoder import GLOBAL_FEATURE_DIM


class ValueCritic(nn.Module):
    """V(s_t) from the same decoder context as the actor pointer."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward(self, decoder_context: torch.Tensor) -> torch.Tensor:
        """Return scalar value; input [d] or [B, d]."""
        squeeze = decoder_context.dim() == 1
        if squeeze:
            decoder_context = decoder_context.unsqueeze(0)
        v = self.net(decoder_context).squeeze(-1)
        return v.squeeze(0) if squeeze else v


class QCritic(nn.Module):
    """Q(s, a) for feasible node actions (station node only; charging portion ignored initially)."""

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.state_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.action_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(embedding_dim // 2, 1),
        )

    def forward_actions(
        self,
        decoder_context: torch.Tensor,
        action_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Q-values for K candidates: action_embeddings [K, d] -> [K]."""
        if decoder_context.dim() != 1:
            raise ValueError("forward_actions expects 1D decoder_context")
        q_state = self.state_proj(decoder_context)
        q_action = self.action_proj(action_embeddings)
        fused = q_state.unsqueeze(0) * q_action + q_state.unsqueeze(0)
        return self.head(fused).squeeze(-1)

    def forward(
        self,
        decoder_context: torch.Tensor,
        node_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Q-values for all nodes: [N] or [B, N]."""
        if decoder_context.dim() == 1:
            q_state = self.state_proj(decoder_context)
            q_action = self.action_proj(node_embeddings)
            fused = q_state.unsqueeze(0) * q_action + q_state.unsqueeze(0)
            return self.head(fused).squeeze(-1).squeeze(0)

        q_state = self.state_proj(decoder_context)
        q_action = self.action_proj(node_embeddings)
        fused = q_state.unsqueeze(1) * q_action + q_state.unsqueeze(1)
        return self.head(fused).squeeze(-1)
