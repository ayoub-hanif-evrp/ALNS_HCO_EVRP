"""Optional edge-aware Transformer encoder with distance/time/energy attention bias."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from .node_encoder import NODE_FEATURE_DIM


class EdgeBiasMLP(nn.Module):
    def __init__(self, in_dim: int, num_heads: int, hidden_dim: int, per_head: bool = True):
        super().__init__()
        self.per_head = per_head
        out_dim = num_heads if per_head else 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """edge_feats [N, N, F] -> bias [heads, N, N] or [1, N, N]."""
        b = self.net(edge_feats)
        if self.per_head:
            return b.permute(2, 0, 1)
        return b.squeeze(-1).unsqueeze(0)


class EdgeAwareEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float,
        edge_in_dim: int,
        edge_hidden: int,
        edge_per_head: bool,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        assert self.head_dim * num_heads == embedding_dim

        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.edge_bias = EdgeBiasMLP(edge_in_dim, num_heads, edge_hidden, per_head=edge_per_head)
        self.dropout = nn.Dropout(dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim),
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, h: torch.Tensor, edge_feats: torch.Tensor) -> torch.Tensor:
        """h [N, d], edge_feats [N, N, F]."""
        N = h.size(0)
        q = self.q_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        k = self.k_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = self.v_proj(h).view(N, self.num_heads, self.head_dim).transpose(0, 1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.edge_bias(edge_feats)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v).transpose(0, 1).contiguous().view(N, self.embedding_dim)
        h = self.norm1(h + self.dropout(self.out_proj(ctx)))
        h = self.norm2(h + self.dropout(self.ff(h)))
        return h


class EdgeAwareNodeEncoder(nn.Module):
    """Node encoder with pairwise edge bias in self-attention."""

    def __init__(
        self,
        input_dim: int = NODE_FEATURE_DIM,
        embedding_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        edge_features: list[str] | None = None,
        edge_bias_hidden_dim: int = 64,
        edge_bias_per_head: bool = True,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.edge_features = edge_features or ["distance", "travel_time", "energy"]
        edge_in = len(self.edge_features)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.layers = nn.ModuleList([
            EdgeAwareEncoderLayer(
                embedding_dim, num_heads, ff_dim, dropout, edge_in, edge_bias_hidden_dim, edge_bias_per_head
            )
            for _ in range(num_layers)
        ])
        self._edge_cache: dict[int, torch.Tensor] = {}

    def _edge_tensor(self, instance, device: torch.device | str) -> torch.Tensor:
        key = id(instance)
        if key in self._edge_cache:
            return self._edge_cache[key].to(device)
        mats = []
        if "distance" in self.edge_features:
            mats.append(instance.distance_matrix)
        if "travel_time" in self.edge_features:
            mats.append(instance.travel_time_matrix)
        if "energy" in self.edge_features:
            mats.append(instance.energy_matrix)
        stacked = np.stack(mats, axis=-1).astype(np.float32)
        max_val = float(np.max(stacked)) if stacked.size else 1.0
        norm = stacked / max(max_val, 1e-6)
        t = torch.tensor(norm, device=device)
        self._edge_cache[key] = t.cpu()
        return t

    def encode_instance(self, instance, node_features: torch.Tensor, device: torch.device | str = "cpu") -> torch.Tensor:
        edge_feats = self._edge_tensor(instance, device)
        return self.forward(node_features, edge_feats)

    def forward(self, node_features: torch.Tensor, edge_feats: torch.Tensor | None = None) -> torch.Tensor:
        h = self.input_proj(node_features)
        if edge_feats is None:
            raise ValueError("edge_feats required for EdgeAwareNodeEncoder")
        for layer in self.layers:
            h = layer(h, edge_feats)
        return h
