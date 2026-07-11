"""Decoder: fused context, two-stage pointer, hierarchical action heads (Sections 11–13)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.feature_specs import DYNAMIC_CANDIDATE_FEAT_DIM
from src.models.feature_specs import BATTERY_STATE_FEATURE_DIM
from src.models.mlp_blocks import FusionMLP, ResidualMLP


class DynamicCandidateEncoder(nn.Module):
    """Always-active residual MLP over per-candidate dynamic features."""

    def __init__(self, embedding_dim: int, feat_dim: int = DYNAMIC_CANDIDATE_FEAT_DIM):
        super().__init__()
        self.net = ResidualMLP(feat_dim, embedding_dim, embedding_dim)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)


class PointerDecoder(nn.Module):
    """Two-stage pointer decoder with hierarchical charge and depot heads."""

    def __init__(self, embedding_dim: int = 128, num_charging_portions: int = 4):
        super().__init__()
        d = embedding_dim
        self.embedding_dim = d
        # 7 context vectors of dimension d (Section 11).
        self.context_fusion = FusionMLP(7 * d, d)
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.dynamic_candidate_encoder = DynamicCandidateEncoder(d)
        self.dynamic_projection = nn.Linear(d, d, bias=False)
        self.candidate_fuse_norm = nn.LayerNorm(d)
        self.compatibility_mlp = nn.Sequential(
            nn.Linear(2 * d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(3 * d + BATTERY_STATE_FEATURE_DIM, d),
            nn.GELU(),
            nn.Linear(d, num_charging_portions),
        )
        self.depot_mode_head = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, 2),
        )

    def fuse_decoder_context(
        self,
        acting_ev_context: torch.Tensor,
        acting_ev_token: torch.Tensor,
        active_fleet_context: torch.Tensor,
        finished_fleet_context: torch.Tensor,
        global_context: torch.Tensor,
        mask_context: torch.Tensor,
        fleet_balance_context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([
            acting_ev_context,
            acting_ev_token,
            active_fleet_context,
            finished_fleet_context,
            global_context,
            mask_context,
            fleet_balance_context,
        ], dim=-1)
        return self.context_fusion(x)

    def fuse_candidates(
        self,
        static_node_embeddings: torch.Tensor,
        dynamic_candidate_feats: torch.Tensor,
        *,
        use_dynamic_encoder: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if use_dynamic_encoder:
            dyn_emb = self.dynamic_candidate_encoder(dynamic_candidate_feats)
            fused = self.candidate_fuse_norm(
                static_node_embeddings + self.dynamic_projection(dyn_emb)
            )
        else:
            dyn_emb = torch.zeros_like(static_node_embeddings)
            fused = static_node_embeddings
        return fused, dyn_emb

    def node_scores(
        self,
        decoder_context: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        dynamic_candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        q = self.wq(decoder_context)
        k = self.wk(candidate_embeddings)
        base = (k @ q) / math.sqrt(self.embedding_dim)
        compat_in = torch.cat([
            decoder_context.unsqueeze(0).expand(candidate_embeddings.shape[0], -1),
            dynamic_candidate_embeddings,
        ], dim=-1)
        compat = self.compatibility_mlp(compat_in).squeeze(-1)
        return base + compat

    def charging_logits(
        self,
        decoder_context: torch.Tensor,
        selected_embedding: torch.Tensor,
        acting_ev_context: torch.Tensor,
        battery_features: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([decoder_context, selected_embedding, acting_ev_context, battery_features], dim=-1)
        return self.charge_head(x)

    def depot_mode_logits(
        self,
        decoder_context: torch.Tensor,
        depot_embedding: torch.Tensor,
        global_context: torch.Tensor,
        mask_context: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([decoder_context, depot_embedding, global_context, mask_context], dim=-1)
        return self.depot_mode_head(x)
