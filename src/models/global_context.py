"""Global and mask-summary encoders (Sections 9–10)."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.models.feature_specs import GLOBAL_STATE_FEATURE_DIM, MASK_SUMMARY_FEATURE_DIM
from src.models.mlp_blocks import ResidualMLP


class GlobalStateEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, global_feature_dim: int = GLOBAL_STATE_FEATURE_DIM):
        super().__init__()
        self.net = ResidualMLP(global_feature_dim, embedding_dim, embedding_dim)

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.net(global_features)


class MaskSummaryEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, mask_feature_dim: int = MASK_SUMMARY_FEATURE_DIM):
        super().__init__()
        self.net = ResidualMLP(mask_feature_dim, embedding_dim, embedding_dim)

    def forward(self, mask_summary: torch.Tensor) -> torch.Tensor:
        return self.net(mask_summary)


# Backward alias
GlobalContext = GlobalStateEncoder

GLOBAL_FEATURE_DIM = GLOBAL_STATE_FEATURE_DIM
