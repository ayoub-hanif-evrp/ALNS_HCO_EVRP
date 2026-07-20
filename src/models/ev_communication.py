"""Inter-EV communication with masked Transformer pooling (Section 6)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from src.models.feature_specs import EV_TOKEN_SCALAR_DIM, FLEET_BALANCE_FEATURE_DIM
from src.models.mlp_blocks import ResidualMLP


@dataclass
class EVCommunicationOutput:
    communicated_tokens: torch.Tensor
    acting_token: torch.Tensor
    active_pool: torch.Tensor
    finished_pool: torch.Tensor
    fleet_balance_context: torch.Tensor


class EVCommunication(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 1,
        token_scalar_dim: int = EV_TOKEN_SCALAR_DIM,
        fleet_balance_dim: int = FLEET_BALANCE_FEATURE_DIM,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.token_proj = ResidualMLP(
            token_scalar_dim + embedding_dim,
            embedding_dim,
            embedding_dim,
            dropout=dropout,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.fleet_balance_proj = ResidualMLP(fleet_balance_dim, embedding_dim, embedding_dim)

    def forward(
        self,
        ev_token_scalars: torch.Tensor,
        ev_current_node_embeddings: torch.Tensor,
        acting_idx: int,
        finished_mask: torch.Tensor | None = None,
        fleet_balance: torch.Tensor | None = None,
    ) -> EVCommunicationOutput:
        device = ev_current_node_embeddings.device
        tokens_in = torch.cat([ev_token_scalars, ev_current_node_embeddings], dim=-1)
        tokens = self.token_proj(tokens_in).unsqueeze(0)
        m = tokens_in.shape[0]

        attn_mask = None
        if finished_mask is not None and finished_mask.any():
            # Finished EVs remain in the sequence but must not be used as keys.
            # Mask columns only — do NOT symmetrize. Symmetrizing fully masks
            # finished rows (no valid keys), which yields NaN softmax outputs and
            # poisons finished_pool → decoder_context (decoder_finite assertion).
            attn_mask = torch.zeros((m, m), dtype=torch.bool, device=device)
            for idx in range(m):
                if finished_mask[idx]:
                    attn_mask[:, idx] = True

        encoded = self.encoder(tokens, mask=attn_mask).squeeze(0)
        acting_token = encoded[acting_idx]

        active_mask = torch.ones(m, dtype=torch.bool, device=device)
        active_mask[acting_idx] = False
        if finished_mask is not None:
            active_mask = active_mask & (~finished_mask)
        finished_only = finished_mask if finished_mask is not None else torch.zeros(m, dtype=torch.bool, device=device)

        active_pool = (
            encoded[active_mask].mean(dim=0)
            if active_mask.any()
            else torch.zeros(self.embedding_dim, device=device, dtype=encoded.dtype)
        )
        finished_pool = (
            encoded[finished_only].mean(dim=0)
            if finished_only.any()
            else torch.zeros(self.embedding_dim, device=device, dtype=encoded.dtype)
        )
        # Defense in depth: never let a non-finite pool poison decoder fusion.
        if not torch.isfinite(active_pool).all():
            active_pool = torch.zeros_like(active_pool)
        if not torch.isfinite(finished_pool).all():
            finished_pool = torch.zeros_like(finished_pool)
        if fleet_balance is None:
            fleet_balance = torch.zeros(FLEET_BALANCE_FEATURE_DIM, device=device)
        fleet_balance_context = self.fleet_balance_proj(fleet_balance)

        return EVCommunicationOutput(
            communicated_tokens=encoded,
            acting_token=acting_token,
            active_pool=active_pool,
            finished_pool=finished_pool,
            fleet_balance_context=fleet_balance_context,
        )

    def legacy_other_context(self, output: EVCommunicationOutput) -> torch.Tensor:
        """Backward-compatible pooled non-acting context."""
        return output.active_pool
