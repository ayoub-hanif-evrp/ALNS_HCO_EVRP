"""Complete decision-state representation for actor and critic."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from src.env.action_masks import ActionMaskResult


@dataclass
class DecisionState:
    """All tensors/features at one policy decision epoch."""

    decoder_context: torch.Tensor
    acting_ev_context: torch.Tensor
    communicated_ev_tokens: torch.Tensor
    acting_ev_token: torch.Tensor
    active_fleet_context: torch.Tensor
    finished_fleet_context: torch.Tensor
    fleet_balance_context: torch.Tensor
    global_context: torch.Tensor
    mask_context: torch.Tensor
    static_node_embeddings: torch.Tensor
    candidate_embeddings: torch.Tensor
    dynamic_candidate_embeddings: torch.Tensor
    dynamic_candidate_feats: torch.Tensor
    global_features_raw: torch.Tensor
    mask_summary_raw: torch.Tensor
    fleet_balance_raw: torch.Tensor
    feasible_mask: np.ndarray
    mask_result: ActionMaskResult

    def detach_for_critic(self) -> "DecisionState":
        """Return a detached copy for critic forward passes."""
        return DecisionState(
            decoder_context=self.decoder_context.detach(),
            acting_ev_context=self.acting_ev_context.detach(),
            communicated_ev_tokens=self.communicated_ev_tokens.detach(),
            acting_ev_token=self.acting_ev_token.detach(),
            active_fleet_context=self.active_fleet_context.detach(),
            finished_fleet_context=self.finished_fleet_context.detach(),
            fleet_balance_context=self.fleet_balance_context.detach(),
            global_context=self.global_context.detach(),
            mask_context=self.mask_context.detach(),
            static_node_embeddings=self.static_node_embeddings.detach(),
            candidate_embeddings=self.candidate_embeddings.detach(),
            dynamic_candidate_embeddings=self.dynamic_candidate_embeddings.detach(),
            dynamic_candidate_feats=self.dynamic_candidate_feats.detach(),
            global_features_raw=self.global_features_raw.detach(),
            mask_summary_raw=self.mask_summary_raw.detach(),
            fleet_balance_raw=self.fleet_balance_raw.detach(),
            feasible_mask=self.feasible_mask.copy(),
            mask_result=self.mask_result,
        )
