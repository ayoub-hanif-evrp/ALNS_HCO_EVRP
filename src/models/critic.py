"""Centralized fleet critic with auxiliary value heads (Section 14–16)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decision_state import DecisionState
from src.models.feature_specs import GLOBAL_STATE_FEATURE_DIM, MASK_SUMMARY_FEATURE_DIM
from src.models.mlp_blocks import ResidualMLP


class CentralFleetValueCritic(nn.Module):
    """Fleet + node + global centralized critic for semi-MDP decision states."""

    def __init__(self, embedding_dim: int = 192):
        super().__init__()
        d = embedding_dim
        self.embedding_dim = d
        self.fleet_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.customer_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.station_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.depot_attn = nn.MultiheadAttention(d, num_heads=4, batch_first=True)
        self.fleet_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.customer_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.station_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        self.depot_query = nn.Parameter(torch.randn(1, 1, d) * 0.02)

        critic_in = 8 * d
        self.value_trunk = ResidualMLP(critic_in, 2 * d, d)
        self.value_head = nn.Linear(d, 1)
        self.aux_unserved = nn.Linear(d, 1)
        self.aux_serviceable = nn.Linear(d, 1)
        self.aux_objective = nn.Linear(d, 1)
        self.aux_feasibility = nn.Linear(d, 1)

    def _pool_nodes(
        self,
        attn: nn.MultiheadAttention,
        query: torch.Tensor,
        embeddings: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Pool over node embeddings. ``mask`` True = valid key (included)."""
        if embeddings.numel() == 0:
            return torch.zeros(self.embedding_dim, device=query.device, dtype=query.dtype)
        keys = embeddings.unsqueeze(0)  # [1, N, d]
        key_padding_mask = None
        if mask is not None:
            if not bool(mask.any()):
                return torch.zeros(self.embedding_dim, device=query.device, dtype=query.dtype)
            # MultiheadAttention: True means *ignore* this key.
            key_padding_mask = (~mask).unsqueeze(0)
        out, _ = attn(query, keys, keys, key_padding_mask=key_padding_mask)
        return out.squeeze(0).squeeze(0)

    def _pool_nodes_batched(
        self,
        attn: nn.MultiheadAttention,
        query: torch.Tensor,
        embeddings: torch.Tensor,
        sel: torch.Tensor,
    ) -> torch.Tensor:
        """Batched node pooling — ``embeddings`` [B, N, d], ``sel`` [B, N] True=valid.

        Empty selections return a true zero vector (matches single-state path).
        Non-empty selections use ``key_padding_mask`` so padded/invalid tokens
        contribute zero attention probability.
        """
        B, _, d = embeddings.shape
        device, dtype = embeddings.device, embeddings.dtype
        q = query.expand(B, -1, -1)
        has_any = sel.any(dim=-1)  # [B]
        # True = ignore key in MultiheadAttention
        key_padding_mask = ~sel
        # All-invalid rows would make softmax over empty set ill-defined;
        # keep one dummy valid slot then overwrite those outputs with zeros.
        empty = ~has_any
        if bool(empty.any()):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[empty, 0] = False
        out, _ = attn(q, embeddings, embeddings, key_padding_mask=key_padding_mask)
        pooled = out.squeeze(1)
        if bool(empty.any()):
            pooled = pooled.clone()
            pooled[empty] = 0
        return pooled

    def _fleet_key_padding(self, states: list[DecisionState], fleet_tokens: torch.Tensor) -> torch.Tensor | None:
        """Finished / inactive EV tokens should not dilute fleet attention."""
        # Prefer feasible / active indicators stored on decision states if present.
        masks = []
        for s in states:
            ft = getattr(s, "fleet_active_mask", None)
            if ft is None:
                return None
            masks.append(torch.as_tensor(ft, device=fleet_tokens.device, dtype=torch.bool))
        stacked = torch.stack(masks)
        if stacked.shape[:2] != fleet_tokens.shape[:2]:
            return None
        return ~stacked  # True = pad/ignore

    def _encode_states(self, states: list[DecisionState]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Batched critic trunk — identical attention semantics to ``forward_from_state``."""
        if len(states) == 1:
            value, aux = self.forward_from_state(states[0])
            if value.dim() == 0:
                value = value.unsqueeze(0)
            aux = {k: v.unsqueeze(0) if v.dim() == 0 else v for k, v in aux.items()}
            return value, aux

        device = states[0].decoder_context.device
        acting = torch.stack([s.acting_ev_context for s in states])
        fleet_tokens = torch.stack([s.communicated_ev_tokens for s in states])
        cand = torch.stack([s.candidate_embeddings for s in states])
        dyn_feats = torch.stack([s.dynamic_candidate_feats for s in states])
        global_ctx = torch.stack([s.global_context for s in states])
        mask_ctx = torch.stack([s.mask_context for s in states])
        fleet_bal = torch.stack([s.fleet_balance_context for s in states])
        feas = torch.stack([
            torch.as_tensor(s.feasible_mask, device=device, dtype=torch.bool) for s in states
        ])

        B = len(states)
        fleet_q = self.fleet_query.expand(B, -1, -1)
        fleet_kpm = self._fleet_key_padding(states, fleet_tokens)
        fleet_ctx, _ = self.fleet_attn(
            fleet_q, fleet_tokens, fleet_tokens, key_padding_mask=fleet_kpm,
        )
        fleet_value_context = fleet_ctx.squeeze(1)

        is_customer = dyn_feats[:, :, 17] > 0.5
        is_station = dyn_feats[:, :, 18] > 0.5
        is_depot = dyn_feats[:, :, 19] > 0.5

        customer_pool = self._pool_nodes_batched(
            self.customer_attn, self.customer_query, cand, is_customer & feas,
        )
        station_pool = self._pool_nodes_batched(
            self.station_attn, self.station_query, cand, is_station & feas,
        )
        depot_pool = self._pool_nodes_batched(
            self.depot_attn, self.depot_query, cand, is_depot & feas,
        )

        parts = torch.cat([
            acting,
            fleet_value_context,
            customer_pool,
            station_pool,
            depot_pool,
            global_ctx,
            mask_ctx,
            fleet_bal,
        ], dim=-1)
        h = self.value_trunk(parts)
        value = self.value_head(h).squeeze(-1)
        aux = {
            "unserved_ratio": self.aux_unserved(h).squeeze(-1),
            "serviceable_ratio": self.aux_serviceable(h).squeeze(-1),
            "objective_estimate": self.aux_objective(h).squeeze(-1),
            "feasibility_prob": torch.sigmoid(self.aux_feasibility(h).squeeze(-1)),
        }
        return value, aux

    def forward_from_states_batch(
        self, states: list[DecisionState],
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """GPU-batched critic forward for all SMDP decisions in one trajectory."""
        if not states:
            device = next(self.parameters()).device
            z = torch.zeros(0, device=device)
            return z, {
                k: z.clone() for k in (
                    "unserved_ratio", "serviceable_ratio", "objective_estimate", "feasibility_prob",
                )
            }
        return self._encode_states(states)

    def forward_from_state(self, state: DecisionState) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        d = self.embedding_dim
        device = state.decoder_context.device
        acting = state.acting_ev_context

        fleet_tokens = state.communicated_ev_tokens.unsqueeze(0)
        fleet_ctx, _ = self.fleet_attn(self.fleet_query, fleet_tokens, fleet_tokens)
        fleet_value_context = fleet_ctx.squeeze(0).squeeze(0)

        cand = state.candidate_embeddings
        feas = torch.tensor(state.feasible_mask, device=device, dtype=torch.bool)
        is_customer = state.dynamic_candidate_feats[:, 17] > 0.5
        is_station = state.dynamic_candidate_feats[:, 18] > 0.5
        is_depot = state.dynamic_candidate_feats[:, 19] > 0.5

        def _pool(attn, query, type_mask):
            sel = type_mask & feas
            if not bool(sel.any()):
                return torch.zeros(self.embedding_dim, device=device, dtype=acting.dtype)
            return self._pool_nodes(attn, query, cand[sel], None)

        customer_pool = _pool(self.customer_attn, self.customer_query, is_customer)
        station_pool = _pool(self.station_attn, self.station_query, is_station)
        depot_pool = _pool(self.depot_attn, self.depot_query, is_depot)

        parts = torch.cat([
            acting,
            fleet_value_context,
            customer_pool,
            station_pool,
            depot_pool,
            state.global_context,
            state.mask_context,
            state.fleet_balance_context,
        ], dim=-1)
        h = self.value_trunk(parts)
        value = self.value_head(h).squeeze(-1)
        aux = {
            "unserved_ratio": self.aux_unserved(h).squeeze(-1),
            "serviceable_ratio": self.aux_serviceable(h).squeeze(-1),
            "objective_estimate": self.aux_objective(h).squeeze(-1),
            "feasibility_prob": torch.sigmoid(self.aux_feasibility(h).squeeze(-1)),
        }
        return value, aux

    def forward_decision(self, trajectory, *, detach: bool = True) -> torch.Tensor:
        values: list[torch.Tensor] = []
        device = next(self.parameters()).device
        for dec in trajectory.decision_transitions:
            state = getattr(dec, "decision_state", None)
            if state is None:
                values.append(torch.zeros((), device=device))
                continue
            if detach:
                state = state.detach_for_critic()
            v, _ = self.forward_from_state(state)
            values.append(v.squeeze() if v.dim() > 0 else v)
        if not values:
            return torch.zeros(0, device=device)
        return torch.stack(values)

    def forward_with_aux(self, state: DecisionState) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.forward_from_state(state)


def build_decision_value_features(state: DecisionState) -> dict:
    """Store critic inputs on decision transitions."""
    return {"decision_state": state.detach_for_critic()}


def auxiliary_critic_loss(
    aux_preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    *,
    coef: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    device = next(iter(aux_preds.values())).device
    total = torch.zeros((), device=device)
    parts: dict[str, float] = {}
    for key, pred in aux_preds.items():
        if key not in targets:
            continue
        pred_f = pred.float()
        tgt = targets[key].detach().float()
        if key == "feasibility_prob":
            part = F.binary_cross_entropy(pred_f, tgt)
        elif key == "objective_estimate":
            part = F.smooth_l1_loss(pred_f, tgt)
        else:
            part = F.mse_loss(pred_f, tgt)
        parts[f"aux_{key}_loss"] = float(part.detach())
        total = total + coef * part
    return total, parts


def build_trajectory_auxiliary_targets(
    trajectory,
    *,
    num_customers: int,
    terminal_feasible: bool,
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    """Per-decision auxiliary critic targets from an episode trajectory."""
    decisions = trajectory.decision_transitions
    n_cust = max(int(num_customers), 1)
    deltas = [float(d.original_cost_delta) for d in decisions]
    unserved, serviceable, remaining_cost, feasibility = [], [], [], []
    for t, dec in enumerate(decisions):
        remaining_cost.append(sum(deltas[t:]))
        mr = getattr(dec, "mask_result", None)
        if mr is not None:
            gs = mr.global_serviceability
            rem = max(int(gs.remaining_customers_count), 0)
            unserved.append(rem / n_cust)
            serviceable.append(
                float(gs.globally_serviceable_customers_count) / max(rem, 1),
            )
        else:
            unserved.append(0.0)
            serviceable.append(0.0)
        feasibility.append(float(terminal_feasible))
    return {
        "unserved_ratio": torch.tensor(unserved, dtype=torch.float32, device=device),
        "serviceable_ratio": torch.tensor(serviceable, dtype=torch.float32, device=device),
        "objective_estimate": torch.tensor(remaining_cost, dtype=torch.float32, device=device),
        "feasibility_prob": torch.tensor(feasibility, dtype=torch.float32, device=device),
    }


# Legacy stubs kept for tests that import ValueCritic / QCritic.
class ValueCritic(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.net = nn.Linear(embedding_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class QCritic(nn.Module):
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

    def forward_actions(self, decoder_context: torch.Tensor, action_embeddings: torch.Tensor) -> torch.Tensor:
        q_state = self.state_proj(decoder_context)
        q_action = self.action_proj(action_embeddings)
        fused = q_state.unsqueeze(0) * q_action + q_state.unsqueeze(0)
        return self.head(fused).squeeze(-1)

    def forward(self, decoder_context: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        q_state = self.state_proj(decoder_context)
        q_action = self.action_proj(node_embeddings)
        fused = q_state.unsqueeze(0) * q_action + q_state.unsqueeze(0)
        return self.head(fused).squeeze(-1).squeeze(0)
