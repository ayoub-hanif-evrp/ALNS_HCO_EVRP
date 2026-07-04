"""Transformer multi-agent policy orchestrating encoder + event-driven decoder."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.utils.tensor_utils import NEG_INF

from .decoder import GlobalContext, PointerDecoder
from .ev_communication import EVCommunication
from .ev_state_encoder import EVStateEncoder
from .edge_encoder import EdgeAwareNodeEncoder
from .node_encoder import NodeEncoder


@dataclass
class ActionOutput:
    node: int
    log_prob: torch.Tensor  # node + charging log-prob (sum), differentiable
    entropy: torch.Tensor
    charge_portion: float | None = None
    node_probs: torch.Tensor | None = None
    decoder_context: torch.Tensor | None = None


class TransformerEVRPTWPolicy(nn.Module):
    def __init__(self, model_cfg: dict, charging_portions: list[float] | None = None):
        super().__init__()
        d = int(model_cfg.get("embedding_dim", 128))
        heads = int(model_cfg.get("heads", 8))
        layers = int(model_cfg.get("encoder_layers", 3))
        ff = int(model_cfg.get("feedforward_dim", 512))
        dropout = float(model_cfg.get("dropout", 0.1))
        self.use_ev_communication = bool(model_cfg.get("use_ev_communication", True))
        self.use_edge_aware_encoder = bool(model_cfg.get("use_edge_aware_encoder", False))
        self.charging_portions = charging_portions or [0.25, 0.50, 0.75, 1.0]
        if self.use_edge_aware_encoder:
            self.node_encoder = EdgeAwareNodeEncoder(
                embedding_dim=d,
                num_heads=heads,
                num_layers=layers,
                ff_dim=ff,
                dropout=dropout,
                edge_features=list(model_cfg.get("edge_features", ["distance", "travel_time", "energy"])),
                edge_bias_hidden_dim=int(model_cfg.get("edge_bias_hidden_dim", 64)),
                edge_bias_per_head=bool(model_cfg.get("edge_bias_per_head", True)),
            )
        else:
            self.node_encoder = NodeEncoder(
                embedding_dim=d,
                num_heads=heads,
                num_layers=layers,
                ff_dim=ff,
                dropout=dropout,
            )
        self.ev_state_encoder = EVStateEncoder(embedding_dim=d)
        self.ev_communication = EVCommunication(embedding_dim=d, dropout=dropout)
        self.global_context = GlobalContext(embedding_dim=d)
        self.pointer = PointerDecoder(embedding_dim=d, num_charging_portions=len(self.charging_portions))
        self.embedding_dim = d

    # ------------------------------------------------------------------ #
    def encode(self, instance, device: torch.device | str = "cpu") -> torch.Tensor:
        """Encode all nodes of an instance once; return H [N, d]."""
        feats = torch.tensor(instance.node_feature_matrix(), device=device)
        if self.use_edge_aware_encoder:
            return self.node_encoder.encode_instance(instance, feats, device=device)
        return self.node_encoder(feats)

    def decoder_context(self, env, ev, node_embeddings: torch.Tensor) -> torch.Tensor:
        """Public accessor for centralized critic input."""
        return self._contexts(env, ev, node_embeddings)

    # ------------------------------------------------------------------ #
    def _contexts(self, env, ev, node_embeddings: torch.Tensor) -> torch.Tensor:
        device = node_embeddings.device
        ev_feats = torch.tensor(env.ev_dynamic_features(ev), device=device)
        h_current = node_embeddings[ev.current_node]
        ev_context = self.ev_state_encoder(ev_feats, h_current)

        if self.use_ev_communication and len(env.evs) > 1:
            token_feats = torch.tensor(
                np.stack([env.ev_token_features(e) for e in env.evs]), device=device
            )
            ev_node_embs = node_embeddings[[e.current_node for e in env.evs]]
            finished = torch.tensor([e.finished for e in env.evs], dtype=torch.bool, device=device)
            acting_idx = [e.ev_id for e in env.evs].index(ev.ev_id)
            other_ctx = self.ev_communication(token_feats, ev_node_embs, acting_idx, finished)
        else:
            other_ctx = torch.zeros(self.embedding_dim, device=device)

        global_feats = torch.tensor(env.global_features(), device=device)
        global_ctx = self.global_context(global_feats)

        return self.pointer.decoder_context(ev_context, other_ctx, global_ctx)

    # ------------------------------------------------------------------ #
    def select_action(
        self,
        env,
        ev,
        node_embeddings: torch.Tensor,
        feasible_mask: np.ndarray,
        mode: str = "sampling",
        *,
        depot_logit_penalty: float = 0.0,
        customer_logit_bonus: float = 0.0,
        station_logit_penalty: float = 0.0,
        use_dynamic_temporal_bias: bool = True,
    ) -> ActionOutput:
        """Select the next node (and charging portion if a station) for the EV."""
        device = node_embeddings.device
        decoder_context = self._contexts(env, ev, node_embeddings)

        scores = self.pointer.node_scores(decoder_context, node_embeddings)  # [N]
        if use_dynamic_temporal_bias:
            from src.env.tw_diagnostics import compute_dynamic_action_features

            dyn = torch.tensor(
                compute_dynamic_action_features(env, ev),
                device=device,
                dtype=scores.dtype,
            )
            scores = scores + self.pointer.dynamic_score_bias(dyn)
        mask_t = torch.tensor(feasible_mask, dtype=torch.bool, device=device)
        masked_scores = scores.masked_fill(~mask_t, NEG_INF)
        inst = env.instance
        if customer_logit_bonus > 0.0:
            masked_scores = masked_scores.clone()
            for j in inst.customer_indices:
                if mask_t[j]:
                    masked_scores[j] = masked_scores[j] + float(customer_logit_bonus)
        if station_logit_penalty > 0.0:
            masked_scores = masked_scores.clone()
            for s in inst.charging_station_indices:
                if mask_t[s]:
                    masked_scores[s] = masked_scores[s] - float(station_logit_penalty)
        if depot_logit_penalty > 0.0:
            depot_idx = int(env.instance.depot_index)
            if mask_t[depot_idx]:
                masked_scores = masked_scores.clone()
                masked_scores[depot_idx] = masked_scores[depot_idx] - float(depot_logit_penalty)
        log_probs = torch.log_softmax(masked_scores, dim=-1)
        probs = log_probs.exp()

        if mode == "greedy":
            node = int(torch.argmax(masked_scores).item())
        else:
            feas = mask_t & torch.isfinite(probs) & (probs > 0)
            if not feas.any():
                node = int(torch.argmax(masked_scores).item())
            else:
                p = probs.clone()
                p[~feas] = 0.0
                p = p / p.sum().clamp_min(1e-12)
                node = int(torch.multinomial(p, num_samples=1).item())

        node_log_prob = log_probs[node]
        # Entropy over feasible nodes (finite terms only).
        finite = torch.isfinite(log_probs)
        entropy = -(probs[finite] * log_probs[finite]).sum()

        total_log_prob = node_log_prob
        charge_portion = None

        if env.instance.nodes[node].is_charging_station or (
            env.instance.nodes[node].is_depot and env.use_depot_charging and env.instance.depot_has_charging
        ):
            charge_portion, charge_log_prob, charge_entropy = self._select_charge(
                env, ev, node, node_embeddings, decoder_context, mode
            )
            total_log_prob = total_log_prob + charge_log_prob
            entropy = entropy + charge_entropy

        return ActionOutput(
            node=node,
            log_prob=total_log_prob,
            entropy=entropy,
            charge_portion=charge_portion,
            node_probs=probs,
            decoder_context=decoder_context,
        )

    def _select_charge(self, env, ev, station, node_embeddings, decoder_context, mode):
        vp = env.instance.vehicle
        horizon = env.instance.planning_horizon if env.instance.planning_horizon > 0 else 1.0
        battery_norm = ev.remaining_battery / vp.battery_capacity if vp.battery_capacity else 0.0
        time_norm = ev.current_time / horizon
        logits = self.pointer.charging_logits(
            decoder_context, node_embeddings[station], float(battery_norm), float(time_norm)
        )
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if mode == "greedy":
            idx = int(torch.argmax(logits).item())
        else:
            idx = int(torch.multinomial(probs, num_samples=1).item())
        entropy = -(probs * log_probs).sum()
        return self.charging_portions[idx], log_probs[idx], entropy
