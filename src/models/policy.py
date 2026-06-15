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
from .node_encoder import NodeEncoder


@dataclass
class ActionOutput:
    node: int
    log_prob: torch.Tensor  # node + charging log-prob (sum), differentiable
    entropy: torch.Tensor
    charge_portion: float | None = None


class TransformerEVRPTWPolicy(nn.Module):
    def __init__(self, model_cfg: dict, charging_portions: list[float] | None = None):
        super().__init__()
        d = int(model_cfg.get("embedding_dim", 128))
        heads = int(model_cfg.get("heads", 8))
        layers = int(model_cfg.get("encoder_layers", 3))
        ff = int(model_cfg.get("feedforward_dim", 512))
        dropout = float(model_cfg.get("dropout", 0.1))
        self.use_ev_communication = bool(model_cfg.get("use_ev_communication", True))
        self.charging_portions = charging_portions or [0.25, 0.50, 0.75, 1.0]

        self.node_encoder = NodeEncoder(
            embedding_dim=d, num_heads=heads, num_layers=layers, ff_dim=ff, dropout=dropout
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
        return self.node_encoder(feats)

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
    ) -> ActionOutput:
        """Select the next node (and charging portion if a station) for the EV."""
        device = node_embeddings.device
        decoder_context = self._contexts(env, ev, node_embeddings)

        scores = self.pointer.node_scores(decoder_context, node_embeddings)  # [N]
        mask_t = torch.tensor(feasible_mask, dtype=torch.bool, device=device)
        masked_scores = scores.masked_fill(~mask_t, NEG_INF)
        log_probs = torch.log_softmax(masked_scores, dim=-1)
        probs = log_probs.exp()

        if mode == "greedy":
            node = int(torch.argmax(masked_scores).item())
        else:
            node = int(torch.multinomial(probs, num_samples=1).item())

        node_log_prob = log_probs[node]
        # Entropy over feasible nodes (finite terms only).
        finite = torch.isfinite(log_probs)
        entropy = -(probs[finite] * log_probs[finite]).sum()

        total_log_prob = node_log_prob
        charge_portion = None

        if env.instance.nodes[node].is_charging_station:
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
