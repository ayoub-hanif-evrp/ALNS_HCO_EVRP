"""Transformer multi-agent policy orchestrating encoder + event-driven decoder."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.utils.tensor_utils import NEG_INF, safe_mask_logits

from .decoder import PointerDecoder
from .decision_state import DecisionState
from .dynamic_candidate import compute_dynamic_candidate_features
from .ev_communication import EVCommunication
from .ev_state_encoder import EVStateEncoder
from .edge_encoder import EdgeAwareNodeEncoder
from .global_context import GlobalStateEncoder, MaskSummaryEncoder
from .node_encoder import NodeEncoder
from .feature_specs import (
    ACTING_EV_FEATURE_DIM,
    DYNAMIC_CANDIDATE_FEAT_DIM,
    GLOBAL_STATE_FEATURE_DIM,
)
from .normalization import FeatureNormalizer
from src.models.architecture_assertions import assert_decision_state
from .state_features import (
    acting_ev_features,
    battery_state_features,
    ev_token_scalars,
    fleet_balance_features,
    global_state_features,
    mask_summary_features,
)


@dataclass
class ActionOutput:
    node: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    node_log_prob: torch.Tensor | None = None
    secondary_log_prob: torch.Tensor | None = None
    charge_portion: float | None = None
    depot_finish: bool = False
    node_probs: torch.Tensor | None = None
    decoder_context: torch.Tensor | None = None
    decision_state: DecisionState | None = None
    depot_mode_log_prob: torch.Tensor | None = None
    depot_mode_entropy: torch.Tensor | None = None
    depot_mode_finish_prob: float | None = None
    depot_mode_recharge_prob: float | None = None


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
        self.use_dynamic_candidate_encoder = bool(model_cfg.get("use_dynamic_candidate_encoder", True))
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
        self.global_state_encoder = GlobalStateEncoder(embedding_dim=d)
        self.mask_summary_encoder = MaskSummaryEncoder(embedding_dim=d)
        self.pointer = PointerDecoder(embedding_dim=d, num_charging_portions=len(self.charging_portions))
        self.embedding_dim = d
        self.acting_ev_normalizer = FeatureNormalizer(ACTING_EV_FEATURE_DIM)
        self.global_normalizer = FeatureNormalizer(GLOBAL_STATE_FEATURE_DIM)
        self.dynamic_continuous_normalizer = FeatureNormalizer(DYNAMIC_CANDIDATE_FEAT_DIM - 4)

    def encode(self, instance, device: torch.device | str = "cpu") -> torch.Tensor:
        feats = torch.tensor(instance.node_feature_matrix(), device=device)
        if self.use_edge_aware_encoder:
            return self.node_encoder.encode_instance(instance, feats, device=device)
        return self.node_encoder(feats)

    def build_decision_state(
        self,
        env,
        ev,
        node_embeddings: torch.Tensor,
        mask_result,
        *,
        analysis=None,
    ) -> DecisionState:
        from src.env.action_masks import ActionMaskResult
        from src.env.decision_analysis import DecisionStateAnalysis

        if analysis is not None and not isinstance(analysis, DecisionStateAnalysis):
            raise TypeError("analysis must be DecisionStateAnalysis when provided")
        if analysis is not None:
            analysis.assert_fresh(env, ev)
            mask_result = analysis.mask_result
        if not isinstance(mask_result, ActionMaskResult):
            raise TypeError("build_decision_state requires ActionMaskResult")

        device = node_embeddings.device
        dtype = node_embeddings.dtype

        ev_feats = self.acting_ev_normalizer(
            torch.tensor(acting_ev_features(env, ev), device=device, dtype=dtype),
        )
        h_current = node_embeddings[ev.current_node]
        acting_ev_context = self.ev_state_encoder(ev_feats, h_current)

        token_scalars = torch.tensor(
            np.stack([ev_token_scalars(env, e) for e in env.evs]), device=device, dtype=dtype,
        )
        ev_node_embs = node_embeddings[[e.current_node for e in env.evs]]
        finished = torch.tensor([e.finished for e in env.evs], dtype=torch.bool, device=device)
        acting_idx = [e.ev_id for e in env.evs].index(ev.ev_id)
        fleet_balance_raw = torch.tensor(fleet_balance_features(env), device=device, dtype=dtype)

        if self.use_ev_communication and len(env.evs) > 1:
            comm = self.ev_communication(
                token_scalars, ev_node_embs, acting_idx, finished, fleet_balance_raw,
            )
        else:
            from src.models.ev_communication import EVCommunicationOutput
            z = torch.zeros(self.embedding_dim, device=device, dtype=dtype)
            comm = EVCommunicationOutput(
                communicated_tokens=acting_ev_context.unsqueeze(0),
                acting_token=acting_ev_context,
                active_pool=z,
                finished_pool=z,
                fleet_balance_context=self.ev_communication.fleet_balance_proj(fleet_balance_raw),
            )

        if analysis is not None and analysis.global_features is not None:
            global_raw = self.global_normalizer(
                torch.tensor(analysis.global_features, device=device, dtype=dtype),
            )
        else:
            global_raw = self.global_normalizer(
                torch.tensor(global_state_features(env, mask_result), device=device, dtype=dtype),
            )
        global_context = self.global_state_encoder(global_raw)
        if analysis is not None and analysis.mask_summary_features is not None:
            mask_raw = torch.tensor(analysis.mask_summary_features, device=device, dtype=dtype)
        else:
            mask_raw = torch.tensor(mask_summary_features(env, mask_result), device=device, dtype=dtype)
        mask_context = self.mask_summary_encoder(mask_raw)

        decoder_context = self.pointer.fuse_decoder_context(
            acting_ev_context,
            comm.acting_token,
            comm.active_pool,
            comm.finished_pool,
            global_context,
            mask_context,
            comm.fleet_balance_context,
        )

        # Prefer canonical dynamic features — do not recompute when analysis is given.
        if analysis is not None and analysis.dynamic_candidate_features is not None:
            dyn_feats_np = analysis.dynamic_candidate_features
        else:
            from src.env.analysis_counters import bump_outside_analyze
            bump_outside_analyze("dynamic_feature_recompute_calls")
            dyn_feats_np = compute_dynamic_candidate_features(env, ev, mask_result)
        dyn_cont = self.dynamic_continuous_normalizer(
            torch.tensor(dyn_feats_np[:, : DYNAMIC_CANDIDATE_FEAT_DIM - 4], device=device, dtype=dtype),
        )
        dyn_feats = torch.cat([
            dyn_cont,
            torch.tensor(dyn_feats_np[:, DYNAMIC_CANDIDATE_FEAT_DIM - 4 :], device=device, dtype=dtype),
        ], dim=-1)
        candidate_embeddings, dyn_emb = self.pointer.fuse_candidates(
            node_embeddings,
            dyn_feats,
            use_dynamic_encoder=self.use_dynamic_candidate_encoder,
        )

        state = DecisionState(
            decoder_context=decoder_context,
            acting_ev_context=acting_ev_context,
            communicated_ev_tokens=comm.communicated_tokens,
            acting_ev_token=comm.acting_token,
            active_fleet_context=comm.active_pool,
            finished_fleet_context=comm.finished_pool,
            fleet_balance_context=comm.fleet_balance_context,
            global_context=global_context,
            mask_context=mask_context,
            static_node_embeddings=node_embeddings,
            candidate_embeddings=candidate_embeddings,
            dynamic_candidate_embeddings=dyn_emb,
            dynamic_candidate_feats=dyn_feats,
            global_features_raw=global_raw,
            mask_summary_raw=mask_raw,
            fleet_balance_raw=fleet_balance_raw,
            feasible_mask=mask_result.node_mask.copy(),
            mask_result=mask_result,
        )
        assert_decision_state(state, instance_name=getattr(env.instance, "instance_name", "?"))
        return state

    def decoder_context(self, env, ev, node_embeddings: torch.Tensor, mask_result=None) -> torch.Tensor:
        if mask_result is None:
            mask_result = env.compute_action_masks(ev)
        return self.build_decision_state(env, ev, node_embeddings, mask_result).decoder_context

    def select_action(
        self,
        env,
        ev,
        node_embeddings: torch.Tensor,
        mask_result,
        mode: str = "sampling",
        *,
        analysis=None,
    ) -> ActionOutput:
        if analysis is None:
            last = getattr(env, "last_decision_analysis", None)
            if (
                last is not None
                and int(getattr(last, "state_version", -1)) == int(getattr(env, "state_version", -2))
                and int(getattr(last, "active_ev_id", -1)) == int(ev.ev_id)
            ):
                analysis = last
        state = self.build_decision_state(
            env, ev, node_embeddings, mask_result, analysis=analysis,
        )
        return self.select_action_from_state(
            env, ev, node_embeddings, mask_result, state, mode=mode,
        )

    def select_action_from_state(
        self,
        env,
        ev,
        node_embeddings: torch.Tensor,
        mask_result,
        state: DecisionState,
        mode: str = "sampling",
    ) -> ActionOutput:
        device = node_embeddings.device
        decoder_context = state.decoder_context

        scores = self.pointer.node_scores(
            decoder_context,
            state.candidate_embeddings,
            state.dynamic_candidate_embeddings,
        )
        mask_t = torch.tensor(state.feasible_mask, dtype=torch.bool, device=device)
        masked_scores = safe_mask_logits(scores, mask_t)
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
        finite = torch.isfinite(log_probs)
        entropy = -(probs[finite] * log_probs[finite]).sum()
        total_log_prob = node_log_prob
        secondary_log_prob = None
        charge_portion = None
        depot_finish = False
        depot_mode_log_prob = None
        depot_mode_entropy = None
        depot_mode_finish_prob = None
        depot_mode_recharge_prob = None

        inst = env.instance
        if inst.nodes[node].is_depot:
            ok_recharge = bool(mask_result.depot_mode_mask[0])
            ok_finish = bool(mask_result.depot_mode_mask[1])
            if ok_finish and not ok_recharge:
                depot_finish = True
            elif ok_recharge and not ok_finish:
                depot_finish = False
            elif ok_finish and ok_recharge:
                mode_logits = self.pointer.depot_mode_logits(
                    decoder_context,
                    state.candidate_embeddings[node],
                    state.global_context,
                    state.mask_context,
                )
                dm = mask_result.depot_mode_mask
                mode_logits = safe_mask_logits(
                    mode_logits,
                    torch.tensor(dm, device=device, dtype=torch.bool),
                )
                mode_lp = torch.log_softmax(mode_logits, dim=-1)
                mode_probs = mode_lp.exp()
                if mode == "greedy":
                    depot_finish = bool(torch.argmax(mode_logits).item() == 1)
                else:
                    depot_finish = bool(torch.multinomial(mode_probs, 1).item() == 1)
                depot_mode_lp = mode_lp[1 if depot_finish else 0]
                total_log_prob = total_log_prob + depot_mode_lp
                secondary_log_prob = depot_mode_lp
                mode_ent = -(mode_probs * mode_lp).sum()
                entropy = entropy + mode_ent
                depot_mode_log_prob = depot_mode_lp
                depot_mode_entropy = mode_ent
                depot_mode_finish_prob = float(mode_probs[1].detach())
                depot_mode_recharge_prob = float(mode_probs[0].detach())

            if not depot_finish:
                charge_portion, charge_log_prob, charge_entropy = self._select_charge(
                    env, ev, node, state, mode,
                )
                total_log_prob = total_log_prob + charge_log_prob
                secondary_log_prob = charge_log_prob
                entropy = entropy + charge_entropy
        elif inst.nodes[node].is_charging_station:
            charge_portion, charge_log_prob, charge_entropy = self._select_charge(
                env, ev, node, state, mode,
            )
            total_log_prob = total_log_prob + charge_log_prob
            secondary_log_prob = charge_log_prob
            entropy = entropy + charge_entropy

        return ActionOutput(
            node=node,
            log_prob=total_log_prob,
            entropy=entropy,
            node_log_prob=node_log_prob,
            secondary_log_prob=secondary_log_prob,
            charge_portion=charge_portion,
            depot_finish=depot_finish,
            node_probs=probs,
            decoder_context=decoder_context,
            decision_state=state,
            depot_mode_log_prob=depot_mode_log_prob,
            depot_mode_entropy=depot_mode_entropy,
            depot_mode_finish_prob=depot_mode_finish_prob,
            depot_mode_recharge_prob=depot_mode_recharge_prob,
        )

    def _select_charge(self, env, ev, station, state: DecisionState, mode: str):
        from src.env.mask_errors import InvalidActionMaskError

        device = state.decoder_context.device
        dtype = state.decoder_context.dtype
        batt = torch.tensor(battery_state_features(env, ev), device=device, dtype=dtype)
        logits = self.pointer.charging_logits(
            state.decoder_context,
            state.candidate_embeddings[station],
            state.acting_ev_context,
            batt,
        )
        mask_result = state.mask_result
        if station not in mask_result.charge_masks:
            raise InvalidActionMaskError(f"no charge_masks for station node {station}")
        portion_mask = mask_result.charge_masks[station]
        mask_t = torch.tensor(portion_mask, device=device, dtype=torch.bool)
        logits = safe_mask_logits(logits, mask_t)
        log_probs = torch.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        if not mask_t.any():
            raise InvalidActionMaskError(f"empty charge mask for node {station}")
        if mode == "greedy":
            idx = int(torch.argmax(logits).item())
        else:
            p = probs.clone()
            p[~mask_t] = 0.0
            p = p / p.sum().clamp_min(1e-12)
            idx = int(torch.multinomial(p, num_samples=1).item())
        entropy = -(probs[mask_t] * log_probs[mask_t]).sum()
        return self.charging_portions[idx], log_probs[idx], entropy
