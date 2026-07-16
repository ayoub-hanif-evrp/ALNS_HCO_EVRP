"""Centralized actor-critic: GAE, optional counterfactual Q, optional multi-sample rollout baseline, potential shaping."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from src.models.critic import auxiliary_critic_loss, build_trajectory_auxiliary_targets
from src.training.q_critic_ops import q_loss_and_cf_for_record
from src.training.reward_shaping import TerminalPenaltySchedule
from src.training.trajectory import Trajectory


@dataclass
class RLConfig:
    """Actor-critic hyperparameters with memory-safe defaults for large instances."""

    gamma: float = 1.0
    critic_coef: float = 0.5
    q_critic_coef: float = 0.5
    entropy_coef: float = 0.01
    normalize_advantages: bool = True
    reward_scale: float = 1000.0
    gae_lambda: float = 0.95
    counterfactual_mix: float = 0.5
    potential_eta_served: float = 5.0
    potential_eta_remaining_demand: float = 2.0
    potential_eta_battery_risk: float = 3.0
    potential_eta_time_risk: float = 3.0
    battery_safe_ratio: float = 0.2
    multi_sample_K: int = 1
    multi_sample_mix_rho: float = 0.1
    use_multi_sample_rollout_baseline: bool = False
    use_counterfactual_critic: bool = False
    use_step_rewards: bool = True
    use_gae: bool = True
    use_potential_shaping: bool = False
    use_served_customer_bonus: bool = True
    served_customer_bonus: float | None = None
    served_bonus_mode: str = "penalty_equivalent"
    served_bonus_coef: float = 0.5
    use_early_depot_penalty: bool = True
    early_depot_penalty_coef: float = 1.0
    use_step_penalty: bool = True
    step_penalty: float = 0.01
    use_depot_logit_penalty: bool = True
    depot_logit_penalty: float = 2.0
    depot_penalty_until_served_ratio: float = 0.95
    use_urgency_bonus: bool = True
    urgency_bonus_coef: float = 0.5
    use_newly_expired_customer_penalty: bool = True
    newly_expired_customer_penalty: float = 1.0
    use_time_waste_penalty: bool = True
    time_waste_penalty_coef: float = 0.05
    time_waste_scale: float = 1.0
    use_unnecessary_station_penalty: bool = True
    unnecessary_station_penalty: float = 1.0
    use_urgent_customer_rank_penalty: bool = True
    urgent_rank_penalty_coef: float = 0.2
    urgent_rank_k: int = 10
    use_customer_priority_logit_bias: bool = True
    customer_logit_bonus: float = 1.0
    unnecessary_station_logit_penalty: float = 1.5
    use_no_progress_penalty: bool = True
    no_progress_threshold: int = 10
    no_progress_penalty_coef: float = 0.05
    normalize_returns_for_critic: bool = False
    terminal_penalty_schedule: "TerminalPenaltySchedule | None" = None
    q_critic_warmup_epochs: int = 0
    q_critic_enable_after_epoch: int | None = None
    q_eval_mode: str = "topk"
    max_q_actions: int = 64
    q_chunk_size: int = 128
    q_record_batch_size: int = 32
    detach_critic_inputs: bool = True
    assert_action_mask: bool = False
    use_dynamic_candidate_encoder: bool = True
    critic_loss_type: str = "huber"
    critic_updates_per_actor_update: int = 1
    actor_grad_clip: float = 1.0
    critic_grad_clip: float = 1.0
    auxiliary_coef: float = 0.1

    def effective_multi_sample_k(self) -> int:
        return max(1, int(self.multi_sample_K)) if self.use_multi_sample_rollout_baseline else 1

    def effective_pomo_k(self) -> int:
        """Deprecated alias for ``effective_multi_sample_k``."""
        return self.effective_multi_sample_k()

    @property
    def use_pomo(self) -> bool:
        return self.use_multi_sample_rollout_baseline

    @property
    def pomo_K(self) -> int:
        return self.multi_sample_K

    @property
    def pomo_mix_rho(self) -> float:
        return self.multi_sample_mix_rho

    def q_critic_active(self, global_epoch: int) -> bool:
        if not self.use_counterfactual_critic:
            return False
        if self.q_critic_enable_after_epoch is not None:
            return global_epoch >= int(self.q_critic_enable_after_epoch)
        return global_epoch >= int(self.q_critic_warmup_epochs)

    @classmethod
    def from_config(cls, config: dict) -> "RLConfig":
        tcfg = config.get("training") or {}
        rcfg = config.get("rl") or {}
        ocfg = config.get("optimizer") or {}
        ccfg = config.get("critic") or {}
        use_ms = bool(rcfg.get(
            "use_multi_sample_rollout_baseline",
            rcfg.get("use_pomo", False),
        ))
        default_k = int(rcfg.get(
            "multi_sample_K",
            rcfg.get("pomo_K", 4 if use_ms else 1),
        ))
        enable_after = rcfg.get("q_critic_enable_after_epoch")
        sched_raw = rcfg.get("terminal_penalty_schedule") or {}
        terminal_schedule = TerminalPenaltySchedule(
            enabled=bool(sched_raw.get("enabled", False)),
            start_coef=float(sched_raw.get("start_coef", 0.25)),
            end_coef=float(sched_raw.get("end_coef", 1.0)),
            warmup_cycles=int(sched_raw.get("warmup_cycles", 20)),
        )
        return cls(
            gamma=float(rcfg.get("gamma", 1.0)),
            critic_coef=float(rcfg.get("critic_coef", 0.5)),
            q_critic_coef=float(rcfg.get("q_critic_coef", 0.5)),
            entropy_coef=float(tcfg.get("entropy_coef", rcfg.get("entropy_coef", 0.01))),
            normalize_advantages=bool(rcfg.get("normalize_advantages", tcfg.get("normalize_advantage", True))),
            reward_scale=float(tcfg.get("reward_scale", rcfg.get("reward_scale", 1000.0))),
            gae_lambda=float(rcfg.get("gae_lambda", 0.95)),
            counterfactual_mix=float(rcfg.get("counterfactual_mix", 0.5)),
            potential_eta_served=float(rcfg.get("potential_eta_served", 5.0)),
            potential_eta_remaining_demand=float(rcfg.get("potential_eta_remaining_demand", 2.0)),
            potential_eta_battery_risk=float(rcfg.get("potential_eta_battery_risk", 3.0)),
            potential_eta_time_risk=float(rcfg.get("potential_eta_time_risk", 3.0)),
            battery_safe_ratio=float(rcfg.get("battery_safe_ratio", 0.2)),
            multi_sample_K=default_k,
            multi_sample_mix_rho=float(rcfg.get(
                "multi_sample_mix_rho",
                rcfg.get("pomo_mix_rho", 0.1),
            )),
            use_multi_sample_rollout_baseline=use_ms,
            use_counterfactual_critic=bool(rcfg.get("use_counterfactual_critic", False)),
            use_step_rewards=bool(rcfg.get("use_step_rewards", True)),
            use_gae=bool(rcfg.get("use_gae", True)),
            use_potential_shaping=bool(rcfg.get("use_potential_shaping", False)),
            use_served_customer_bonus=bool(rcfg.get("use_served_customer_bonus", True)),
            served_customer_bonus=rcfg.get("served_customer_bonus"),
            served_bonus_mode=str(rcfg.get("served_bonus_mode", "penalty_equivalent")),
            served_bonus_coef=float(rcfg.get("served_bonus_coef", 0.5)),
            use_early_depot_penalty=bool(rcfg.get("use_early_depot_penalty", True)),
            early_depot_penalty_coef=float(rcfg.get("early_depot_penalty_coef", 2.0)),
            use_step_penalty=bool(rcfg.get("use_step_penalty", False)),
            step_penalty=float(rcfg.get("step_penalty", 0.01)),
            use_depot_logit_penalty=bool(rcfg.get("use_depot_logit_penalty", True)),
            depot_logit_penalty=float(rcfg.get("depot_logit_penalty", 2.0)),
            depot_penalty_until_served_ratio=float(rcfg.get("depot_penalty_until_served_ratio", 0.95)),
            use_urgency_bonus=bool(rcfg.get("use_urgency_bonus", True)),
            urgency_bonus_coef=float(rcfg.get("urgency_bonus_coef", 0.5)),
            use_newly_expired_customer_penalty=bool(rcfg.get("use_newly_expired_customer_penalty", True)),
            newly_expired_customer_penalty=float(rcfg.get("newly_expired_customer_penalty", 1.0)),
            use_time_waste_penalty=bool(rcfg.get("use_time_waste_penalty", True)),
            time_waste_penalty_coef=float(rcfg.get("time_waste_penalty_coef", 0.05)),
            time_waste_scale=float(rcfg.get("time_waste_scale", 1.0)),
            use_unnecessary_station_penalty=bool(rcfg.get("use_unnecessary_station_penalty", True)),
            unnecessary_station_penalty=float(rcfg.get("unnecessary_station_penalty", 1.0)),
            use_urgent_customer_rank_penalty=bool(rcfg.get("use_urgent_customer_rank_penalty", True)),
            urgent_rank_penalty_coef=float(rcfg.get("urgent_rank_penalty_coef", 0.2)),
            urgent_rank_k=int(rcfg.get("urgent_rank_k", 10)),
            use_customer_priority_logit_bias=bool(rcfg.get("use_customer_priority_logit_bias", True)),
            customer_logit_bonus=float(rcfg.get("customer_logit_bonus", 1.0)),
            unnecessary_station_logit_penalty=float(rcfg.get("unnecessary_station_logit_penalty", 1.5)),
            normalize_returns_for_critic=bool(rcfg.get("normalize_returns_for_critic", True)),
            terminal_penalty_schedule=terminal_schedule,
            q_critic_warmup_epochs=int(rcfg.get("q_critic_warmup_epochs", 0)),
            q_critic_enable_after_epoch=int(enable_after) if enable_after is not None else None,
            q_eval_mode=str(rcfg.get("q_eval_mode", "topk")),
            max_q_actions=int(rcfg.get("max_q_actions", 64)),
            q_chunk_size=int(rcfg.get("q_chunk_size", 128)),
            q_record_batch_size=int(rcfg.get("q_record_batch_size", 32)),
            detach_critic_inputs=bool(rcfg.get("detach_critic_inputs", True)),
            assert_action_mask=bool(rcfg.get("assert_action_mask", True)),
            use_dynamic_candidate_encoder=bool(rcfg.get("use_dynamic_candidate_encoder", True)),
            critic_loss_type=str(rcfg.get("critic_loss", "huber")),
            critic_updates_per_actor_update=int(rcfg.get(
                "critic_updates_per_actor_update",
                ccfg.get("critic_updates_per_actor_update", 1),
            )),
            auxiliary_coef=float(ccfg.get("auxiliary_coef", rcfg.get("auxiliary_coef", 0.1))),
            actor_grad_clip=float(rcfg.get("actor_grad_clip", ocfg.get("actor_grad_clip", tcfg.get("grad_clip", 1.0)))),
            critic_grad_clip=float(rcfg.get("critic_grad_clip", ocfg.get("critic_grad_clip", tcfg.get("grad_clip", 1.0)))),
        )


def compute_potential(env, rl_cfg: RLConfig) -> float:
    inst = env.instance
    num_c = max(inst.num_customers, 1)
    served_ratio = env.num_served_customers / num_c
    total_demand = inst.total_demand or 1.0
    remaining = sum(
        inst.nodes[c].demand for c in inst.customer_indices if not env.served_mask[c]
    )
    remaining_ratio = remaining / total_demand
    vp = inst.vehicle
    b_safe = rl_cfg.battery_safe_ratio * vp.battery_capacity
    battery_risk = 0.0
    if env.m > 0 and vp.battery_capacity > 0:
        risks = [max(0.0, (b_safe - ev.remaining_battery) / vp.battery_capacity) for ev in env.evs]
        battery_risk = float(np.mean(risks))
    horizon = inst.planning_horizon if inst.planning_horizon > 0 else 1.0
    tw_risk = 0.0
    unserved = [c for c in inst.customer_indices if not env.served_mask[c]]
    if unserved:
        tw_risk = float(
            np.mean([max(0.0, (env.current_event_time - inst.nodes[c].due_time) / horizon) for c in unserved])
        )
    return (
        rl_cfg.potential_eta_served * served_ratio
        - rl_cfg.potential_eta_remaining_demand * remaining_ratio
        - rl_cfg.potential_eta_battery_risk * battery_risk
        - rl_cfg.potential_eta_time_risk * tw_risk
    )


def apply_potential_shaping(
    raw_reward: float,
    phi_before: float,
    phi_after: float,
    gamma: float,
) -> tuple[float, float]:
    shaping = gamma * phi_after - phi_before
    return raw_reward + shaping, shaping


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over a full reward sequence; ``values[t]`` may be zero for non-critic steps."""
    T = rewards.numel()
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_v = 0.0 if t == T - 1 else float(values[t + 1].detach())
        cur_v = float(values[t].detach()) if t < values.numel() else 0.0
        delta = float(rewards[t]) + gamma * next_v - cur_v
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T].detach()
    return advantages, returns


def explained_variance(returns: torch.Tensor, values: torch.Tensor) -> float:
    var_g = float(returns.var(unbiased=False))
    if var_g < 1e-12:
        return 0.0
    return float(1.0 - (returns - values.detach()).var(unbiased=False) / var_g)


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.numel() <= 1:
        return x
    std = x.std()
    if float(std) < 1e-8:
        return x - x.mean()
    return (x - x.mean()) / (std + 1e-8)


@dataclass
class ActorCriticLossOut:
    loss: torch.Tensor
    actor_loss: torch.Tensor
    critic_loss: torch.Tensor
    q_critic_loss: torch.Tensor
    entropy_term: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    explained_var: float
    counterfactual_advantage_mean: float
    mean_q_candidates: float = 0.0
    value_mean: float = 0.0
    value_std: float = 0.0
    return_mean: float = 0.0
    return_std: float = 0.0
    advantage_min: float = 0.0
    advantage_max: float = 0.0
    mean_log_prob: float = 0.0
    aux_loss: float = 0.0
    aux_loss_parts: dict[str, float] | None = None


def _maybe_detach(x: torch.Tensor, detach: bool) -> torch.Tensor:
    return x.detach() if detach else x


def compute_smdp_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over semi-MDP decision epochs (terminal bootstrap = 0 on last epoch)."""
    T = rewards.numel()
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        cur_v = float(values[t].detach())
        if t == T - 1:
            next_v = 0.0
            delta = float(rewards[t]) - cur_v
        else:
            next_v = float(values[t + 1].detach())
            delta = float(rewards[t]) + gamma * next_v - cur_v
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values[:T].detach()
    return advantages, returns


def actor_critic_loss(
    trajectory: Trajectory,
    value_critic: nn.Module,
    q_critic: nn.Module | None,
    rl_cfg: RLConfig,
    *,
    multi_sample_advantage: float | None = None,
    pomo_advantage: float | None = None,
    device: torch.device | str = "cpu",
    global_epoch: int = 0,
    use_q_critic: bool | None = None,
    use_huber: bool = False,
    critic_only: bool = False,
    actor_only: bool = False,
    use_amp: bool = False,
) -> ActorCriticLossOut:
    decisions = trajectory.smdp_decisions()
    z = torch.zeros((), device=device, requires_grad=True)
    if not decisions:
        return ActorCriticLossOut(z, z, z, z, z, z, z, 0.0, 0.0)

    detach = rl_cfg.detach_critic_inputs
    log_probs = trajectory.stack_smdp_log_probs()
    entropies = trajectory.stack_smdp_entropies()

    device_t = torch.device(device) if isinstance(device, str) else device
    n_cust = max(int(getattr(trajectory, "num_customers", 0) or 0), 1)
    compute_aux = bool(decisions) and hasattr(value_critic, "forward_from_states_batch")

    values_list = [torch.zeros((), device=device_t) for _ in decisions]
    aux_preds_stacked: dict[str, list[torch.Tensor]] = {
        k: [torch.zeros((), device=device_t) for _ in decisions]
        for k in ("unserved_ratio", "serviceable_ratio", "objective_estimate", "feasibility_prob")
    } if compute_aux else {k: [] for k in ("unserved_ratio", "serviceable_ratio", "objective_estimate", "feasibility_prob")}

    # Actor-only: reuse stored value predictions — never re-forward the critic.
    use_stored_values = False
    if actor_only:
        stored = []
        for dec in decisions:
            vp = getattr(dec, "value_prediction", None)
            if vp is None:
                stored = []
                break
            stored.append(vp.detach().to(device_t).reshape(()))
        if stored and len(stored) == len(decisions):
            values_list = stored
            use_stored_values = True

    if not use_stored_values:
        critic_states: list = []
        state_indices: list[int] = []
        for i, dec in enumerate(decisions):
            state = getattr(dec, "decision_state", None)
            if state is None:
                continue
            st = state.detach_for_critic() if detach else state
            critic_states.append(st)
            state_indices.append(i)

        if critic_states:
            from src.utils.gpu_runtime import autocast_ctx
            from src.env.analysis_counters import bump
            if actor_only:
                bump("critic_forward_calls_actor_phase")
            amp_on = bool(use_amp and device_t.type == "cuda")
            forward_ctx = torch.no_grad() if actor_only else contextlib.nullcontext()
            with forward_ctx:
                with autocast_ctx(device_t, enabled=amp_on):
                    if hasattr(value_critic, "forward_from_states_batch"):
                        batch_values, batch_aux = value_critic.forward_from_states_batch(critic_states)
                    else:
                        batch_values_list = []
                        batch_aux_lists = {k: [] for k in aux_preds_stacked}
                        for st in critic_states:
                            v, aux_out = value_critic.forward_from_state(st)
                            batch_values_list.append(v.squeeze() if v.dim() > 0 else v)
                            for k in batch_aux_lists:
                                batch_aux_lists[k].append(aux_out[k].reshape(()))
                        batch_values = torch.stack(batch_values_list)
                        batch_aux = {k: torch.stack(v) for k, v in batch_aux_lists.items()}

            for j in range(len(critic_states)):
                pos = state_indices[j]
                val = batch_values[j].detach() if actor_only else batch_values[j]
                values_list[pos] = val
                if critic_only:
                    decisions[pos].value_prediction = val.detach()
                if compute_aux and not actor_only:
                    for k in aux_preds_stacked:
                        aux_preds_stacked[k][pos] = batch_aux[k][j].reshape(())

    policy_values = torch.stack(values_list) if values_list else torch.zeros(0, device=device_t)
    policy_values = policy_values.float()
    if critic_only:
        for i, dec in enumerate(decisions):
            if getattr(dec, "value_prediction", None) is None and i < len(values_list):
                dec.value_prediction = values_list[i].detach()

    rewards_list = trajectory.smdp_reward_sequence()
    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device)
    values = policy_values[: rewards.numel()]

    if rl_cfg.use_gae:
        advantages_full, returns_full = compute_smdp_gae(
            rewards, values, rl_cfg.gamma, rl_cfg.gae_lambda,
        )
    else:
        returns_full = torch.zeros_like(rewards)
        running = 0.0
        for t in range(returns_full.numel() - 1, -1, -1):
            running = float(rewards[t]) + rl_cfg.gamma * running
            returns_full[t] = running
        advantages_full = returns_full - policy_values.detach()

    actor_adv = advantages_full[: len(decisions)]
    critic_returns = returns_full[: len(decisions)]
    critic_values = policy_values[: len(decisions)]

    if rl_cfg.normalize_returns_for_critic and critic_returns.numel() > 1:
        critic_returns = normalize_tensor(critic_returns)

    ms_adv = multi_sample_advantage if multi_sample_advantage is not None else pomo_advantage
    if ms_adv is not None and rl_cfg.use_multi_sample_rollout_baseline:
        actor_adv = actor_adv + rl_cfg.multi_sample_mix_rho * float(ms_adv)

    active_q = use_q_critic if use_q_critic is not None else rl_cfg.q_critic_active(global_epoch)
    cf_list: list[torch.Tensor] = []
    q_loss_terms: list[torch.Tensor] = []
    q_cand_counts: list[int] = []

    if active_q and q_critic is not None and trajectory.node_embeddings is not None:
        H = _maybe_detach(trajectory.node_embeddings, detach)
        batch_size = max(1, int(rl_cfg.q_record_batch_size))
        for start in range(0, len(decisions), batch_size):
            batch = decisions[start : start + batch_size]
            for i, rec in enumerate(batch, start=start):
                if rec.q_candidate_indices is None:
                    continue
                dec = _maybe_detach(rec.decoder_context, detach)
                if rec.q_candidate_probs is None or rec.q_candidate_indices is None:
                    continue
                q_loss, cf = q_loss_and_cf_for_record(
                    q_critic,
                    dec,
                    H,
                    rec.q_candidate_indices,
                    rec.q_candidate_probs,
                    rec.node,
                    critic_returns[i],
                    q_eval_mode=rl_cfg.q_eval_mode,
                    q_chunk_size=rl_cfg.q_chunk_size,
                )
                cf_list.append(cf)
                q_loss_terms.append(q_loss)
                q_cand_counts.append(len(rec.q_candidate_indices))

    if cf_list:
        cf_adv = torch.stack(cf_list)
        cf_mean = float(cf_adv.mean().detach())
        actor_adv = (1.0 - rl_cfg.counterfactual_mix) * actor_adv + rl_cfg.counterfactual_mix * cf_adv.detach()
        q_critic_loss = torch.stack(q_loss_terms).mean()
        mean_q_cand = float(np.mean(q_cand_counts)) if q_cand_counts else 0.0
    else:
        cf_mean = 0.0
        q_critic_loss = torch.zeros((), device=device)
        mean_q_cand = 0.0

    if rl_cfg.normalize_advantages:
        actor_adv = normalize_tensor(actor_adv)

    actor_loss = -(actor_adv.detach() * log_probs).mean()
    if use_huber:
        value_loss = torch.nn.functional.smooth_l1_loss(
            critic_values.float(), critic_returns.detach().float(),
        )
    else:
        value_loss = ((critic_values.float() - critic_returns.detach().float()) ** 2).mean()

    aux_loss = torch.zeros((), device=device)
    aux_parts: dict[str, float] = {}
    if compute_aux and aux_preds_stacked["unserved_ratio"]:
        aux_preds = {k: torch.stack(v) for k, v in aux_preds_stacked.items()}
        targets = build_trajectory_auxiliary_targets(
            trajectory,
            num_customers=n_cust,
            terminal_feasible=bool(getattr(trajectory, "terminal_feasible", False)),
            device=device,
        )
        aux_loss, aux_parts = auxiliary_critic_loss(
            aux_preds, targets, coef=rl_cfg.auxiliary_coef,
        )

    critic_loss = value_loss + aux_loss
    entropy_term = entropies.mean()

    if critic_only:
        actor_loss = torch.zeros((), device=device)
        q_critic_loss = torch.zeros((), device=device)
        loss = rl_cfg.critic_coef * critic_loss
    elif actor_only:
        loss = actor_loss - rl_cfg.entropy_coef * entropy_term
        if active_q:
            loss = loss + rl_cfg.q_critic_coef * q_critic_loss
        critic_loss = torch.zeros((), device=device)
    else:
        loss = actor_loss + rl_cfg.critic_coef * critic_loss
        if active_q:
            loss = loss + rl_cfg.q_critic_coef * q_critic_loss
        loss = loss - rl_cfg.entropy_coef * entropy_term

    ev = explained_variance(critic_returns, critic_values)
    return ActorCriticLossOut(
        loss=loss,
        actor_loss=actor_loss,
        critic_loss=critic_loss,
        q_critic_loss=q_critic_loss,
        entropy_term=entropy_term,
        advantages=actor_adv,
        returns=critic_returns,
        explained_var=ev,
        counterfactual_advantage_mean=cf_mean,
        mean_q_candidates=mean_q_cand,
        value_mean=float(critic_values.mean().detach()),
        value_std=float(critic_values.std().detach()) if critic_values.numel() > 1 else 0.0,
        return_mean=float(critic_returns.mean().detach()),
        return_std=float(critic_returns.std().detach()) if critic_returns.numel() > 1 else 0.0,
        advantage_min=float(actor_adv.min().detach()),
        advantage_max=float(actor_adv.max().detach()),
        mean_log_prob=float(log_probs.mean().detach()),
        aux_loss=float(aux_loss.detach()) if aux_loss.requires_grad or aux_loss.numel() else float(aux_loss),
        aux_loss_parts=aux_parts,
    )
