"""Centralized actor-critic: GAE, optional counterfactual Q, optional POMO, potential shaping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

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
    pomo_K: int = 1
    pomo_mix_rho: float = 0.1
    use_pomo: bool = False
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
    normalize_returns_for_critic: bool = True
    terminal_penalty_schedule: "TerminalPenaltySchedule | None" = None
    q_critic_warmup_epochs: int = 0
    q_critic_enable_after_epoch: int | None = None
    q_eval_mode: str = "topk"
    max_q_actions: int = 64
    q_chunk_size: int = 128
    q_record_batch_size: int = 32
    detach_critic_inputs: bool = True

    def effective_pomo_k(self) -> int:
        return max(1, int(self.pomo_K)) if self.use_pomo else 1

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
        use_pomo = bool(rcfg.get("use_pomo", False))
        default_k = int(rcfg.get("pomo_K", 4 if use_pomo else 1))
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
            pomo_K=default_k,
            pomo_mix_rho=float(rcfg.get("pomo_mix_rho", 0.1)),
            use_pomo=use_pomo,
            use_counterfactual_critic=bool(rcfg.get("use_counterfactual_critic", False)),
            use_step_rewards=bool(rcfg.get("use_step_rewards", True)),
            use_gae=bool(rcfg.get("use_gae", True)),
            use_potential_shaping=bool(rcfg.get("use_potential_shaping", False)),
            use_served_customer_bonus=bool(rcfg.get("use_served_customer_bonus", True)),
            served_customer_bonus=rcfg.get("served_customer_bonus"),
            served_bonus_mode=str(rcfg.get("served_bonus_mode", "penalty_equivalent")),
            served_bonus_coef=float(rcfg.get("served_bonus_coef", 0.5)),
            use_early_depot_penalty=bool(rcfg.get("use_early_depot_penalty", True)),
            early_depot_penalty_coef=float(rcfg.get("early_depot_penalty_coef", 1.0)),
            use_step_penalty=bool(rcfg.get("use_step_penalty", True)),
            step_penalty=float(rcfg.get("step_penalty", 0.01)),
            use_depot_logit_penalty=bool(rcfg.get("use_depot_logit_penalty", True)),
            depot_logit_penalty=float(rcfg.get("depot_logit_penalty", 2.0)),
            depot_penalty_until_served_ratio=float(rcfg.get("depot_penalty_until_served_ratio", 0.95)),
            normalize_returns_for_critic=bool(rcfg.get("normalize_returns_for_critic", True)),
            terminal_penalty_schedule=terminal_schedule,
            q_critic_warmup_epochs=int(rcfg.get("q_critic_warmup_epochs", 0)),
            q_critic_enable_after_epoch=int(enable_after) if enable_after is not None else None,
            q_eval_mode=str(rcfg.get("q_eval_mode", "topk")),
            max_q_actions=int(rcfg.get("max_q_actions", 64)),
            q_chunk_size=int(rcfg.get("q_chunk_size", 128)),
            q_record_batch_size=int(rcfg.get("q_record_batch_size", 32)),
            detach_critic_inputs=bool(rcfg.get("detach_critic_inputs", True)),
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
    T = rewards.numel()
    advantages = torch.zeros_like(rewards)
    last_gae = 0.0
    for t in range(T - 1, -1, -1):
        next_v = 0.0 if t == T - 1 else float(values[t + 1].detach())
        delta = float(rewards[t]) + gamma * next_v - float(values[t].detach())
        last_gae = delta + gamma * lam * last_gae
        advantages[t] = last_gae
    returns = advantages + values.detach()
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


def _maybe_detach(x: torch.Tensor, detach: bool) -> torch.Tensor:
    return x.detach() if detach else x


def actor_critic_loss(
    trajectory: Trajectory,
    value_critic: nn.Module,
    q_critic: nn.Module | None,
    rl_cfg: RLConfig,
    *,
    pomo_advantage: float | None = None,
    device: torch.device | str = "cpu",
    global_epoch: int = 0,
    use_q_critic: bool | None = None,
) -> ActorCriticLossOut:
    steps = trajectory.steps
    z = torch.zeros((), device=device, requires_grad=True)
    if not steps:
        return ActorCriticLossOut(z, z, z, z, z, z, z, 0.0, 0.0)

    detach = rl_cfg.detach_critic_inputs
    contexts = _maybe_detach(trajectory.stack_contexts(), detach)
    log_probs = trajectory.stack_log_probs()
    entropies = trajectory.stack_entropies()
    values = value_critic(contexts)

    step_rewards = [s.shaped_reward for s in steps]
    if trajectory.terminal_info is not None:
        step_rewards.append(trajectory.terminal_info.reward)
    rewards = torch.tensor(step_rewards, dtype=torch.float32, device=device)

    if rewards.numel() > values.numel():
        values = torch.cat([values, torch.zeros(1, device=device)])

    if rl_cfg.use_gae:
        advantages, returns = compute_gae(
            rewards[: values.numel()], values, rl_cfg.gamma, rl_cfg.gae_lambda,
        )
    else:
        returns = torch.zeros_like(rewards[: values.numel()])
        running = 0.0
        for t in range(returns.numel() - 1, -1, -1):
            running = float(rewards[t]) + rl_cfg.gamma * running
            returns[t] = running
        advantages = returns - values[: returns.numel()].detach()
    actor_adv = advantages[: len(steps)]
    critic_returns = returns[: len(steps)]
    critic_values = values[: len(steps)]

    if rl_cfg.normalize_returns_for_critic and critic_returns.numel() > 1:
        critic_returns = normalize_tensor(critic_returns)

    if pomo_advantage is not None and rl_cfg.use_pomo:
        actor_adv = actor_adv + rl_cfg.pomo_mix_rho * float(pomo_advantage)

    active_q = use_q_critic if use_q_critic is not None else rl_cfg.q_critic_active(global_epoch)
    cf_list: list[torch.Tensor] = []
    q_loss_terms: list[torch.Tensor] = []
    q_cand_counts: list[int] = []

    if active_q and q_critic is not None and trajectory.node_embeddings is not None:
        H = _maybe_detach(trajectory.node_embeddings, detach)
        batch_size = max(1, int(rl_cfg.q_record_batch_size))
        for start in range(0, len(steps), batch_size):
            batch = steps[start : start + batch_size]
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
    critic_loss = ((critic_values - critic_returns.detach()) ** 2).mean()
    entropy_term = entropies.mean()
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
    )
