"""Extended rollout with step rewards and trajectory collection for actor-critic."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from src.env.cost import CostWeights, training_reward
from src.env.evrptw_env import EVRPTWEnv
from src.env.reward import StepRewardCalculator, snapshot_env
from src.env.tw_diagnostics import (
    aggregate_mask_reasons,
    count_time_expired_customers,
    selected_station_is_necessary,
    urgent_rank_stats,
)
from src.training.actor_critic import RLConfig, apply_potential_shaping, compute_potential
from src.training.reward_shaping import StepContext, TrainingRewardCalculator, TrainingRewardConfig
from src.training.trajectory import EpisodeTransition, Trajectory
from src.training.rl_memory import select_q_candidates
from src.utils.logging_utils import get_logger

from src.training.rollout_transitions import record_transition_reward
from .rollout import EpisodeResult, _is_better

LOGGER = get_logger()
DEBUG_ASSERT_DIR = Path("logs/consistency_assertions")


def _aggregate_reward_breakdown(step_infos: list) -> dict[str, float]:
    keys = [
        "delta_distance_cost", "delta_duration_cost", "delta_charging_time_cost",
        "delta_charging_price_cost", "delta_waiting_cost", "delta_penalty_cost",
        "terminal_penalty_cost", "served_bonus", "early_depot_penalty", "step_penalty",
        "terminal_unserved_penalty_raw", "terminal_unserved_penalty_scaled",
    ]
    out = {k: 0.0 for k in keys}
    for info in step_infos:
        extra = getattr(info, "extra", None) or {}
        for k in keys:
            out[k] += float(extra.get(k, 0.0))
    return out


def _save_assert_debug_row(row: dict) -> None:
    DEBUG_ASSERT_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_ASSERT_DIR / "assert_failures.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def run_episode_trajectory(
    instance,
    policy,
    m: int,
    weights: CostWeights,
    *,
    mode: str = "sampling",
    device: torch.device | str = "cpu",
    charging_portions: list[float] | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    max_steps_factor: int = 6,
    min_decisions_warn: int | None = None,
    rl_cfg: RLConfig | None = None,
    reward_calculator: StepRewardCalculator | TrainingRewardCalculator | None = None,
    collect_trajectory: bool = False,
    cycle_id: int = 1,
    training_mode: bool = True,
    hard_time_windows: bool = True,
    global_epoch: int = 0,
) -> tuple[EpisodeResult, Trajectory | None]:
    """Run episode; optionally collect per-step rewards and critic trajectory."""
    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=charging_portions or policy.charging_portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        max_steps_factor=max_steps_factor,
        hard_time_windows=hard_time_windows,
    )
    node_embeddings = policy.encode(instance, device=device)

    train_calc: TrainingRewardCalculator | None = None
    if training_mode and rl_cfg is not None:
        train_calc = TrainingRewardCalculator(
            weights,
            TrainingRewardConfig.from_rl_config(rl_cfg),
            num_customers=instance.num_customers,
            cycle_id=cycle_id,
        )
        reward_calculator = train_calc
    elif reward_calculator is None and rl_cfg is not None:
        reward_calculator = StepRewardCalculator(weights, reward_scale=rl_cfg.reward_scale)

    log_prob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    num_decisions = 0
    depot_recharge_actions = 0
    depot_finish_actions = 0
    depot_mode_entropy_sum = 0.0
    depot_mode_finish_prob_sum = 0.0
    depot_mode_recharge_prob_sum = 0.0
    trajectory = Trajectory() if collect_trajectory else None
    if trajectory is not None:
        trajectory.node_embeddings = node_embeddings.detach()
    step_infos = []
    phi_prev = compute_potential(env, rl_cfg) if rl_cfg is not None and rl_cfg.use_potential_shaping else 0.0
    phi_holder = [phi_prev]
    original_step_infos: list = []

    def _append_env_transition(before, info: dict, ctx: StepContext | None = None) -> None:
        raw_r, shaping, shaped_r, step_info = record_transition_reward(
            env=env,
            before=before,
            info=info,
            reward_calculator=reward_calculator,
            train_calc=train_calc,
            rl_cfg=rl_cfg,
            step_infos=step_infos,
            ctx=ctx,
            phi_prev_holder=phi_holder,
        )
        original_step_infos.append(step_info)
        if collect_trajectory and trajectory is not None:
            trajectory.transitions.append(
                EpisodeTransition(
                    is_policy_action=False,
                    shaped_reward=shaped_r,
                    raw_reward=raw_r,
                    shaping_reward=shaping,
                    step_info=step_info,
                    action_type=str(info.get("type", "auto_close")),
                    ev_id=int(info.get("ev_id", -1)),
                )
            )

    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            env.done_reason = "no_acting_ev"
            break
        before_close = snapshot_env(env)
        close_info = env.try_auto_close_route(ev)
        if close_info is not None:
            _append_env_transition(before_close, close_info)
            continue

        before = snapshot_env(env)
        pre_context = env.snapshot_decision_context(ev)
        pre_expired = count_time_expired_customers(env)
        mask_counts, feasible_mask, mask_reasons = aggregate_mask_reasons(
            instance, env, ev, use_horizon_check=use_horizon_check, use_depot_charging=use_depot_charging
        )
        cust_bonus, st_pen, dep_pen = (0.0, 0.0, 0.0)
        if train_calc is not None:
            unserved_ratio = 1.0 - env.num_served_customers / max(instance.num_customers, 1)
            cust_bonus, st_pen, dep_pen = train_calc.training_logit_adjustments(
                env, ev, feasible_mask, unserved_ratio=unserved_ratio,
            )
        action = policy.select_action(
            env, ev, node_embeddings, feasible_mask, mode=mode,
            depot_logit_penalty=dep_pen,
            customer_logit_bonus=cust_bonus,
            station_logit_penalty=st_pen,
            use_dynamic_temporal_bias=training_mode,
        )
        pre_rank = urgent_rank_stats(env, ev, action.node, feasible_mask)
        inst = env.instance
        pre_st_necessary = True
        pre_st_diag: dict[str, float] = {}
        if inst.nodes[action.node].is_charging_station:
            pre_st_necessary, pre_st_diag = selected_station_is_necessary(env, ev, action.node)

        pre_action_features = env.compute_action_features(
            ev, action.node, action.charge_portion, depot_finish=action.depot_finish,
        )

        if env.hard_time_windows:
            try:
                env.assert_pre_action_consistency(
                    ev, action.node, action.charge_portion, action.depot_finish,
                    feasible_mask, mask_reasons, global_epoch=global_epoch,
                )
            except AssertionError as exc:
                _save_assert_debug_row({
                    "epoch": global_epoch,
                    "instance": getattr(instance, "instance_name", "?"),
                    "ev_id": ev.ev_id,
                    "action_type": pre_action_features.get("action_type"),
                    "from_node": ev.current_node,
                    "to_node": action.node,
                    "charge_portion": action.charge_portion,
                    "depot_finish": action.depot_finish,
                    "error": str(exc),
                    **pre_context,
                    **pre_action_features,
                    "mask_reason": mask_reasons.get(action.node, ""),
                    "feasible_mask": feasible_mask.tolist(),
                })
                raise

        info = env.apply_action(
            ev, action.node, action.charge_portion, depot_finish=action.depot_finish,
        )
        expired_after = count_time_expired_customers(env)
        node_type = str(info.get("type", "unknown"))
        served = bool(info.get("customer_served", False))

        travel_t = float(info.get("delta_travel_time", 0.0))
        wait_t = float(info.get("waiting", 0.0))
        charge_t = float(info.get("charging_time", 0.0))
        service_t = float(inst.nodes[action.node].service_time) if inst.nodes[action.node].is_customer and served else 0.0
        ctx = StepContext(
            expired_before=pre_expired,
            expired_after=expired_after,
            mask_counts=mask_counts.to_dict(),
            action_time_consumed=travel_t + wait_t + service_t + charge_t,
            travel_time=travel_t,
            waiting_time=wait_t,
            service_time=service_t,
            charging_time=charge_t,
            station_necessary=pre_st_necessary,
            served_slack=pre_rank.served_customer_slack,
            mean_feasible_slack=pre_rank.mean_feasible_customer_slack,
            min_feasible_slack=pre_rank.min_feasible_customer_slack,
            max_feasible_slack=pre_rank.max_feasible_customer_slack,
            urgency_rank=pre_rank.selected_customer_slack_rank,
            urgent_rank_k=train_calc.cfg.urgent_rank_k if train_calc else 10,
        )
        for k, v in pre_st_diag.items():
            ctx.mask_counts[k] = ctx.mask_counts.get(k, 0.0) + float(v)

        raw_r, shaping, shaped_r, step_info = record_transition_reward(
            env=env,
            before=before,
            info=info,
            reward_calculator=reward_calculator,
            train_calc=train_calc,
            rl_cfg=rl_cfg,
            step_infos=step_infos,
            ctx=ctx,
            phi_prev_holder=phi_holder,
            node_type=node_type,
            customer_served=served,
        )
        original_step_infos.append(step_info)

        log_prob_sum = log_prob_sum + action.log_prob
        entropy_sum = entropy_sum + action.entropy
        num_decisions += 1
        if action.depot_finish:
            depot_finish_actions += 1
        elif inst.nodes[action.node].is_depot:
            depot_recharge_actions += 1
        if action.depot_mode_entropy is not None:
            depot_mode_entropy_sum += float(action.depot_mode_entropy.detach())
        if action.depot_mode_finish_prob is not None:
            depot_mode_finish_prob_sum += action.depot_mode_finish_prob
            depot_mode_recharge_prob_sum += action.depot_mode_recharge_prob or 0.0

        if collect_trajectory and trajectory is not None:
            cand_idx = None
            cand_probs = None
            if rl_cfg is not None and rl_cfg.use_counterfactual_critic and action.node_probs is not None:
                cand_idx = select_q_candidates(
                    feasible_mask,
                    action.node_probs,
                    action.node,
                    rl_cfg,
                )
                cand_probs = action.node_probs[torch.tensor(cand_idx, device=action.node_probs.device)].detach().cpu()
            trajectory.transitions.append(
                EpisodeTransition(
                    is_policy_action=True,
                    decoder_context=action.decoder_context if action.decoder_context is not None else policy.decoder_context(env, ev, node_embeddings),
                    feasible_mask=feasible_mask.copy(),
                    node=action.node,
                    ev_id=ev.ev_id,
                    log_prob=action.log_prob,
                    entropy=action.entropy,
                    node_probs=None,
                    q_candidate_indices=cand_idx,
                    q_candidate_probs=cand_probs,
                    raw_reward=raw_r,
                    shaping_reward=shaping,
                    shaped_reward=shaped_r,
                    step_info=step_info,
                    charge_portion=action.charge_portion,
                    action_type=node_type,
                )
            )

    for ev in env.evs:
        if not ev.finished:
            before_close = snapshot_env(env)
            close_info = env.auto_close_route(ev, forced=True, return_info=True)
            _append_env_transition(before_close, close_info)

    done_reason = env.get_done_reason()
    j, breakdown, summary = env.score(weights)

    training_stats: dict[str, float] = {}
    if train_calc is not None and trajectory is not None:
        before_terminal = snapshot_env(env)
        terminal_info = train_calc.terminal_adjustment(before_terminal, summary)
        trajectory.terminal_info = terminal_info
        step_infos.append(terminal_info)
        training_stats = train_calc.finalize(j)
        training_stats["depot_recharge_actions"] = float(depot_recharge_actions)
        training_stats["depot_finish_actions"] = float(depot_finish_actions)
        if num_decisions > 0 and depot_mode_entropy_sum > 0:
            n_depot_modes = depot_finish_actions + depot_recharge_actions
            if n_depot_modes > 0:
                training_stats["depot_mode_entropy"] = depot_mode_entropy_sum / n_depot_modes
                training_stats["depot_mode_finish_prob"] = depot_mode_finish_prob_sum / n_depot_modes
                training_stats["depot_mode_recharge_prob"] = depot_mode_recharge_prob_sum / n_depot_modes
        trajectory.training_stats = training_stats
        original_infos = [s for s in step_infos if s is not None and not getattr(s, "terminal", False)]
        raw_terminal = StepRewardCalculator(weights, reward_scale=rl_cfg.reward_scale).terminal_adjustment(
            before_terminal, summary,
        )
        original_infos.append(raw_terminal)
        recon_original = sum(s.delta_objective for s in original_infos)
        rel_orig = abs(recon_original - j) / max(1.0, abs(j))
        trajectory.reconstructed_original_objective = recon_original
        trajectory.original_objective_mismatch = abs(recon_original - j)
        trajectory.relative_original_objective_mismatch = rel_orig
        _, rel_mismatch, ok = train_calc.base.verify_alignment(original_infos, j)
        trajectory.objective = j
        trajectory.reconstructed_objective = recon_original
        trajectory.objective_mismatch = trajectory.original_objective_mismatch
        trajectory.relative_objective_mismatch = rel_orig
        trajectory.reward_breakdown = _aggregate_reward_breakdown(step_infos)
        if rel_orig > 0.01:
            LOGGER.warning(
                "Original objective mismatch rel=%.4f (final J=%.4f, reconstructed=%.4f)",
                rel_orig, j, recon_original,
            )
        elif not ok:
            LOGGER.debug(
                "Objective mismatch rel=%.4f (final J=%.4f, reconstructed=%.4f)",
                rel_mismatch, j, trajectory.reconstructed_objective,
            )
    elif reward_calculator is not None and trajectory is not None:
        before_terminal = snapshot_env(env)
        terminal_info = reward_calculator.terminal_adjustment(before_terminal, summary)
        trajectory.terminal_info = terminal_info
        step_infos.append(terminal_info)
        mismatch, rel_mismatch, ok = reward_calculator.verify_alignment(step_infos, j)
        trajectory.objective = j
        trajectory.reconstructed_objective = sum(s.delta_objective for s in step_infos)
        trajectory.objective_mismatch = mismatch
        trajectory.relative_objective_mismatch = rel_mismatch
        trajectory.reward_breakdown = _aggregate_reward_breakdown(step_infos)

    if min_decisions_warn is not None and instance.num_customers >= 100:
        if num_decisions < min_decisions_warn:
            LOGGER.warning(
                "Suspiciously few decisions (%d) on instance %s (N=%d); done_reason=%s",
                num_decisions, getattr(instance, "instance_name", "?"),
                instance.num_customers, done_reason,
            )

    result = EpisodeResult(
        env=env,
        objective=j,
        breakdown=breakdown,
        summary=summary,
        log_prob_sum=log_prob_sum,
        entropy_sum=entropy_sum,
        feasible=env.feasible_solution(),
        num_decisions=num_decisions,
        done_reason=done_reason,
        fleet_size=m,
    )
    return result, trajectory


def run_episode(
    instance,
    policy,
    m: int,
    weights: CostWeights,
    mode: str = "sampling",
    device: torch.device | str = "cpu",
    charging_portions: list[float] | None = None,
    use_horizon_check: bool = True,
    max_steps_factor: int = 6,
    min_decisions_warn: int | None = None,
    **kwargs,
) -> EpisodeResult:
    """Backward-compatible episode runner (inference / evaluation)."""
    result, _ = run_episode_trajectory(
        instance, policy, m, weights,
        mode=mode, device=device, charging_portions=charging_portions,
        use_horizon_check=use_horizon_check, max_steps_factor=max_steps_factor,
        min_decisions_warn=min_decisions_warn, collect_trajectory=False,
        training_mode=False, **kwargs,
    )
    return result


def run_pomo_episodes(
    instance,
    policy,
    m: int,
    weights: CostWeights,
    rl_cfg: RLConfig,
    *,
    device: torch.device | str = "cpu",
    charging_portions: list[float] | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    max_steps_factor: int = 6,
    min_decisions_warn: int | None = None,
    cycle_id: int = 1,
) -> tuple[list[tuple[EpisodeResult, Trajectory]], float]:
    """Run K rollouts and return trajectories with group baseline on terminal reward."""
    K = rl_cfg.effective_pomo_k()
    outputs = []
    terminal_rewards = []
    for _ in range(K):
        result, traj = run_episode_trajectory(
            instance, policy, m, weights,
            mode="sampling", device=device,
            charging_portions=charging_portions,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            max_steps_factor=max_steps_factor,
            min_decisions_warn=min_decisions_warn,
            rl_cfg=rl_cfg,
            collect_trajectory=True,
            cycle_id=cycle_id,
            training_mode=True,
            hard_time_windows=True,
        )
        assert traj is not None
        tr = float(getattr(traj, "training_stats", {}).get(
            "shaped_training_return",
            training_reward(result.objective, rl_cfg.reward_scale),
        ))
        terminal_rewards.append(tr)
        traj.pomo_group_reward = None
        outputs.append((result, traj))
    group_mean = float(sum(terminal_rewards) / len(terminal_rewards))
    for i, (res, traj) in enumerate(outputs):
        adv = terminal_rewards[i] - group_mean
        traj.pomo_advantage = adv
        traj.pomo_group_reward = group_mean
    return outputs, group_mean
