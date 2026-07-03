"""Extended rollout with step rewards and trajectory collection for actor-critic."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.env.cost import CostWeights, training_reward
from src.env.evrptw_env import EVRPTWEnv
from src.env.reward import StepRewardCalculator, snapshot_env
from src.training.actor_critic import RLConfig, apply_potential_shaping, compute_potential
from src.training.reward_shaping import TrainingRewardCalculator, TrainingRewardConfig
from src.training.trajectory import StepRecord, Trajectory
from src.training.rl_memory import select_q_candidates
from src.utils.logging_utils import get_logger

from .rollout import EpisodeResult, _is_better, _step_or_close

LOGGER = get_logger()


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
) -> tuple[EpisodeResult, Trajectory | None]:
    """Run episode; optionally collect per-step rewards and critic trajectory."""
    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=charging_portions or policy.charging_portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        max_steps_factor=max_steps_factor,
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
    trajectory = Trajectory() if collect_trajectory else None
    if trajectory is not None:
        trajectory.node_embeddings = node_embeddings.detach()
    step_infos = []
    phi_prev = compute_potential(env, rl_cfg) if rl_cfg is not None and rl_cfg.use_potential_shaping else 0.0

    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            env.done_reason = "no_acting_ev"
            break
        if _step_or_close(env, ev):
            continue

        before = snapshot_env(env)
        depot_pen = train_calc.depot_logit_penalty(env) if train_calc is not None else 0.0
        feasible_mask, _ = env.feasible_mask(ev)
        action = policy.select_action(
            env, ev, node_embeddings, feasible_mask, mode=mode,
            depot_logit_penalty=depot_pen,
        )
        info = env.apply_action(ev, action.node, action.charge_portion)
        after = snapshot_env(env)
        node_type = str(info.get("type", "unknown"))
        served = bool(info.get("customer_served", False))

        step_info = None
        raw_r = 0.0
        shaping = 0.0
        shaped_r = 0.0
        if reward_calculator is not None:
            if isinstance(reward_calculator, TrainingRewardCalculator):
                step_info = reward_calculator.base.step_reward(
                    before,
                    after,
                    ev_id=int(info.get("ev_id", ev.ev_id)),
                    selected_node=action.node,
                    node_type=node_type,
                    previous_node=int(info.get("previous_node", ev.current_node)),
                    delta_travel_time=float(info.get("delta_travel_time", 0.0)),
                    delta_energy_used=float(info.get("delta_energy_used", 0.0)),
                    charged_energy=float(info.get("charged_energy", 0.0)),
                    customer_served=served,
                )
                step_info = reward_calculator.shape_step(
                    step_info, before=before, after=after,
                    node_type=node_type, customer_served=served,
                )
            else:
                step_info = reward_calculator.step_reward(
                    before,
                    after,
                    ev_id=int(info.get("ev_id", ev.ev_id)),
                    selected_node=action.node,
                    node_type=node_type,
                    previous_node=int(info.get("previous_node", ev.current_node)),
                    delta_travel_time=float(info.get("delta_travel_time", 0.0)),
                    delta_energy_used=float(info.get("delta_energy_used", 0.0)),
                    charged_energy=float(info.get("charged_energy", 0.0)),
                    customer_served=served,
                )
            raw_r = step_info.raw_reward
            if rl_cfg is not None and rl_cfg.use_potential_shaping:
                phi_new = compute_potential(env, rl_cfg)
                shaped_r, shaping = apply_potential_shaping(raw_r, phi_prev, phi_new, rl_cfg.gamma)
                step_info.shaping_reward = shaping
                step_info.reward = shaped_r + (step_info.reward - raw_r)
                phi_prev = phi_new
            else:
                shaped_r = step_info.reward
                step_info.shaping_reward = 0.0
            step_infos.append(step_info)

        log_prob_sum = log_prob_sum + action.log_prob
        entropy_sum = entropy_sum + action.entropy
        num_decisions += 1

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
            trajectory.steps.append(
                StepRecord(
                    decoder_context=action.decoder_context if action.decoder_context is not None else policy.decoder_context(env, ev, node_embeddings),
                    feasible_mask=feasible_mask.copy(),
                    node=action.node,
                    ev_id=ev.ev_id,
                    log_prob=action.log_prob,
                    entropy=action.entropy,
                    node_probs=None,
                    q_candidate_indices=cand_idx,
                    q_candidate_probs=cand_probs,
                    reward=raw_r,
                    raw_reward=raw_r,
                    shaping_reward=shaping,
                    shaped_reward=float(step_info.reward) if step_info is not None else shaped_r,
                    step_info=step_info,
                    charge_portion=action.charge_portion,
                )
            )

    for ev in env.evs:
        if not ev.finished:
            env.auto_close_route(ev)

    done_reason = env.get_done_reason()
    j, breakdown, summary = env.score(weights)

    training_stats: dict[str, float] = {}
    if train_calc is not None and trajectory is not None:
        before_terminal = snapshot_env(env)
        terminal_info = train_calc.terminal_adjustment(before_terminal, summary)
        trajectory.terminal_info = terminal_info
        step_infos.append(terminal_info)
        training_stats = train_calc.finalize(j)
        trajectory.training_stats = training_stats
        # Alignment uses objective-aligned deltas only (pre-shaping step + full terminal objective)
        base_infos = []
        for s in step_infos[:-1]:
            base_infos.append(s)
        raw_terminal = StepRewardCalculator(weights, reward_scale=rl_cfg.reward_scale).terminal_adjustment(
            before_terminal, summary,
        )
        base_infos.append(raw_terminal)
        _, rel_mismatch, ok = train_calc.base.verify_alignment(base_infos, j)
        trajectory.objective = j
        trajectory.reconstructed_objective = sum(s.delta_objective for s in base_infos)
        trajectory.objective_mismatch = abs(trajectory.reconstructed_objective - j)
        trajectory.relative_objective_mismatch = rel_mismatch
        trajectory.reward_breakdown = _aggregate_reward_breakdown(step_infos)
        if not ok:
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
