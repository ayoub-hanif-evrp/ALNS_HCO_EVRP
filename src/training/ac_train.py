"""Actor-critic training step (centralized V + optional Q, memory-safe)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.env.cost import CostWeights, training_reward
from src.models.critic import QCritic, ValueCritic
from src.training.actor_critic import RLConfig, actor_critic_loss
from src.training.curriculum import Phase, sample_phase_m
from src.training.rl_memory import cuda_memory_stats, reset_cuda_peak_stats
from src.training.rollout_ac import run_pomo_episodes
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def build_critics(
    policy,
    device: torch.device | str,
    *,
    build_q: bool = True,
) -> tuple[ValueCritic, QCritic | None]:
    d = policy.embedding_dim
    value_critic = ValueCritic(d).to(device)
    q_critic = QCritic(d).to(device) if build_q else None
    return value_critic, q_critic


def critic_parameters(value_critic: ValueCritic, q_critic: QCritic | None) -> list[nn.Parameter]:
    params = list(value_critic.parameters())
    if q_critic is not None:
        params += list(q_critic.parameters())
    return params


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


def actor_critic_train_step(
    policy,
    value_critic: ValueCritic,
    q_critic: QCritic | None,
    optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    train_ds,
    batch_idx: list[int],
    weights: CostWeights,
    device: torch.device | str,
    charging_portions: list[float],
    use_horizon: bool,
    use_depot_charging: bool,
    max_steps_factor: int,
    phase: Phase,
    rng,
    rl_cfg: RLConfig,
    entropy_coef: float,
    grad_clip: float,
    min_decisions_warn: int | None = None,
    *,
    global_epoch: int = 0,
    cycle_id: int = 1,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    critic_warmup_steps: int = 0,
    critic_updates_per_step: int = 1,
) -> dict:
    """One actor-critic update with per-rollout backward for memory safety."""
    policy.train()
    value_critic.train()
    if q_critic is not None:
        q_critic.train()

    device_t = torch.device(device) if isinstance(device, str) else device
    use_q_now = rl_cfg.q_critic_active(global_epoch) and q_critic is not None

    raw_costs, feas_flags = [], []
    used_evs_list, dist_list, charge_list, price_list, wait_list = [], [], [], [], []
    viol_list, unserved_list, duration_list, num_decisions_list = [], [], [], []
    served_rate_list, customers_list = [], []
    breakdowns: list[dict] = []
    lateness_list, battery_viol_list, capacity_viol_list = [], [], []
    depot_viol_list, duplicate_list = [], []
    route_balance_list, station_visits_list, min_battery_list = [], [], []
    num_nodes_list = []

    actor_losses, critic_losses, q_losses, entropies = [], [], [], []
    adv_list, ev_list, cf_adv_list = [], [], []
    value_mean_list, value_std_list = [], []
    return_mean_list, return_std_list = [], []
    step_reward_means, terminal_reward_means, shaping_means = [], [], []
    mismatch_list, rel_mismatch_list = [], []
    reconstructed_list: list[float] = []
    breakdown_acc: dict[str, float] = {}
    feas_action_list, q_cand_list = [], []
    log_prob_list, adv_min_list, adv_max_list = [], [], []
    shaped_return_list, raw_obj_return_list = [], []
    served_bonus_list, early_depot_list = [], []
    early_depot_penalty_list, step_penalty_list, terminal_unserved_penalty_list = [], [], []
    terminal_coef_list = []
    frac_customer_list, frac_station_list, frac_depot_list = [], [], []

    actor_params_before = _flat_params(policy)
    critic_params_before = _flat_params(value_critic)

    optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    reset_cuda_peak_stats(device_t)

    total_rollouts = 0
    accum = max(1, int(gradient_accumulation_steps))

    for idx in batch_idx:
        instance = train_ds[idx]
        m = sample_phase_m(instance, phase.extra_margin, rng)

        rollouts, _ = run_pomo_episodes(
            instance, policy, m, weights, rl_cfg,
            device=device, charging_portions=charging_portions,
            use_horizon_check=use_horizon, use_depot_charging=use_depot_charging,
            max_steps_factor=max_steps_factor,
            min_decisions_warn=min_decisions_warn,
            cycle_id=cycle_id,
        )
        n_roll = max(1, len(rollouts))
        for result, trajectory in rollouts:
            total_rollouts += 1

            def _compute_loss():
                return actor_critic_loss(
                    trajectory, value_critic, q_critic, rl_cfg,
                    pomo_advantage=trajectory.pomo_advantage if rl_cfg.use_pomo else None,
                    device=device,
                    global_epoch=global_epoch,
                    use_q_critic=use_q_now,
                )

            # Keep loss / critic regression in fp32 — large terminal rewards overflow fp16 grads.
            out = _compute_loss()
            critic_mult = max(1, int(critic_updates_per_step))
            scaled = (out.actor_loss - rl_cfg.entropy_coef * out.entropy_term) / (n_roll * accum)
            scaled = scaled + (rl_cfg.critic_coef * critic_mult * out.critic_loss) / (n_roll * accum)
            if use_q_now and out.q_critic_loss.requires_grad:
                scaled = scaled + (rl_cfg.q_critic_coef * out.q_critic_loss) / (n_roll * accum)
            if use_amp and device_t.type == "cuda" and scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            actor_losses.append(out.actor_loss.detach())
            critic_losses.append(out.critic_loss.detach())
            q_losses.append(out.q_critic_loss.detach())
            entropies.append(out.entropy_term.detach())
            adv_list.append(float(out.advantages.mean().detach()))
            adv_min_list.append(out.advantage_min)
            adv_max_list.append(out.advantage_max)
            log_prob_list.append(out.mean_log_prob)
            ev_list.append(out.explained_var)
            cf_adv_list.append(out.counterfactual_advantage_mean)
            value_mean_list.append(out.value_mean)
            value_std_list.append(out.value_std)
            return_mean_list.append(out.return_mean)
            return_std_list.append(out.return_std)
            q_cand_list.append(out.mean_q_candidates)
            num_decisions_list.append(trajectory.num_steps)
            feas_action_list.append(trajectory.mean_feasible_actions())
            ts = getattr(trajectory, "training_stats", None) or {}
            shaped_return_list.append(float(ts.get("shaped_training_return", 0.0)))
            raw_obj_return_list.append(float(ts.get("raw_objective_return", 0.0)))
            served_bonus_list.append(float(ts.get("served_bonus_total", 0.0)))
            early_depot_list.append(float(ts.get("early_depot_returns", 0.0)))
            early_depot_penalty_list.append(float(ts.get("early_depot_penalty_total", 0.0)))
            step_penalty_list.append(float(ts.get("step_penalty_total", 0.0)))
            terminal_unserved_penalty_list.append(float(ts.get("terminal_unserved_penalty_scaled", 0.0)))
            terminal_coef_list.append(float(ts.get("terminal_penalty_coef", 1.0)))
            frac_customer_list.append(float(ts.get("fraction_customer_actions", 0.0)))
            frac_station_list.append(float(ts.get("fraction_station_actions", 0.0)))
            frac_depot_list.append(float(ts.get("fraction_depot_actions", 0.0)))
            _collect_episode_metrics(
                result, trajectory, instance, raw_costs, feas_flags, used_evs_list, dist_list,
                charge_list, price_list, wait_list, unserved_list, duration_list,
                customers_list, served_rate_list, viol_list, breakdowns,
                step_reward_means, terminal_reward_means, shaping_means, mismatch_list,
                rel_mismatch_list, lateness_list, battery_viol_list, capacity_viol_list, depot_viol_list,
                duplicate_list, route_balance_list, station_visits_list, min_battery_list,
                num_nodes_list, reconstructed_list, breakdown_acc,
            )
            del out

            if device_t.type == "cuda" and total_rollouts % max(1, n_roll) == 0:
                torch.cuda.empty_cache()

    if not actor_losses:
        return {"loss": 0.0, "algorithm": "actor_critic"}

    critic_only = global_epoch < int(critic_warmup_steps)
    actor_grad_norm = torch.tensor(0.0)
    critic_grad_norm = torch.tensor(0.0)

    if use_amp and device_t.type == "cuda" and scaler is not None:
        scaler.unscale_(optimizer)
        scaler.unscale_(critic_optimizer)
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.parameters()), grad_clip)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_parameters(value_critic, q_critic), grad_clip,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + critic_parameters(value_critic, q_critic), grad_clip,
        )
        if not torch.isfinite(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            scaler.update()
            LOGGER.warning("Skipping optimizer step: non-finite gradient norm (%.4g)", float(grad_norm))
            grad_norm = torch.tensor(float("nan"))
        else:
            if not critic_only:
                scaler.step(optimizer)
            scaler.step(critic_optimizer)
            scaler.update()
    else:
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.parameters()), grad_clip)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_parameters(value_critic, q_critic), grad_clip,
        )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            list(policy.parameters()) + critic_parameters(value_critic, q_critic), grad_clip,
        )
        if not critic_only:
            optimizer.step()
        critic_optimizer.step()

    actor_param_delta = float(torch.norm(_flat_params(policy) - actor_params_before).item())
    critic_param_delta = float(torch.norm(_flat_params(value_critic) - critic_params_before).item())

    mem = cuda_memory_stats(device_t)
    breakdown_means = {}
    if breakdowns:
        breakdown_means = {
            f"mean_{k}": float(np.mean([bd.get(k, 0.0) for bd in breakdowns])) for k in breakdowns[0]
        }

    return {
        "algorithm": "actor_critic",
        "cost": float(np.mean(raw_costs)),
        "cost_raw": float(np.mean(raw_costs)),
        "reward": float(np.mean(shaped_return_list)) if shaped_return_list else float(
            np.mean([training_reward(c, rl_cfg.reward_scale) for c in raw_costs])
        ),
        "shaped_training_return": float(np.mean(shaped_return_list)) if shaped_return_list else 0.0,
        "raw_objective_return": float(np.mean(raw_obj_return_list)) if raw_obj_return_list else float(
            np.mean([training_reward(c, rl_cfg.reward_scale) for c in raw_costs])
        ),
        "reward_scale": rl_cfg.reward_scale,
        "loss": float(torch.stack(actor_losses).mean().item()),
        "actor_loss": float(torch.stack(actor_losses).mean().item()),
        "critic_loss": float(torch.stack(critic_losses).mean().item()),
        "q_critic_loss": float(torch.stack(q_losses).mean().item()) if q_losses else 0.0,
        "entropy": float(torch.stack(entropies).mean().item()),
        "advantage": float(np.mean(adv_list)),
        "advantage_std": float(np.std(adv_list)) if adv_list else 0.0,
        "advantage_min": float(np.min(adv_min_list)) if adv_min_list else 0.0,
        "advantage_max": float(np.max(adv_max_list)) if adv_max_list else 0.0,
        "mean_log_prob": float(np.mean(log_prob_list)) if log_prob_list else 0.0,
        "actor_grad_norm": float(actor_grad_norm),
        "critic_grad_norm": float(critic_grad_norm),
        "actor_param_delta_norm": actor_param_delta,
        "critic_param_delta_norm": critic_param_delta,
        "served_bonus_total": float(np.mean(served_bonus_list)) if served_bonus_list else 0.0,
        "early_depot_returns": float(np.mean(early_depot_list)) if early_depot_list else 0.0,
        "early_depot_penalty_total": float(np.mean(early_depot_penalty_list)) if early_depot_penalty_list else 0.0,
        "step_penalty_total": float(np.mean(step_penalty_list)) if step_penalty_list else 0.0,
        "terminal_unserved_penalty_scaled": float(np.mean(terminal_unserved_penalty_list)) if terminal_unserved_penalty_list else 0.0,
        "terminal_penalty_coef": float(np.mean(terminal_coef_list)) if terminal_coef_list else 1.0,
        "fraction_customer_actions": float(np.mean(frac_customer_list)) if frac_customer_list else 0.0,
        "fraction_station_actions": float(np.mean(frac_station_list)) if frac_station_list else 0.0,
        "fraction_depot_actions": float(np.mean(frac_depot_list)) if frac_depot_list else 0.0,
        "counterfactual_advantage": float(np.mean(cf_adv_list)) if cf_adv_list else 0.0,
        "critic_explained_variance": float(np.mean(ev_list)),
        "value_mean": float(np.mean(value_mean_list)) if value_mean_list else 0.0,
        "value_std": float(np.mean(value_std_list)) if value_std_list else 0.0,
        "return_mean": float(np.mean(return_mean_list)) if return_mean_list else 0.0,
        "return_std": float(np.mean(return_std_list)) if return_std_list else 0.0,
        "grad_norm": float(grad_norm),
        "feasibility_rate": float(np.mean(feas_flags)),
        "feasibility_rate_pct": 100.0 * float(np.mean(feas_flags)),
        "served_rate": float(np.mean(served_rate_list)),
        "served_rate_pct": 100.0 * float(np.mean(served_rate_list)),
        "avg_used_evs": float(np.mean(used_evs_list)),
        "avg_distance": float(np.mean(dist_list)),
        "avg_duration": float(np.mean(duration_list)),
        "avg_charging_time": float(np.mean(charge_list)),
        "avg_charging_price": float(np.mean(price_list)),
        "avg_waiting_time": float(np.mean(wait_list)),
        "avg_violation": float(np.mean(viol_list)),
        "unserved_customers": float(np.mean(unserved_list)),
        "mean_step_reward": float(np.mean(step_reward_means)) if step_reward_means else 0.0,
        "mean_terminal_reward": float(np.mean(terminal_reward_means)) if terminal_reward_means else 0.0,
        "mean_shaping_reward": float(np.mean(shaping_means)) if shaping_means else 0.0,
        "objective_mismatch": float(np.mean(mismatch_list)) if mismatch_list else 0.0,
        "relative_objective_mismatch": float(np.mean(rel_mismatch_list)) if rel_mismatch_list else 0.0,
        "final_objective": float(np.mean(raw_costs)),
        "reconstructed_objective": float(np.mean(reconstructed_list)) if reconstructed_list else 0.0,
        "num_decisions": float(np.mean(num_decisions_list)) if num_decisions_list else 0.0,
        "mean_feasible_actions": float(np.mean(feas_action_list)) if feas_action_list else 0.0,
        "mean_q_candidates": float(np.mean(q_cand_list)) if q_cand_list else 0.0,
        "avg_lateness": float(np.mean(lateness_list)) if lateness_list else 0.0,
        "avg_battery_violation": float(np.mean(battery_viol_list)) if battery_viol_list else 0.0,
        "avg_capacity_violation": float(np.mean(capacity_viol_list)) if capacity_viol_list else 0.0,
        "avg_depot_violation": float(np.mean(depot_viol_list)) if depot_viol_list else 0.0,
        "avg_duplicate_visits": float(np.mean(duplicate_list)) if duplicate_list else 0.0,
        "route_balance_cv": float(np.mean(route_balance_list)) if route_balance_list else 0.0,
        "charging_station_visits": float(np.mean(station_visits_list)) if station_visits_list else 0.0,
        "min_remaining_battery": float(np.mean(min_battery_list)) if min_battery_list else 0.0,
        "num_nodes": float(np.mean(num_nodes_list)) if num_nodes_list else 0.0,
        "feasible_action_ratio": (
            float(np.mean(feas_action_list)) / max(float(np.mean(num_nodes_list)), 1.0)
            if feas_action_list and num_nodes_list else 0.0
        ),
        "total_loss": float(
            torch.stack(actor_losses).mean().item() + torch.stack(critic_losses).mean().item()
            + (torch.stack(q_losses).mean().item() if q_losses else 0.0)
        ),
        "pomo_K": rl_cfg.effective_pomo_k(),
        "use_q_critic": float(use_q_now),
        "use_amp": float(use_amp),
        **mem,
        **breakdown_means,
        **{f"mean_{k}": v / max(total_rollouts, 1) for k, v in breakdown_acc.items()},
    }


def _collect_episode_metrics(
    result, trajectory, instance, raw_costs, feas_flags, used_evs_list, dist_list,
    charge_list, price_list, wait_list, unserved_list, duration_list,
    customers_list, served_rate_list, viol_list, breakdowns,
    step_reward_means, terminal_reward_means, shaping_means, mismatch_list,
    rel_mismatch_list, lateness_list, battery_viol_list, capacity_viol_list, depot_viol_list,
    duplicate_list, route_balance_list, station_visits_list, min_battery_list,
    num_nodes_list, reconstructed_list, breakdown_acc,
):
    summary = result.summary
    raw_costs.append(float(result.objective))
    feas_flags.append(float(result.feasible))
    used_evs_list.append(summary.used_evs)
    dist_list.append(summary.total_distance)
    charge_list.append(summary.total_charging_time)
    price_list.append(summary.total_charging_price)
    wait_list.append(summary.total_waiting_time)
    unserved_list.append(summary.unserved_customers)
    duration_list.append(summary.total_duration)
    n_cust = max(int(instance.num_customers), 1)
    customers_list.append(n_cust)
    served_rate_list.append(1.0 - summary.unserved_customers / n_cust)
    viol_list.append(
        summary.total_battery_violation
        + summary.total_capacity_violation
        + summary.total_lateness
        + summary.unserved_customers
    )
    lateness_list.append(summary.total_lateness)
    battery_viol_list.append(summary.total_battery_violation)
    capacity_viol_list.append(summary.total_capacity_violation)
    depot_viol_list.append(float(summary.depot_return_violation))
    duplicate_list.append(float(summary.duplicate_customer_visits))
    breakdowns.append(result.breakdown)
    num_nodes_list.append(int(instance.num_nodes))

    dists = [ev.route_distance for ev in result.env.evs if ev.used()]
    if len(dists) >= 2 and float(np.mean(dists)) > 1e-9:
        route_balance_list.append(float(np.std(dists) / np.mean(dists)))
    else:
        route_balance_list.append(0.0)

    station_visits_list.append(
        float(sum(1 for s in trajectory.steps if instance.nodes[s.node].is_charging_station))
    )
    min_battery_list.append(float(min((ev.remaining_battery for ev in result.env.evs), default=0.0)))

    if trajectory.steps:
        step_reward_means.append(float(np.mean([s.raw_reward for s in trajectory.steps])))
        shaping_means.append(float(np.mean([s.shaping_reward for s in trajectory.steps])))
    if trajectory.terminal_info:
        terminal_reward_means.append(float(trajectory.terminal_info.reward))
    mismatch_list.append(float(trajectory.objective_mismatch))
    rel_mismatch_list.append(float(getattr(trajectory, "relative_objective_mismatch", 0.0)))
    reconstructed_list.append(float(getattr(trajectory, "reconstructed_objective", result.objective)))
    for k, v in (getattr(trajectory, "reward_breakdown", None) or {}).items():
        breakdown_acc[k] = breakdown_acc.get(k, 0.0) + float(v)
