"""Actor-critic training step (centralized fleet critic + optional Q, memory-safe)."""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from src.data.instance_class import CLASSES, instance_class
from src.env.cost import CostWeights, training_reward
from src.models.critic import CentralFleetValueCritic, QCritic
from src.training.actor_critic import RLConfig, ActorCriticLossOut, actor_critic_loss
from src.training.flat_schedule import sample_fleet_size
from src.training.rl_memory import cuda_memory_stats, reset_cuda_peak_stats
from src.training.rollout_ac import run_multi_sample_rollout_baseline
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def _group_rollouts_by_class(rollout_pairs: list[tuple]) -> dict[str, list[tuple]]:
    groups: dict[str, list[tuple]] = {c: [] for c in CLASSES}
    for pair in rollout_pairs:
        _, _, instance = pair
        cls = instance_class(str(getattr(instance, "instance_name", "") or ""))
        if cls not in groups:
            groups[cls] = []
        groups[cls].append(pair)
    return groups


def _class_equal_mean(tensors: list[torch.Tensor]) -> torch.Tensor:
    if not tensors:
        raise ValueError("empty class loss list")
    return torch.stack(tensors).mean()


def build_critics(
    policy,
    device: torch.device | str,
    *,
    build_q: bool = True,
) -> tuple[CentralFleetValueCritic, QCritic | None]:
    d = policy.embedding_dim
    value_critic = CentralFleetValueCritic(d).to(device)
    q_critic = QCritic(d).to(device) if build_q else None
    return value_critic, q_critic


def critic_parameters(value_critic: CentralFleetValueCritic, q_critic: QCritic | None) -> list[nn.Parameter]:
    params = list(value_critic.parameters())
    if q_critic is not None:
        params += list(q_critic.parameters())
    return params


def _flat_params(module: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().flatten().cpu() for p in module.parameters()])


def actor_critic_train_step(
    policy,
    value_critic: CentralFleetValueCritic,
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
    extra_margin: int,
    rng,
    rl_cfg: RLConfig,
    entropy_coef: float,
    actor_grad_clip: float,
    critic_grad_clip: float,
    min_decisions_warn: int | None = None,
    *,
    global_epoch: int = 0,
    cycle_id: int = 1,
    use_amp: bool = False,
    gradient_accumulation_steps: int = 1,
    scaler: torch.amp.GradScaler | None = None,
    critic_warmup_steps: int = 0,
    critic_updates_per_step: int = 2,
    environment_config: dict | None = None,
    parameter_delta_every_updates: int = 20,
    bc_episodes: list | None = None,
    bc_instance_by_name: dict | None = None,
    bc_regularization_cfg: dict | None = None,
    bc_decision_cache: list | None = None,
    bc_instance_embeddings: dict | None = None,
    fleet_cfg: dict | None = None,
) -> dict:
    """One actor-critic update: separate critic passes then actor pass."""
    policy.train()
    value_critic.train()
    if q_critic is not None:
        q_critic.train()

    device_t = torch.device(device) if isinstance(device, str) else device
    use_q_now = rl_cfg.q_critic_active(global_epoch) and q_critic is not None

    measure_param_delta = (
        int(parameter_delta_every_updates) > 0
        and (int(global_epoch) % int(parameter_delta_every_updates) == 0)
    )
    actor_params_before = _flat_params(policy) if measure_param_delta else None
    critic_params_before = _flat_params(value_critic) if measure_param_delta else None
    use_huber = str(getattr(rl_cfg, "critic_loss_type", "huber")).lower() == "huber"

    raw_costs, feas_flags = [], []
    used_evs_list, dist_list, charge_list, price_list, wait_list = [], [], [], [], []
    viol_list, unserved_list, duration_list, num_decisions_list = [], [], [], []
    served_rate_list, customers_list = [], []
    breakdowns: list[dict] = []
    lateness_list, battery_viol_list, capacity_viol_list = [], [], []
    customer_lateness_list, depot_lateness_list = [], []
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
    newly_expired_list, tw_masked_list, urgency_bonus_list = [], [], []
    unnecessary_station_list, expired_pen_list, time_waste_list = [], [], []
    feasible_cust_ratio_list, served_slack_list = [], []
    frac_customer_list, frac_station_list, frac_depot_list = [], [], []

    reset_cuda_peak_stats(device_t)
    accum = max(1, int(gradient_accumulation_steps))
    critic_passes = max(1, int(critic_updates_per_step))
    metrics_out: ActorCriticLossOut | None = None

    rollout_pairs: list[tuple] = []
    total_rollouts = 0
    per_class_stats: dict[str, dict[str, list]] = {
        c: {
            "objective": [], "served_rate": [], "feasibility": [],
            "serviceability": [], "instance_name": [],
        }
        for c in CLASSES
    }
    timing = {
        "decision_analysis_ms": 0.0,
        "controlled_closure_ms": 0.0,
        "policy_forward_ms": 0.0,
        "environment_step_ms": 0.0,
        "reward_diagnostics_ms": 0.0,
        "critic_update_ms": 0.0,
        "actor_update_ms": 0.0,
        "logging_ms": 0.0,
        "checkpoint_ms": 0.0,
        "dashboard_ms": 0.0,
    }
    t_epoch0 = time.perf_counter()

    for idx in batch_idx:
        instance = train_ds[idx]
        m = sample_fleet_size(
            instance, extra_margin, rng, fleet_cfg=fleet_cfg, mstar_map=(fleet_cfg or {}).get("_mstar_map_cache"),
        )

        rollouts, _ = run_multi_sample_rollout_baseline(
            instance, policy, m, weights, rl_cfg,
            device=device, charging_portions=charging_portions,
            use_horizon_check=use_horizon, use_depot_charging=use_depot_charging,
            max_steps_factor=max_steps_factor,
            min_decisions_warn=min_decisions_warn,
            cycle_id=cycle_id,
            environment_config=environment_config,
            use_amp=use_amp,
        )
        for result, trajectory in rollouts:
            total_rollouts += 1
            rollout_pairs.append((result, trajectory, instance))
            ts = getattr(trajectory, "training_stats", None) or {}
            for k in (
                "decision_analysis_ms", "controlled_closure_ms", "policy_forward_ms",
                "environment_step_ms", "reward_diagnostics_ms",
            ):
                timing[k] += float(ts.get(k, 0.0))

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
            newly_expired_list.append(float(ts.get("newly_expired_customers", 0.0)))
            tw_masked_list.append(float(ts.get("masked_by_time_window_count", 0.0)))
            urgency_bonus_list.append(float(ts.get("urgency_bonus_total", 0.0)))
            unnecessary_station_list.append(float(ts.get("unnecessary_station_actions", 0.0)))
            expired_pen_list.append(float(ts.get("expired_customer_penalty_total", 0.0)))
            time_waste_list.append(float(ts.get("time_waste_penalty_total", 0.0)))
            feasible_cust_ratio_list.append(float(ts.get("feasible_customer_ratio", 0.0)))
            served_slack_list.append(float(ts.get("served_customer_slack", 0.0)))
            _collect_episode_metrics(
                result, trajectory, instance, raw_costs, feas_flags, used_evs_list, dist_list,
                charge_list, price_list, wait_list, unserved_list, duration_list,
                customers_list, served_rate_list, viol_list, breakdowns,
                step_reward_means, terminal_reward_means, shaping_means, mismatch_list,
                rel_mismatch_list, lateness_list, battery_viol_list, capacity_viol_list, depot_viol_list,
                duplicate_list, route_balance_list, station_visits_list, min_battery_list,
                num_nodes_list, reconstructed_list, breakdown_acc,
                customer_lateness_list, depot_lateness_list,
            )
            cls = instance_class(str(getattr(instance, "instance_name", "") or ""))
            n_cust = max(int(instance.num_customers), 1)
            served = 1.0 - float(result.summary.unserved_customers) / n_cust
            gs = getattr(result.env, "last_global_serviceability", None)
            if gs is not None:
                svc = float(getattr(gs, "globally_serviceable_customers_count", 0)) / n_cust
            else:
                svc = served
            per_class_stats[cls]["objective"].append(float(result.objective))
            per_class_stats[cls]["served_rate"].append(served)
            per_class_stats[cls]["feasibility"].append(float(result.feasible))
            per_class_stats[cls]["serviceability"].append(svc)
            per_class_stats[cls]["instance_name"].append(
                str(getattr(instance, "instance_name", "") or "")
            )

    n_roll = max(1, len(rollout_pairs))
    if not rollout_pairs:
        return {"loss": 0.0, "algorithm": "actor_critic"}

    by_class = _group_rollouts_by_class(rollout_pairs)

    def _compute_loss(pair, **kwargs):
        _, trajectory, _ = pair
        return actor_critic_loss(
            trajectory, value_critic, q_critic, rl_cfg,
            multi_sample_advantage=(
                trajectory.multi_sample_advantage
                if rl_cfg.use_multi_sample_rollout_baseline
                else None
            ),
            device=device,
            global_epoch=global_epoch,
            use_q_critic=use_q_now,
            use_huber=use_huber,
            use_amp=use_amp,
            **kwargs,
        )

    # Metrics from the last backward pass (avoid an extra forward-only pass).
    critic_only_warmup = global_epoch < int(critic_warmup_steps)
    actor_grad_norm = torch.tensor(0.0)
    critic_grad_norm = torch.tensor(0.0)
    grad_norm = torch.tensor(0.0)
    class_actor_loss_log: dict[str, float] = {}
    class_critic_loss_log: dict[str, float] = {}
    class_entropy_log: dict[str, float] = {}
    bc_coef = 0.0
    bc_loss_val = 0.0

    for _pass in range(critic_passes):
        t_c0 = time.perf_counter()
        critic_optimizer.zero_grad(set_to_none=True)
        class_critic_terms: list[torch.Tensor] = []
        for cls in CLASSES:
            pairs = by_class.get(cls) or []
            if not pairs:
                continue
            outs = [_compute_loss(pair, critic_only=True) for pair in pairs]
            metrics_out = outs[-1]
            cls_loss = torch.stack([o.critic_loss for o in outs]).mean()
            class_critic_terms.append(cls_loss)
            class_critic_loss_log[cls] = float(cls_loss.detach())
        if not class_critic_terms:
            continue
        macro_critic = _class_equal_mean(class_critic_terms)
        scaled = (rl_cfg.critic_coef * macro_critic) / accum
        if use_amp and device_t.type == "cuda" and scaler is not None:
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        amp_critic = use_amp and device_t.type == "cuda" and scaler is not None
        if amp_critic:
            scaler.unscale_(critic_optimizer)
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            critic_parameters(value_critic, q_critic), critic_grad_clip,
        )
        if not torch.isfinite(critic_grad_norm):
            critic_optimizer.zero_grad(set_to_none=True)
            LOGGER.warning("Skipping critic step: non-finite gradient norm")
            # After unscale_(), GradScaler requires update() even when step() is skipped,
            # otherwise the next critic pass raises "unscale_() has already been called".
            if amp_critic:
                scaler.update()
        elif amp_critic:
            scaler.step(critic_optimizer)
            scaler.update()
        else:
            critic_optimizer.step()
        timing["critic_update_ms"] += (time.perf_counter() - t_c0) * 1000.0

    if not critic_only_warmup:
        t_a0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        class_actor_terms: list[torch.Tensor] = []
        class_entropy_terms: list[torch.Tensor] = []
        class_q_terms: list[torch.Tensor] = []
        for cls in CLASSES:
            pairs = by_class.get(cls) or []
            if not pairs:
                continue
            outs = [_compute_loss(pair, actor_only=True) for pair in pairs]
            metrics_out = outs[-1]
            # Advantages already normalized per-trajectory inside actor_critic_loss.
            cls_actor = torch.stack([o.actor_loss for o in outs]).mean()
            cls_ent = torch.stack([o.entropy_term for o in outs]).mean()
            class_actor_terms.append(cls_actor)
            class_entropy_terms.append(cls_ent)
            class_actor_loss_log[cls] = float(cls_actor.detach())
            class_entropy_log[cls] = float(cls_ent.detach())
            if use_q_now:
                q_terms = [o.q_critic_loss for o in outs if o.q_critic_loss.requires_grad]
                if q_terms:
                    class_q_terms.append(torch.stack(q_terms).mean())
        if class_actor_terms:
            macro_actor = _class_equal_mean(class_actor_terms)
            macro_entropy = _class_equal_mean(class_entropy_terms)
            scaled = (macro_actor - entropy_coef * macro_entropy) / accum
            if class_q_terms:
                scaled = scaled + (rl_cfg.q_critic_coef * _class_equal_mean(class_q_terms)) / accum
            # Optional decaying BC regularization (warm-start fine-tune).
            # Prefer cached exact-mask decisions (no CPU re-analysis).
            if bc_regularization_cfg:
                from src.training.behavior_cloning import (
                    bc_coef_at_update,
                    bc_loss_on_cached_batch,
                    bc_loss_on_demo_batch,
                )
                bc_coef = bc_coef_at_update(
                    {"bc_regularization": bc_regularization_cfg}, global_epoch,
                )
                if bc_coef > 0.0:
                    if bc_decision_cache and bc_instance_by_name and bc_instance_embeddings:
                        bc_loss, _bc_m = bc_loss_on_cached_batch(
                            policy,
                            bc_decision_cache,
                            bc_instance_by_name,
                            bc_instance_embeddings,
                            device=device_t,
                            batch_decisions=int(
                                bc_regularization_cfg.get("batch_decisions", 32)
                            ),
                            rng=rng,
                            loss_mode=str(
                                bc_regularization_cfg.get("loss_mode", "ce")
                            ),
                            label_smoothing=float(
                                bc_regularization_cfg.get("label_smoothing", 0.05)
                            ),
                        )
                    elif bc_episodes and bc_instance_by_name:
                        bc_loss, _bc_m = bc_loss_on_demo_batch(
                            policy, bc_episodes, bc_instance_by_name,
                            device=device_t,
                            batch_episodes=int(bc_regularization_cfg.get("batch_episodes", 1)),
                            rng=rng,
                        )
                    else:
                        bc_loss = None
                    if bc_loss is not None:
                        scaled = scaled + (bc_coef * bc_loss) / accum
                        bc_loss_val = float(bc_loss.detach())
            if use_amp and device_t.type == "cuda" and scaler is not None:
                scaler.scale(scaled).backward()
            else:
                scaled.backward()

            amp_actor = use_amp and device_t.type == "cuda" and scaler is not None
            if amp_actor:
                scaler.unscale_(optimizer)
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(list(policy.parameters()), actor_grad_clip)
            grad_norm = actor_grad_norm
            if not torch.isfinite(actor_grad_norm):
                optimizer.zero_grad(set_to_none=True)
                LOGGER.warning("Skipping actor step: non-finite gradient norm")
                if amp_actor:
                    scaler.update()
            elif amp_actor:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
        timing["actor_update_ms"] += (time.perf_counter() - t_a0) * 1000.0

    if measure_param_delta and actor_params_before is not None and critic_params_before is not None:
        actor_param_delta = float(torch.norm(_flat_params(policy) - actor_params_before).item())
        critic_param_delta = float(torch.norm(_flat_params(value_critic) - critic_params_before).item())
    else:
        actor_param_delta = 0.0
        critic_param_delta = 0.0

    if metrics_out is not None:
        actor_losses.append(metrics_out.actor_loss.detach())
        critic_losses.append(metrics_out.critic_loss.detach())
        q_losses.append(metrics_out.q_critic_loss.detach())
        entropies.append(metrics_out.entropy_term.detach())
        adv_list.append(float(metrics_out.advantages.mean().detach()))
        adv_min_list.append(metrics_out.advantage_min)
        adv_max_list.append(metrics_out.advantage_max)
        log_prob_list.append(metrics_out.mean_log_prob)
        ev_list.append(metrics_out.explained_var)
        cf_adv_list.append(metrics_out.counterfactual_advantage_mean)
        value_mean_list.append(metrics_out.value_mean)
        value_std_list.append(metrics_out.value_std)
        return_mean_list.append(metrics_out.return_mean)
        return_std_list.append(metrics_out.return_std)
        q_cand_list.append(metrics_out.mean_q_candidates)

    mem = cuda_memory_stats(device_t)
    breakdown_means = {}
    if breakdowns:
        breakdown_means = {
            f"mean_{k}": float(np.mean([bd.get(k, 0.0) for bd in breakdowns])) for k in breakdowns[0]
        }

    from src.env.analysis_counters import get_cycle_counters, snapshot_for_log
    timing["total_epoch_ms"] = (time.perf_counter() - t_epoch0) * 1000.0
    counter_snap = snapshot_for_log(get_cycle_counters())

    # Equal-weight macro metrics across present classes.
    class_metric_out: dict[str, float] = {}
    present_classes = [c for c in CLASSES if per_class_stats[c]["objective"]]
    for cls in CLASSES:
        st = per_class_stats[cls]
        if not st["objective"]:
            continue
        class_metric_out[f"objective_{cls}"] = float(np.mean(st["objective"]))
        class_metric_out[f"served_rate_{cls}"] = float(np.mean(st["served_rate"]))
        class_metric_out[f"feasibility_{cls}"] = float(np.mean(st["feasibility"]))
        class_metric_out[f"serviceability_{cls}"] = float(np.mean(st["serviceability"]))
        if cls in class_actor_loss_log:
            class_metric_out[f"actor_loss_{cls}"] = class_actor_loss_log[cls]
        if cls in class_critic_loss_log:
            class_metric_out[f"critic_loss_{cls}"] = class_critic_loss_log[cls]
        if cls in class_entropy_log:
            class_metric_out[f"entropy_{cls}"] = class_entropy_log[cls]
        if st["instance_name"]:
            class_metric_out[f"instance_name_{cls}"] = st["instance_name"][0]

    def _macro_of(key: str) -> float:
        vals = [class_metric_out[f"{key}_{c}"] for c in present_classes if f"{key}_{c}" in class_metric_out]
        return float(np.mean(vals)) if vals else 0.0

    macro_objective = _macro_of("objective")
    macro_served = _macro_of("served_rate")
    macro_feas = _macro_of("feasibility")
    macro_svc = _macro_of("serviceability")

    # Prefer class-equal logged losses when available.
    if class_actor_loss_log:
        logged_actor = float(np.mean(list(class_actor_loss_log.values())))
    else:
        logged_actor = float(torch.stack(actor_losses).mean().item()) if actor_losses else 0.0
    if class_critic_loss_log:
        logged_critic = float(np.mean(list(class_critic_loss_log.values())))
    else:
        logged_critic = float(torch.stack(critic_losses).mean().item()) if critic_losses else 0.0
    if class_entropy_log:
        logged_entropy = float(np.mean(list(class_entropy_log.values())))
    else:
        logged_entropy = float(torch.stack(entropies).mean().item()) if entropies else 0.0

    return {
        "algorithm": "actor_critic",
        "macro_update": float(len(present_classes) >= 2),
        "n_classes_in_update": float(len(present_classes)),
        "bc_regularization_coef": float(bc_coef),
        "bc_regularization_loss": float(bc_loss_val),
        **{f"timing_{k}": float(v) for k, v in timing.items()},
        **{f"dup_{k}": int(v) for k, v in counter_snap.items()},
        **class_metric_out,
        "macro_objective": macro_objective,
        "macro_served_rate": macro_served,
        "macro_feasibility": macro_feas,
        "macro_serviceability": macro_svc,
        # Primary curves use equal-weight macro averages (not raw mixed objective).
        "cost": macro_objective if present_classes else float(np.mean(raw_costs)),
        "cost_raw": macro_objective if present_classes else float(np.mean(raw_costs)),
        "final_objective": macro_objective if present_classes else float(np.mean(raw_costs)),
        "final_original_objective": macro_objective if present_classes else float(np.mean(raw_costs)),
        "reward": float(np.mean(shaped_return_list)) if shaped_return_list else float(
            np.mean([training_reward(c, rl_cfg.reward_scale) for c in raw_costs])
        ),
        "shaped_training_return": float(np.mean(shaped_return_list)) if shaped_return_list else 0.0,
        "raw_objective_return": float(np.mean(raw_obj_return_list)) if raw_obj_return_list else float(
            np.mean([training_reward(c, rl_cfg.reward_scale) for c in raw_costs])
        ),
        "reward_scale": rl_cfg.reward_scale,
        "loss": logged_actor,
        "actor_loss": logged_actor,
        "critic_loss": logged_critic,
        "q_critic_loss": float(torch.stack(q_losses).mean().item()) if q_losses else 0.0,
        "entropy": logged_entropy,
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
        "feasibility_rate": macro_feas if present_classes else float(np.mean(feas_flags)),
        "feasibility_rate_pct": 100.0 * (macro_feas if present_classes else float(np.mean(feas_flags))),
        "served_rate": macro_served if present_classes else float(np.mean(served_rate_list)),
        "served_rate_pct": 100.0 * (macro_served if present_classes else float(np.mean(served_rate_list))),
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
        "reconstructed_objective": float(np.mean(reconstructed_list)) if reconstructed_list else 0.0,
        "reconstructed_original_objective": float(np.mean(reconstructed_list)) if reconstructed_list else 0.0,
        "reconstructed_training_cost": float(np.mean(reconstructed_list)) if reconstructed_list else 0.0,
        "original_objective_mismatch": float(np.mean(mismatch_list)) if mismatch_list else 0.0,
        "relative_original_objective_mismatch": float(np.mean(rel_mismatch_list)) if rel_mismatch_list else 0.0,
        "newly_expired_customers": float(np.mean(newly_expired_list)) if newly_expired_list else 0.0,
        "masked_by_time_window_count": float(np.mean(tw_masked_list)) if tw_masked_list else 0.0,
        "urgency_bonus_total": float(np.mean(urgency_bonus_list)) if urgency_bonus_list else 0.0,
        "unnecessary_station_actions": float(np.mean(unnecessary_station_list)) if unnecessary_station_list else 0.0,
        "expired_customer_penalty_total": float(np.mean(expired_pen_list)) if expired_pen_list else 0.0,
        "time_waste_penalty_total": float(np.mean(time_waste_list)) if time_waste_list else 0.0,
        "feasible_customer_ratio": float(np.mean(feasible_cust_ratio_list)) if feasible_cust_ratio_list else 0.0,
        "served_customer_slack": float(np.mean(served_slack_list)) if served_slack_list else 0.0,
        "num_decisions": float(np.mean(num_decisions_list)) if num_decisions_list else 0.0,
        "mean_feasible_actions": float(np.mean(feas_action_list)) if feas_action_list else 0.0,
        "mean_q_candidates": float(np.mean(q_cand_list)) if q_cand_list else 0.0,
        "avg_lateness": float(np.mean(lateness_list)) if lateness_list else 0.0,
        "customer_lateness": float(np.mean(customer_lateness_list)) if customer_lateness_list else 0.0,
        "depot_lateness": float(np.mean(depot_lateness_list)) if depot_lateness_list else 0.0,
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
            (torch.stack(actor_losses).mean().item() if actor_losses else 0.0)
            + (torch.stack(critic_losses).mean().item() if critic_losses else 0.0)
            + (torch.stack(q_losses).mean().item() if q_losses else 0.0)
        ),
        "multi_sample_K": rl_cfg.effective_multi_sample_k(),
        "use_multi_sample_rollout_baseline": float(rl_cfg.use_multi_sample_rollout_baseline),
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
    customer_lateness_list, depot_lateness_list,
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
    customer_lateness_list.append(float(getattr(summary, "customer_lateness", 0.0)))
    depot_lateness_list.append(float(getattr(summary, "depot_lateness", 0.0)))
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
    mismatch_list.append(float(getattr(trajectory, "original_objective_mismatch", trajectory.objective_mismatch)))
    rel_mismatch_list.append(float(getattr(trajectory, "relative_original_objective_mismatch", getattr(trajectory, "relative_objective_mismatch", 0.0))))
    reconstructed_list.append(float(getattr(trajectory, "reconstructed_original_objective", getattr(trajectory, "reconstructed_objective", result.objective))))
    for k, v in (getattr(trajectory, "reward_breakdown", None) or {}).items():
        breakdown_acc[k] = breakdown_acc.get(k, 0.0) + float(v)
