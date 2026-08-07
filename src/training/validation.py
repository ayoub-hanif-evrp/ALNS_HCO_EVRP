"""Per-class validation with full fleet-size search.

Validation always runs the full fleet-size search (unlike training, which samples
a single fleet size) and reports feasibility and cost separately for each
customer-distribution class (C, R, RC).
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.fleet_search import FleetCandidate, fleet_search
from src.training.rollout import run_episode, run_multi_sample

INFEASIBILITY_PENALTY = 10000.0


def _neural_runner(policy, weights, mode, device, n_samples, charging_portions, use_horizon, use_depot_charging, max_steps_factor):
    def runner(instance, m):
        if mode == "multi_sample":
            res = run_multi_sample(
                instance, policy, m, weights, n_samples=n_samples, device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                use_depot_charging=use_depot_charging,
                max_steps_factor=max_steps_factor,
            )
        else:
            res = run_episode(
                instance, policy, m, weights, mode="greedy", device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                use_depot_charging=use_depot_charging,
                max_steps_factor=max_steps_factor,
            )
        return FleetCandidate(m=m, feasible=res.feasible, objective=res.objective, summary=res.summary, result=res)

    return runner


def _aggregate_metrics(
    costs, feas, dist, evs, durations, charge, price, wait, unserved, viol,
    batt_viol, cap_viol, tw_viol, served_rates, n_inst: int,
) -> dict:
    if n_inst == 0:
        return {}
    feasibility_rate = float(np.mean(feas))
    mean_cost = float(np.mean(costs))
    served_rate = float(np.mean(served_rates)) if served_rates else 0.0
    return {
        "num_instances": n_inst,
        "cost": mean_cost,
        "feasibility_rate": feasibility_rate,
        "feasibility_rate_pct": 100.0 * feasibility_rate,
        "score": mean_cost + INFEASIBILITY_PENALTY * (1.0 - feasibility_rate),
        "served_rate": served_rate,
        "served_rate_pct": 100.0 * served_rate,
        "avg_distance": float(np.mean(dist)),
        "avg_used_evs": float(np.mean(evs)),
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
        "avg_charging_time": float(np.mean(charge)),
        "avg_charging_price": float(np.mean(price)),
        "avg_waiting_time": float(np.mean(wait)),
        "avg_unserved": float(np.mean(unserved)),
        "avg_violation": float(np.mean(viol)),
        "avg_battery_violation": float(np.mean(batt_viol)) if batt_viol else 0.0,
        "avg_capacity_violation": float(np.mean(cap_viol)) if cap_viol else 0.0,
        "avg_time_window_violation": float(np.mean(tw_viol)) if tw_viol else 0.0,
    }


def validate_class(
    eval_ds,
    indices: list[int],
    policy,
    weights,
    *,
    mode: str = "greedy",
    device="cpu",
    n_samples: int = 16,
    charging_portions=None,
    use_horizon: bool = True,
    use_depot_charging: bool = True,
    max_steps_factor: int = 6,
    extra_margin: int = 5,
    selection_mode: str = "hierarchical",
    max_instances: int | None = None,
    exclude_names: set[str] | None = None,
) -> dict:
    """Validate the policy on a list of eval-dataset indices."""
    if exclude_names:
        indices = [i for i in indices if eval_ds.names[i] not in exclude_names]
    if max_instances is not None:
        indices = indices[:max_instances]
    if not indices:
        return {}

    runner = _neural_runner(
        policy, weights, mode, device, n_samples, charging_portions,
        use_horizon, use_depot_charging, max_steps_factor,
    )
    costs, feas, dist, evs = [], [], [], []
    durations, charge, price, wait = [], [], [], []
    unserved, viol, batt_viol, cap_viol, tw_viol, served_rates = [], [], [], [], [], []
    policy.eval()
    with torch.no_grad():
        for idx in indices:
            instance = eval_ds[idx]
            best, _ = fleet_search(
                instance, runner, extra_margin, selection_mode,
                fleet_cfg=config.get("fleet_search") or {},
            )
            s = best.summary
            n_c = max(instance.num_customers, 1)
            costs.append(best.objective)
            feas.append(1.0 if best.feasible else 0.0)
            dist.append(s.total_distance)
            evs.append(s.used_evs)
            durations.append(s.total_duration)
            charge.append(s.total_charging_time)
            price.append(s.total_charging_price)
            wait.append(s.total_waiting_time)
            unserved.append(s.unserved_customers)
            served_rates.append(1.0 - s.unserved_customers / n_c)
            batt_viol.append(s.total_battery_violation)
            cap_viol.append(s.total_capacity_violation)
            tw_viol.append(s.total_lateness)
            viol.append(
                s.total_battery_violation + s.total_capacity_violation
                + s.total_lateness + s.unserved_customers
            )

    return _aggregate_metrics(
        costs, feas, dist, evs, durations, charge, price, wait,
        unserved, viol, batt_viol, cap_viol, tw_viol, served_rates, len(indices),
    )


def validate(
    eval_ds,
    groups: dict[str, list[int]],
    policy,
    weights,
    *,
    classes: list[str] | None = None,
    exclude_names: set[str] | None = None,
    **kwargs,
) -> dict:
    """Validate per class; returns per-class metrics plus ``overall``."""
    if exclude_names:
        overlap = exclude_names & set(eval_ds.names)
        if overlap:
            raise RuntimeError(
                f"Validation eval set overlaps fixed training instances: {sorted(overlap)}"
            )
    per_class: dict[str, dict] = {}
    for cls, idxs in groups.items():
        if not idxs:
            continue
        metrics = validate_class(
            eval_ds, idxs, policy, weights, exclude_names=exclude_names, **kwargs
        )
        if metrics:
            per_class[cls] = metrics

    use = [c for c in (classes or list(per_class)) if c in per_class]
    if use:
        overall = {
            "cost": float(np.mean([per_class[c]["cost"] for c in use])),
            "feasibility_rate": float(np.mean([per_class[c]["feasibility_rate"] for c in use])),
            "served_rate": float(np.mean([per_class[c].get("served_rate", 0.0) for c in use])),
            "score": float(np.mean([per_class[c]["score"] for c in use])),
            "avg_used_evs": float(np.mean([per_class[c]["avg_used_evs"] for c in use])),
            "avg_distance": float(np.mean([per_class[c]["avg_distance"] for c in use])),
            "classes": use,
        }
    else:
        overall = {}
    return {**per_class, "overall": overall}
