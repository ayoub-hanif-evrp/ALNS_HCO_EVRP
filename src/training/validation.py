"""Per-class validation with full fleet-size search.

Validation always runs the full fleet-size search (unlike training, which samples
a single fleet size) and reports feasibility and cost separately for each
customer-distribution class (C, R, RC). The headline number is the validation
score

    score = validation_cost + 10000 * (1 - feasibility_rate)

so a model with low distance but poor feasibility is never preferred.
"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.fleet_search import FleetCandidate, fleet_search
from src.training.rollout import run_episode, run_multi_sample

INFEASIBILITY_PENALTY = 10000.0


def _neural_runner(policy, weights, mode, device, n_samples, charging_portions, use_horizon, max_steps_factor):
    def runner(instance, m):
        if mode == "multi_sample":
            res = run_multi_sample(
                instance, policy, m, weights, n_samples=n_samples, device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                max_steps_factor=max_steps_factor,
            )
        else:
            res = run_episode(
                instance, policy, m, weights, mode="greedy", device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                max_steps_factor=max_steps_factor,
            )
        return FleetCandidate(m=m, feasible=res.feasible, objective=res.objective, summary=res.summary, result=res)

    return runner


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
    max_steps_factor: int = 6,
    extra_margin: int = 5,
    selection_mode: str = "hierarchical",
    max_instances: int | None = None,
    exclude_names: set[str] | None = None,
) -> dict:
    """Validate the policy on a list of eval-dataset indices (one class)."""
    if exclude_names:
        indices = [i for i in indices if eval_ds.names[i] not in exclude_names]
    if max_instances is not None:
        indices = indices[:max_instances]
    if not indices:
        return {}

    runner = _neural_runner(
        policy, weights, mode, device, n_samples, charging_portions, use_horizon, max_steps_factor
    )
    costs, feas, dist, evs, charge, price, wait, unserved, viol = [], [], [], [], [], [], [], [], []
    policy.eval()
    with torch.no_grad():
        for idx in indices:
            instance = eval_ds[idx]
            best, _ = fleet_search(instance, runner, extra_margin, selection_mode)
            s = best.summary
            costs.append(best.objective)
            feas.append(1.0 if best.feasible else 0.0)
            dist.append(s.total_distance)
            evs.append(s.used_evs)
            charge.append(s.total_charging_time)
            price.append(s.total_charging_price)
            wait.append(s.total_waiting_time)
            unserved.append(s.unserved_customers)
            viol.append(
                s.total_battery_violation + s.total_capacity_violation
                + s.total_lateness + s.unserved_customers
            )

    feasibility_rate = float(np.mean(feas))
    mean_cost = float(np.mean(costs))
    return {
        "num_instances": len(indices),
        "cost": mean_cost,
        "feasibility_rate": feasibility_rate,
        "score": mean_cost + INFEASIBILITY_PENALTY * (1.0 - feasibility_rate),
        "avg_distance": float(np.mean(dist)),
        "avg_used_evs": float(np.mean(evs)),
        "avg_charging_time": float(np.mean(charge)),
        "avg_charging_price": float(np.mean(price)),
        "avg_waiting_time": float(np.mean(wait)),
        "avg_unserved": float(np.mean(unserved)),
        "avg_violation": float(np.mean(viol)),
    }


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
    """Validate per class and compute an overall score across the given classes.

    Returns ``{"C": {...}, "R": {...}, "RC": {...}, "overall": {...}}`` where
    only classes with instances are present. ``overall`` averages the per-class
    metrics over ``classes`` (defaults to every class that has data).
    """
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
            "score": float(np.mean([per_class[c]["score"] for c in use])),
            "avg_used_evs": float(np.mean([per_class[c]["avg_used_evs"] for c in use])),
            "avg_distance": float(np.mean([per_class[c]["avg_distance"] for c in use])),
            "classes": use,
        }
    else:
        overall = {}
    return {**per_class, "overall": overall}
