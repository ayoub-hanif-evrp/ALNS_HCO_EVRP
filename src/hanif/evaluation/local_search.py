"""Post-decoding local search (diagnostic-only; does not change reported objective).

Distance-only 2-opt over customer sequences is **not** a valid EVRPTW improvement
because reordering customers changes time windows, battery, charging, and feasibility.
This module may run for diagnostics but must never alter official evaluation scores.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.hanif.env.cost import CostWeights, compute_cost


@dataclass
class LocalSearchResult:
    raw_objective: float
    repaired_objective: float
    raw_feasible: bool
    repaired_feasible: bool
    improved: bool
    iterations: int
    distance_improvement: float
    diagnostic_only: bool = True
    enabled: bool = False
    rejection_reason: str = "local_search_disabled_for_evrptw"


def _customer_subroute(env, ev) -> list[int]:
    return [n for n in ev.route_nodes if env.instance.nodes[n].is_customer]


def _two_opt(instance, route: list[int], max_iterations: int) -> tuple[list[int], float]:
    """2-opt on customer indices using distance matrix only."""
    if len(route) < 3:
        return route, 0.0
    best = list(route)
    dist = instance.distance_matrix
    best_len = _route_length(dist, best)
    improved = True
    it = 0
    while improved and it < max_iterations:
        improved = False
        it += 1
        for i in range(len(best) - 1):
            for k in range(i + 2, len(best)):
                new = best[: i + 1] + best[i + 1 : k + 1][::-1] + best[k + 1 :]
                new_len = _route_length(dist, new)
                if new_len + 1e-9 < best_len:
                    best = new
                    best_len = new_len
                    improved = True
    return best, best_len


def _route_length(dist, route: list[int]) -> float:
    if len(route) < 2:
        return 0.0
    return float(sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1)))


def local_search_repair(env, weights: CostWeights, cfg: dict | None = None) -> LocalSearchResult:
    """Run distance-only 2-opt diagnostics; never mutate the reported EVRPTW objective."""
    cfg = cfg or {}
    enabled = bool(cfg.get("enabled", False))
    summary = env.build_summary()
    raw_j, _ = compute_cost(summary, weights)
    raw_feas = env.feasible_solution()

    if not enabled:
        return LocalSearchResult(
            raw_objective=raw_j,
            repaired_objective=raw_j,
            raw_feasible=raw_feas,
            repaired_feasible=raw_feas,
            improved=False,
            iterations=0,
            distance_improvement=0.0,
            diagnostic_only=True,
            enabled=False,
            rejection_reason="local_search_disabled_for_evrptw",
        )

    max_iter = int(cfg.get("max_iterations", 50))
    total_saved = 0.0
    iterations = 0
    for ev in env.evs:
        customers = _customer_subroute(env, ev)
        if len(customers) < 3:
            continue
        new_order, new_len = _two_opt(env.instance, customers, max_iter)
        old_len = _route_length(env.instance.distance_matrix, customers)
        if new_len + 1e-9 < old_len:
            total_saved += old_len - new_len
            iterations += 1

    return LocalSearchResult(
        raw_objective=raw_j,
        repaired_objective=raw_j,
        raw_feasible=raw_feas,
        repaired_feasible=raw_feas,
        improved=False,
        iterations=iterations,
        distance_improvement=total_saved,
        diagnostic_only=True,
        enabled=True,
        rejection_reason=(
            "distance_only_2opt_not_replayed_through_evrptw"
            if total_saved > 0
            else "no_distance_improvement_found"
        ),
    )
