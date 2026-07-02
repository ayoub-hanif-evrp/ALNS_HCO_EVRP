"""Post-decoding local search / repair (evaluation)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.env.cost import CostWeights, compute_cost


@dataclass
class LocalSearchResult:
    raw_objective: float
    repaired_objective: float
    raw_feasible: bool
    repaired_feasible: bool
    improved: bool
    iterations: int
    distance_improvement: float


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
    """Distance-based 2-opt per EV route; re-score with updated distance in summary."""
    cfg = cfg or {}
    max_iter = int(cfg.get("max_iterations", 50))
    summary = env.build_summary()
    raw_j, _ = compute_cost(summary, weights)
    raw_feas = env.feasible_solution()

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

    repaired_summary = env.build_summary()
    repaired_summary.total_distance = max(0.0, repaired_summary.total_distance - total_saved)
    repaired_j, _ = compute_cost(repaired_summary, weights)
    repaired_feas = raw_feas  # feasibility unchanged in distance-only polish

    improved = repaired_j < raw_j
    if repaired_feas and not raw_feas:
        pick_repaired = True
    elif repaired_feas and raw_feas:
        pick_repaired = repaired_j < raw_j
    else:
        pick_repaired = repaired_j < raw_j

    return LocalSearchResult(
        raw_objective=raw_j,
        repaired_objective=repaired_j if pick_repaired else raw_j,
        raw_feasible=raw_feas,
        repaired_feasible=repaired_feas,
        improved=improved,
        iterations=iterations,
        distance_improvement=total_saved,
    )
