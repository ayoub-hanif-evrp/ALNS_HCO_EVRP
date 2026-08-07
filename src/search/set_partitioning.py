"""Set-partitioning recombination over an exact route pool."""
from __future__ import annotations

from typing import Any

from src.search.route_pool import PooledRoute, RoutePool
from src.search.route_replay import replay_solution
from src.search.score import SearchScore


class IncompleteCoverageError(ValueError):
    """Raised when the pool cannot cover every customer."""


def _lex_route_key(r: PooledRoute) -> tuple:
    return (1, float(r.distance), float(r.charging_time), float(r.duration))


def _greedy_set_partition(
    customers: set[int],
    pool: list[PooledRoute],
    *,
    max_routes: int | None,
) -> list[PooledRoute] | None:
    uncovered = set(customers)
    chosen: list[PooledRoute] = []
    remaining = list(pool)
    while uncovered and remaining:
        if max_routes is not None and len(chosen) >= max_routes:
            return None
        best = None
        best_key = None
        for r in remaining:
            cover = set(r.customers) & uncovered
            if not cover:
                continue
            # Prefer max cover, then exact route lex cost
            key = (-len(cover), float(r.distance), float(r.charging_time), float(r.duration))
            if best_key is None or key < best_key:
                best_key, best = key, r
        if best is None:
            return None
        chosen.append(best)
        uncovered -= set(best.customers)
        remaining = [r for r in remaining if set(r.customers).isdisjoint(best.customers)]
    if uncovered:
        return None
    return chosen


def _ortools_partition(
    customers: set[int],
    routes: list[PooledRoute],
    *,
    max_routes: int | None,
    time_limit: float = 5.0,
) -> tuple[list[PooledRoute] | None, dict]:
    meta: dict[str, Any] = {}
    try:
        from ortools.sat.python import cp_model
    except Exception as exc:
        meta["ortools_error"] = str(exc)
        return None, meta

    if not (1 < len(routes) <= 500):
        return None, meta

    model = cp_model.CpModel()
    x = [model.NewBoolVar(f"r{i}") for i in range(len(routes))]

    # Every customer must have ≥1 covering route; enforce equality only when cover exists.
    for c in customers:
        covering = [x[i] for i, r in enumerate(routes) if c in r.customers]
        if not covering:
            meta["missing_customer"] = int(c)
            return None, meta
        model.Add(sum(covering) == 1)

    if max_routes is not None:
        model.Add(sum(x) <= int(max_routes))

    # True lex via hierarchical CP-SAT: primary routes, then distance, then charging.
    n_routes_var = model.NewIntVar(0, len(routes), "n_routes")
    model.Add(n_routes_var == sum(x))
    # Scale distances to ints
    dist_i = [int(round(r.distance * 1000)) for r in routes]
    charge_i = [int(round(r.charging_time * 1000)) for r in routes]
    max_dist = sum(dist_i) + 1
    max_charge = sum(charge_i) + 1
    total_dist = model.NewIntVar(0, max_dist, "total_dist")
    total_charge = model.NewIntVar(0, max_charge, "total_charge")
    model.Add(total_dist == sum(x[i] * dist_i[i] for i in range(len(routes))))
    model.Add(total_charge == sum(x[i] * charge_i[i] for i in range(len(routes))))
    # Encode lex (n_routes, dist, charge) as one objective with proven separators
    # when max_dist and max_charge bound secondary terms.
    model.Minimize(
        n_routes_var * (max_dist * max_charge)
        + total_dist * max_charge
        + total_charge
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit)
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        meta["ortools_status"] = int(status)
        return None, meta
    chosen = [routes[i] for i, var in enumerate(x) if solver.Value(var) == 1]
    meta["backend"] = "ortools_cpsat_lex"
    return chosen, meta


def solve_set_partitioning(
    instance,
    pool: RoutePool,
    *,
    config: dict | None = None,
    weights=None,
    m: int | None = None,
) -> tuple[list, SearchScore | None, dict]:
    """Cover each customer exactly once; lex min (routes, distance, charging_time).

    If any customer has no covering route in the pool, return no solution
    (do not silently drop the equality constraint).
    """
    customers = {i for i, n in enumerate(instance.nodes) if n.is_customer}
    routes = [r for r in pool.as_list() if r.feasible]
    meta: dict[str, Any] = {"backend": "none", "pool_size": len(routes)}

    uncovered = pool.uncovered_customers(customers) if hasattr(pool, "uncovered_customers") else set()
    # Recompute from filtered routes
    covered: set[int] = set()
    for r in routes:
        covered.update(r.customers)
    missing = customers - covered
    if missing:
        meta["incomplete_coverage"] = True
        meta["missing_customers"] = sorted(int(c) for c in missing)
        meta["rejected_reason"] = "missing_customer_coverage"
        return [], None, meta

    max_routes = int(m) if m is not None else None
    chosen: list[PooledRoute] | None = None

    ort_chosen, ort_meta = _ortools_partition(customers, routes, max_routes=max_routes)
    meta.update(ort_meta)
    if ort_chosen is not None:
        chosen = ort_chosen
        meta["backend"] = ort_meta.get("backend", "ortools_cpsat_lex")
    else:
        chosen = _greedy_set_partition(customers, routes, max_routes=max_routes)
        meta["backend"] = "greedy_lex"

    if not chosen:
        meta["rejected_reason"] = "no_complete_partition"
        return [], None, meta

    # Reuse exact pooled nodes/labels — do not reconstruct from customers alone.
    rebuilt = [(list(r.nodes), list(r.charge_labels)) for r in chosen]
    final = replay_solution(instance, rebuilt, config=config, weights=weights, m=m or len(rebuilt))
    served = set(final.served_customers)
    if served != customers or final.duplicate_customers > 0 or not final.feasible:
        meta["rejected_reason"] = "replay_reject"
        meta["feasible"] = False
        meta["missing_after_replay"] = sorted(customers - served)
        return [], None, meta

    meta["feasible"] = True
    meta["n_routes"] = len(rebuilt)
    meta["total_distance"] = sum(r.distance for r in chosen)
    meta["total_charging_time"] = sum(r.charging_time for r in chosen)
    return rebuilt, final.score, meta
