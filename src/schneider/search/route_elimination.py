"""Genuine route elimination for fleet reduction m → m-1."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.schneider.search.charging_dp import optimize_charging_plan
from src.schneider.search.deadline import SearchDeadline
from src.schneider.search.repair import _clone_routes, regret_insertion, repair_solution
from src.schneider.search.route_replay import replay_solution
from src.schneider.search.score import SearchScore


@dataclass
class RouteEliminationAttempt:
    eliminated_route_index: int
    n_customers: int
    feasible: bool
    score: SearchScore | None
    runtime: float
    reason: str | None = None


def _route_centroid(instance, nodes: list[int]) -> tuple[float, float]:
    custs = [n for n in nodes if instance.nodes[n].is_customer]
    if not custs:
        d = instance.depot_index
        return float(instance.nodes[d].x), float(instance.nodes[d].y)
    xs = [float(instance.nodes[c].x) for c in custs]
    ys = [float(instance.nodes[c].y) for c in custs]
    return sum(xs) / len(xs), sum(ys) / len(ys)


def rank_elimination_candidates(instance, routes: list, *, top_k: int = 8) -> list[int]:
    """Rank nonempty routes for elimination (not only the smallest)."""
    scored: list[tuple[tuple, int]] = []
    centroids = []
    for i, (nodes, _) in enumerate(routes):
        n_c = sum(1 for n in nodes if instance.nodes[n].is_customer)
        if n_c == 0:
            continue
        cx, cy = _route_centroid(instance, nodes)
        centroids.append((i, cx, cy, n_c))

    for i, cx, cy, n_c in centroids:
        # Geographic overlap proxy: mean distance to other route centroids
        others = [(ox, oy) for j, ox, oy, _ in centroids if j != i]
        if others:
            overlap = sum(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 for ox, oy in others) / len(others)
        else:
            overlap = 0.0
        # Prefer eliminating small, overlapping, temporally loose routes
        dues = [
            float(instance.nodes[n].due_time)
            for n, _ in [ (nn, None) for nn in routes[i][0] if instance.nodes[nn].is_customer ]
        ]
        # Fix dues collection
        dues = [
            float(instance.nodes[n].due_time)
            for n in routes[i][0]
            if instance.nodes[n].is_customer
        ]
        tw_slack = (sum(dues) / len(dues)) if dues else 0.0
        key = (n_c, overlap, -tw_slack)
        scored.append((key, i))
    scored.sort()
    return [i for _, i in scored[: max(1, top_k)]]


def _optimize_affected_routes(instance, routes, *, config, m) -> list:
    out = []
    for nodes, labels in routes:
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        if not custs:
            out.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
            continue
        plan = optimize_charging_plan(instance, custs, config=config, m=m)
        if plan.feasible and plan.nodes:
            out.append((list(plan.nodes), list(plan.labels)))
        else:
            out.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
    return out


def eliminate_one_route(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights=None,
    m: int | None = None,
    deadline: SearchDeadline | None = None,
    top_k: int = 6,
    focused_alns: bool = True,
) -> tuple[list, SearchScore | None, list[RouteEliminationAttempt]]:
    """Try eliminating several candidate routes; keep the strongest feasible m-1 solution."""
    cfg = config or {}
    target_m = (m - 1) if m is not None else max(1, len(routes) - 1)
    attempts: list[RouteEliminationAttempt] = []
    best_routes = None
    best_score: SearchScore | None = None

    candidates = rank_elimination_candidates(instance, routes, top_k=top_k)
    for ri in candidates:
        if deadline is not None and deadline.expired():
            break
        t0 = time.perf_counter()
        base = _clone_routes(routes)
        removed = base.pop(ri)
        ejected = [n for n in removed[0] if instance.nodes[n].is_customer]
        # Ensure fleet size slots
        while len(base) < target_m:
            depot = instance.depot_index
            base.append(([depot, depot], [None, None]))
        base = base[:target_m]

        # Regret reinsertion
        trial = _clone_routes(base)
        unserved = list(ejected)
        for _ in range(max(1, len(unserved) * 3)):
            if deadline is not None and deadline.expired():
                break
            if not unserved:
                break
            cand = regret_insertion(
                instance, trial, unserved, regret_k=3, config=cfg, weights=weights, m=target_m,
                deadline=deadline, station_candidates=2,
            )
            if cand is None:
                cand = regret_insertion(
                    instance, trial, unserved, regret_k=2, config=cfg, weights=weights, m=target_m,
                    deadline=deadline, station_candidates=2,
                )
            if cand is None:
                break
            trial = cand.routes
            unserved = [c for c in unserved if c != cand.customer]

        trial, rep = repair_solution(
            instance, trial, config=cfg, weights=weights, m=target_m,
            max_rounds=8, regret_k=3, mode="regret", deadline=deadline,
        )
        trial = _optimize_affected_routes(instance, trial, config=cfg, m=target_m)

        if focused_alns and (deadline is None or deadline.remaining() > 2.0):
            try:
                from src.schneider.search.alns import run_alns
                alns_budget = 3.0 if deadline is None else min(5.0, deadline.remaining())
                alns = run_alns(
                    instance, trial, config=cfg, weights=weights, m=target_m,
                    iterations=20, time_limit_seconds=alns_budget, guide=None,
                )
                trial, score = alns.routes, alns.score
            except Exception:
                score = rep.score
        else:
            score = rep.score

        final = replay_solution(instance, trial, config=cfg, weights=weights, m=target_m)
        feas = bool(final.feasible) and int(final.unserved_customers) == 0
        sc = final.score or score
        attempts.append(
            RouteEliminationAttempt(
                eliminated_route_index=int(ri),
                n_customers=len(ejected),
                feasible=feas,
                score=sc,
                runtime=time.perf_counter() - t0,
                reason=None if feas else (final.first_failure or "elimination_failed"),
            )
        )
        if feas and sc is not None and (best_score is None or sc.dominates(best_score)):
            best_routes, best_score = trial, sc

    if best_routes is None:
        return _clone_routes(routes), None, attempts
    return best_routes, best_score, attempts
