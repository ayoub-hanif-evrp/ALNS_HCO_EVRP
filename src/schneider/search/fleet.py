"""Descending feasible fleet search for TG-FPS."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.schneider.env.cost import CostWeights
from src.schneider.evaluation.fleet_search import lexicographic_fleet_range
from src.schneider.search.repair import _clone_routes, repair_solution
from src.schneider.search.route_replay import replay_solution
from src.schneider.search.score import SearchScore


@dataclass
class FleetAttempt:
    m: int
    variant_count: int
    feasible_count: int
    best_score: SearchScore | None
    runtime: float
    routes: list[tuple[list[int], list[float | None]]] | None = None


@dataclass
class FleetSearchResult:
    m_star: int
    best_routes: list[tuple[list[int], list[float | None]]]
    best_score: SearchScore
    attempts: list[FleetAttempt] = field(default_factory=list)


Constructor = Callable[[Any, int], list[tuple[list[int], list[float | None]]]]


def descending_fleet_search(
    instance,
    construct: Constructor,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    diversified_retries: int = 3,
    m_hi: int | None = None,
    m_lo: int | None = None,
) -> FleetSearchResult:
    """Find smallest reliably feasible fleet via descending search + repair."""
    cfg = config or {}
    fleet_cfg = cfg.get("fleet_search") or {}
    lo, hi = lexicographic_fleet_range(instance, fleet_cfg)
    if m_lo is not None:
        lo = int(m_lo)
    if m_hi is not None:
        hi = int(m_hi)
    lo, hi = min(lo, hi), max(lo, hi)
    retries = int(
        ((cfg.get("hybrid_search") or {}).get("fleet_search") or {}).get(
            "diversified_retries", diversified_retries
        )
    )

    attempts: list[FleetAttempt] = []
    best_feasible: FleetAttempt | None = None

    # Phase 1: obtain feasible at hi.
    m = hi
    while m >= lo:
        t0 = time.perf_counter()
        feasible_count = 0
        best_local: SearchScore | None = None
        best_routes = None
        variant_count = 0
        for trial in range(max(1, retries)):
            variant_count += 1
            routes = construct(instance, m)
            routes, res = repair_solution(
                instance, routes, config=cfg, weights=weights, m=m, max_rounds=8, mode="regret"
            )
            if res.score is None:
                continue
            if res.feasible or (
                res.hard_violation_count == 0 and res.unserved_customers == 0
            ):
                feasible_count += 1
            if best_local is None or res.score.dominates(best_local):
                best_local = res.score
                best_routes = routes
            # Warm-start hint: if we already have a feasible at m+1, constructor may use it;
            # retries still re-construct for diversity.
            _ = trial
        att = FleetAttempt(
            m=m,
            variant_count=variant_count,
            feasible_count=feasible_count,
            best_score=best_local,
            runtime=time.perf_counter() - t0,
            routes=_clone_routes(best_routes) if best_routes is not None else None,
        )
        attempts.append(att)
        if feasible_count > 0 and best_local is not None and best_routes is not None:
            best_feasible = att
            m -= 1
            continue
        # Failed at this m: one more diversified retry block already done; stop descending.
        break

    if best_feasible is None or best_feasible.routes is None or best_feasible.best_score is None:
        # Fall back to hi attempt even if infeasible.
        fallback = attempts[0] if attempts else FleetAttempt(
            m=hi, variant_count=0, feasible_count=0, best_score=SearchScore.worst(), runtime=0.0,
            routes=[([instance.depot_index, instance.depot_index], [None, None])],
        )
        return FleetSearchResult(
            m_star=int(fallback.m),
            best_routes=fallback.routes or [],
            best_score=fallback.best_score or SearchScore.worst(),
            attempts=attempts,
        )
    return FleetSearchResult(
        m_star=int(best_feasible.m),
        best_routes=best_feasible.routes,
        best_score=best_feasible.best_score,
        attempts=attempts,
    )
