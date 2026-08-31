"""Previous Hanif CARE-FI (v1), kept only for K comparison."""
from __future__ import annotations

import time
from typing import Any

from src.hanif.current import SOLVER_VERSION
from src.hanif.current.charging_refinement import refine_charging
from src.hanif.current.construction import construct_feasible
from src.hanif.current.fleet_reduction import reduce_fleet
from src.hanif.replay import replay_solution, sequences_match
from src.hanif.solver import CareFiResult


def solve_stage1(
    instance,
    *,
    budget_s: float = 180.0,
    nearby_routes: int = 12,
    insert_positions: int = 5,
    nearby_stations: int = 3,
    max_elim_candidates: int = 16,
    warm_routes: list | None = None,
) -> CareFiResult:
    t0 = time.perf_counter()
    stages: dict[str, Any] = {"solver_version": SOLVER_VERSION}
    if warm_routes:
        routes = [(list(n), list(l)) for n, l in warm_routes]
        stages["construction"] = {"source": "warm_start", "k": replay_solution(instance, routes).used_k}
    else:
        construct_budget = min(0.45 * budget_s, max(20.0, budget_s * 0.4))
        routes = construct_feasible(
            instance,
            nearby_routes=nearby_routes,
            insert_positions=insert_positions,
            nearby_stations=nearby_stations,
            deadline=construct_budget,
        )
        stages["construction"] = {"k": replay_solution(instance, routes).used_k, "budget_s": construct_budget}
    remaining = max(5.0, budget_s - (time.perf_counter() - t0))
    routes, elog = reduce_fleet(
        instance, routes,
        max_candidates=max_elim_candidates,
        nearby_routes=nearby_routes,
        insert_positions=insert_positions,
        nearby_stations=nearby_stations,
        deadline=remaining,
    )
    stages["fleet_reduction"] = elog
    metrics = replay_solution(instance, routes)
    reason = "completed" if metrics.feasible else (metrics.failure or "infeasible")
    return CareFiResult(
        routes=routes, stage="classical", metrics=metrics,
        runtime_s=time.perf_counter() - t0, stages=stages, termination_reason=reason,
    )


def solve_stage2(instance, classical_routes: list, *, max_passes: int = 3) -> CareFiResult:
    t0 = time.perf_counter()
    refined, clog = refine_charging(instance, classical_routes, max_passes=max_passes)
    metrics = replay_solution(instance, refined)
    frozen = sequences_match(instance, classical_routes, refined)
    k_ok = metrics.used_k == replay_solution(instance, classical_routes).used_k
    validation = {
        "hard_violations": metrics.hard_violations,
        "all_served": metrics.unserved == 0 and metrics.duplicates == 0,
        "k_frozen": k_ok,
        "sequences_frozen": frozen,
        "phi_nonworse": bool(clog.get("phi_proposed", 99) <= clog.get("phi_classical", 0) + 1e-9),
        **clog,
    }
    if not (metrics.feasible and frozen and k_ok and validation["phi_nonworse"]):
        refined = [(list(n), list(l)) for n, l in classical_routes]
        metrics = replay_solution(instance, refined)
    return CareFiResult(
        routes=refined, stage="proposed", metrics=metrics,
        runtime_s=time.perf_counter() - t0, stages=clog, validation=validation,
        termination_reason="completed" if metrics.feasible else (metrics.failure or "infeasible"),
    )


def solve_carefi(
    instance,
    *,
    budget_s: float = 180.0,
    warm_routes: list | None = None,
    stage2: bool = True,
) -> tuple[CareFiResult, CareFiResult | None]:
    stage1 = solve_stage1(instance, budget_s=budget_s, warm_routes=warm_routes)
    if not stage2 or not stage1.metrics.feasible:
        return stage1, None
    return stage1, solve_stage2(instance, stage1.routes)
