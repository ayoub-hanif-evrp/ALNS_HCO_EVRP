"""Schneider EVRPTW: improve a known incumbent; never return a worse plan."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.schneider.env.cost import CostWeights
from src.schneider.search.deadline import SearchDeadline
from src.schneider.search.fast_construct import dense_construct
from src.schneider.search.fleet_search import capacity_lower_bound, compute_fleet_bounds
from src.schneider.search.hanif_fleet_intensify import intensify_hanif_fleet
from src.schneider.search.repair import repair_solution
from src.schneider.search.route_replay import replay_solution
from src.schneider.search.schneider_alns import _replan_charges, _used, run_schneider_alns
from src.schneider.search.schneider_boost import is_long_horizon, squeeze_fleet
from src.schneider.search.score import SearchScore


@dataclass
class SchneiderSolveResult:
    routes: list
    score: SearchScore
    m_star: int
    feasible: bool
    runtime_seconds: float
    stages: dict[str, Any] = field(default_factory=dict)
    exact_check: dict[str, Any] = field(default_factory=dict)
    charging_visits: int = 0
    lower_bound_m: int = 0
    upper_bound_m: int = 0
    fleet_minimization_performed: bool = True
    fleet_trace: list = field(default_factory=list)
    unserved_customers: list = field(default_factory=list)
    termination_reason: str = "completed"
    failure_attribution: list = field(default_factory=list)
    stage_runtimes: dict = field(default_factory=dict)
    budget_adherence: dict = field(default_factory=dict)
    bound_annotations: dict = field(default_factory=dict)


def _compact(instance, routes: list) -> list:
    kept = [
        (list(nodes), list(labels) if labels else [None] * len(nodes))
        for nodes, labels in routes
        if any(instance.nodes[n].is_customer for n in nodes)
    ]
    if not kept:
        d = instance.depot_index
        return [([d, d], [None, None])]
    return kept


def _better(a: SearchScore, b: SearchScore) -> bool:
    return a.as_tuple() < b.as_tuple()


def _validate(instance, routes, *, config, weights) -> tuple[list, SearchScore, bool, int]:
    routes = _compact(instance, routes)
    m = _used(instance, routes) or 1
    res = replay_solution(instance, routes, config=config, weights=weights, m=m)
    score = res.score or SearchScore.worst()
    feas = bool(res.feasible) and int(score.unserved_customer_count) == 0 and int(score.hard_violation_count) == 0
    return routes, score, feas, m


def solve_schneider(
    instance,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    budget_seconds: float = 240.0,
    seed: int = 0,
    warm_start_routes: list | None = None,
) -> SchneiderSolveResult:
    cfg = config or {}
    w = weights or CostWeights.from_eval_config(cfg)
    t0 = time.perf_counter()
    deadline = SearchDeadline.from_seconds(budget_seconds, stage="schneider")
    lo, hi0 = compute_fleet_bounds(instance, cfg)
    short_budget = float(budget_seconds) < 45.0
    stages: dict[str, Any] = {
        "type2": is_long_horizon(instance), "lo": lo, "hi": hi0, "short_budget": short_budget,
    }

    best_routes = None
    best_score = SearchScore.worst()
    best_feas = False
    m_now = hi0

    if warm_start_routes:
        r, sc, ok, m = _validate(instance, warm_start_routes, config=cfg, weights=w)
        stages["warm"] = {"feasible": ok, "k": m, "distance": float(sc.total_distance)}
        if ok:
            best_routes, best_score, best_feas, m_now = r, sc, True, m

    if not best_feas:
        from src.schneider.search.deterministic_fps import solve_deterministic_fps

        fps_budget = min(0.45 * float(budget_seconds), max(20.0, deadline.remaining() * 0.4))
        fps = solve_deterministic_fps(instance, config=cfg, weights=w, budget_seconds=fps_budget)
        r, sc, ok, m = _validate(instance, fps.routes, config=cfg, weights=w)
        stages["fps"] = {"feasible": ok, "k": m, "budget": fps_budget}
        if ok and _better(sc, best_score):
            best_routes, best_score, best_feas, m_now = r, sc, True, m

    if not best_feas:
        k0 = 8 if is_long_horizon(instance) else 20
        r, sc, ok = dense_construct(
            instance, k0, seed=seed, config=cfg, weights=w,
            deadline=deadline.child(min(40.0, deadline.remaining() * 0.3), stage="construct"),
            variant="edf",
        )
        if not ok:
            r, rep = repair_solution(
                instance, r, config=cfg, weights=w, m=k0,
                max_rounds=12, regret_k=3, station_candidates=2, deadline=deadline,
            )
            sc = rep.score or sc
            ok = bool(rep.feasible)
        r, sc, ok, m = _validate(instance, r, config=cfg, weights=w)
        if ok:
            best_routes, best_score, best_feas, m_now = r, sc, True, m

    # Recombine every archived column before local search (cheap; can cut K).
    if not short_budget and deadline.remaining() > 10:
        try:
            from src.schneider.evaluation.schneider_incumbent import iter_archive_solutions
            from src.schneider.search.route_pool import RoutePool
            from src.schneider.search.set_partitioning import solve_set_partitioning

            name = str(getattr(instance, "instance_name", "") or "")
            if name:
                pool = RoutePool()
                if best_routes:
                    pool.add_from_solution(
                        instance, best_routes, source_method="incumbent",
                        source_fleet_size=m_now, config=cfg,
                    )
                for _path, archived, payload in iter_archive_solutions(name, "pr"):
                    pool.add_from_solution(
                        instance, archived,
                        source_method="archive",
                        source_fleet_size=int(payload.get("used_k") or m_now),
                        config=cfg,
                    )
                if len(pool) >= 2:
                    rebuilt, pscore, pmeta = solve_set_partitioning(
                        instance, pool, config=cfg, weights=w, m=max(int(lo), m_now - 1),
                    )
                    stages["archive_sp"] = {**pmeta, "pool": len(pool)}
                    if pscore is not None and rebuilt:
                        r2, s2, ok, m2 = _validate(instance, rebuilt, config=cfg, weights=w)
                        if ok and _better(s2, best_score):
                            best_routes, best_score, best_feas, m_now = r2, s2, True, m2
        except Exception as exc:
            stages["archive_sp"] = {"error": str(exc)}

    if not best_feas or best_routes is None:
        return SchneiderSolveResult(
            routes=best_routes or [],
            score=best_score,
            m_star=int(m_now),
            feasible=False,
            runtime_seconds=time.perf_counter() - t0,
            stages=stages,
            lower_bound_m=int(lo),
            upper_bound_m=int(hi0),
        )

    # Floor: never return worse than this incumbent.
    floor_routes, floor_score, floor_m = best_routes, best_score, m_now

    boost = cfg.get("schneider_boost") or {}
    allow_merge = bool(boost.get("route_merge", True)) and (
        is_long_horizon(instance) or bool(boost.get("merge_type1", True))
    )

    # Fleet shrink: elimination, then absorb/merge (type-1 and type-2).
    if not short_budget and deadline.remaining() > 12 and m_now > lo:
        r2, s2, ilog = intensify_hanif_fleet(
            instance, best_routes, config=cfg, weights=w, m=m_now,
            deadline=deadline.child(min(90.0, deadline.remaining() * 0.40), stage="intensify"),
            max_rounds=40, top_k=12, seeds=[0, 1, 2, 3, 5], lower_bound=lo,
        )
        r2, s2, ok, m2 = _validate(instance, r2, config=cfg, weights=w)
        stages["intensify"] = {**ilog.as_dict(), "accepted": bool(ok and _better(s2, best_score))}
        if ok and _better(s2, best_score):
            best_routes, best_score, m_now = r2, s2, m2

    if not short_budget and deadline.remaining() > 8 and m_now > lo:
        squeezed, sq_score, sq_log = squeeze_fleet(
            instance, best_routes, config=cfg, weights=w,
            deadline=deadline.child(min(70.0, deadline.remaining() * 0.35), stage="squeeze"),
            lower_bound=lo, allow_merge=allow_merge,
        )
        stages["squeeze"] = sq_log
        if sq_score is not None:
            r2, s2, ok, m2 = _validate(instance, squeezed, config=cfg, weights=w)
            if ok and _better(s2, best_score):
                best_routes, best_score, m_now = r2, s2, m2

    alns_rounds = 0
    while not short_budget and best_feas and deadline.remaining() > 16 and alns_rounds < 5:
        slice_s = min(50.0, max(16.0, deadline.remaining() * 0.35))
        alns = run_schneider_alns(
            instance, best_routes, config=cfg, weights=w, m=m_now,
            time_limit_seconds=slice_s, seed=seed + alns_rounds, destroy_k=10,
            prefer_fleet_cut=m_now > lo,
        )
        alns_rounds += 1
        r2, s2, ok, m2 = _validate(instance, alns.routes, config=cfg, weights=w)
        if ok and _better(s2, best_score):
            best_routes, best_score, m_now = r2, s2, m2
        if not short_budget and m_now > lo and deadline.remaining() > 8:
            squeezed, sq_score, sq_log = squeeze_fleet(
                instance, best_routes, config=cfg, weights=w,
                deadline=deadline.child(min(24.0, deadline.remaining() * 0.2), stage="squeeze"),
                lower_bound=lo, allow_merge=allow_merge,
            )
            stages.setdefault("squeeze_after_alns", []).append(sq_log)
            if sq_score is not None:
                r2, s2, ok, m2 = _validate(instance, squeezed, config=cfg, weights=w)
                if ok and _better(s2, best_score):
                    best_routes, best_score, m_now = r2, s2, m2
        if deadline.remaining() < 18:
            break

    # Explicit K-1: squeeze the incumbent first; only then rebuild.
    descents = 0
    while not short_budget and m_now > lo and deadline.remaining() > 18 and descents < 5:
        target = m_now - 1
        descents += 1
        squeezed, sq_score, _ = squeeze_fleet(
            instance, best_routes, config=cfg, weights=w,
            deadline=deadline.child(min(28.0, deadline.remaining() * 0.25), stage="kminus_squeeze"),
            lower_bound=lo, allow_merge=allow_merge,
        )
        if sq_score is not None:
            r2, s2, ok, m2 = _validate(instance, squeezed, config=cfg, weights=w)
            if ok and m2 <= target and _better(s2, best_score):
                best_routes, best_score, m_now = r2, s2, m2
                continue
        trial, tscore, tok = dense_construct(
            instance, target, seed=seed + 19 * descents, config=cfg, weights=w,
            deadline=deadline.child(min(28.0, deadline.remaining() * 0.25), stage="kminus"),
            variant="edf",
        )
        if tok:
            pack = run_schneider_alns(
                instance, _compact(instance, trial), config=cfg, weights=w, m=target,
                time_limit_seconds=min(36.0, deadline.remaining() - 5),
                seed=seed + 31 * descents, prefer_fleet_cut=True,
            )
            r2, s2, ok, m2 = _validate(instance, pack.routes, config=cfg, weights=w)
            if ok and m2 <= target and _better(s2, best_score):
                best_routes, best_score, m_now = r2, s2, m2
                continue
        break

    if floor_score.as_tuple() < best_score.as_tuple():
        best_routes, best_score, m_now = floor_routes, floor_score, floor_m

    best_routes = _replan_charges(instance, _compact(instance, best_routes), config=cfg, deadline=deadline)
    r2, s2, ok, m2 = _validate(instance, best_routes, config=cfg, weights=w)
    if ok and _better(s2, best_score):
        best_routes, best_score, m_now = r2, s2, m2
    elif not ok:
        best_routes, best_score, m_now = floor_routes, floor_score, floor_m
        ok = True

    stages["alns_rounds"] = alns_rounds
    stages["descents"] = descents
    stages["final_k"] = m_now

    from src.schneider.search.exact_checks import exact_accept_solution
    try:
        check, final, missing = exact_accept_solution(
            instance, best_routes, config=cfg, weights=w, m=m_now,
            claimed_score=best_score, fail_loud=False,
        )
        if final.score is not None and final.feasible:
            if _better(final.score, best_score) or final.score.as_tuple() == best_score.as_tuple():
                best_score = final.score
                m_now = int(check.used_vehicles or m_now)
        exact_d = check.as_dict()
        charge_n = int(check.charging_visits)
    except Exception as exc:
        exact_d = {"accepted": False, "error": str(exc)}
        charge_n = 0
        missing = []

    return SchneiderSolveResult(
        routes=best_routes,
        score=best_score,
        m_star=int(m_now),
        feasible=True,
        runtime_seconds=time.perf_counter() - t0,
        stages=stages,
        exact_check=exact_d,
        charging_visits=charge_n,
        unserved_customers=list(missing),
        lower_bound_m=int(lo),
        upper_bound_m=int(hi0),
        bound_annotations={"type2": is_long_horizon(instance), "capacity_lb": capacity_lower_bound(instance)},
    )
