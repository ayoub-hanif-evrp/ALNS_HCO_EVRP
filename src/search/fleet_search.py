"""Genuine descending fleet minimization with warm elimination, cold probes, and route-pool recombination."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from src.env.cost import CostWeights
from src.evaluation.fleet_search import lexicographic_fleet_range
from src.search.deadline import SearchDeadline
from src.search.repair import _clone_routes, repair_solution
from src.search.route_elimination import eliminate_one_route
from src.search.route_pool import RoutePool
from src.search.route_replay import replay_solution
from src.search.score import BenchmarkScore, SearchScore
from src.search.set_partitioning import solve_set_partitioning


@dataclass
class FleetAttemptTrace:
    fleet_size: int
    attempt: int
    variant: str
    seed: int
    feasible: bool
    served: int
    used_vehicles: int
    distance: float
    runtime: float
    failure_reason: str | None = None
    score: SearchScore | None = None
    warm_start_source: str | None = None
    bound_kind: str | None = None  # mathematical_lb | heuristic_estimate | known_feasible | expanded_upper | independent_probe


@dataclass
class FleetSearchResult:
    m_star: int
    best_routes: list
    best_score: SearchScore
    benchmark: BenchmarkScore
    attempts: list[FleetAttemptTrace] = field(default_factory=list)
    upper_bound_m: int = 0
    lower_bound_m: int = 0
    fleet_minimization_performed: bool = True
    route_pool_size: int = 0
    route_elimination_attempts: int = 0
    set_partitioning_hits: int = 0
    bound_annotations: dict[str, Any] = field(default_factory=dict)


def capacity_lower_bound(instance) -> int:
    veh = instance.vehicle
    cap = float(getattr(veh, "capacity", None) or getattr(veh, "load_capacity", None) or getattr(veh, "max_load", 0) or 0)
    if cap <= 0:
        return int(getattr(instance, "m_min", 1) or 1)
    demand = sum(float(n.demand) for n in instance.nodes if n.is_customer)
    return max(1, int(math.ceil(demand / cap)))


def tw_pressure_lower_bound(instance) -> int:
    n = int(instance.num_customers)
    dues = [float(n.due_time) for n in instance.nodes if n.is_customer]
    if len(dues) < 3:
        return max(1, n // 12)
    import numpy as np

    arr = np.asarray(dues, dtype=float)
    span = float(arr.max() - arr.min()) + 1e-9
    early = float(np.mean(arr <= arr.min() + 0.35 * span))
    if early >= 0.55:
        return max(1, int(math.ceil(n / 6.0)))
    return max(1, int(math.ceil(n / 12.0)))


def compute_fleet_bounds(instance, config: dict | None = None) -> tuple[int, int]:
    cfg = (config or {}).get("fleet_search") or {}
    lo_cap = capacity_lower_bound(instance)
    lo_tw = tw_pressure_lower_bound(instance)
    lo = max(lo_cap, lo_tw, int(getattr(instance, "m_min", 1) or 1))
    _, hi = lexicographic_fleet_range(instance, cfg)
    hi = max(lo, int(hi))
    return lo, hi


def construct_solution(
    instance,
    fleet_size: int,
    expert_variant: str = "hybrid",
    seed: int = 0,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    warm_start: list | None = None,
    deadline: SearchDeadline | None = None,
    use_expert: bool | None = None,
):
    cfg = config or {}
    w = weights or CostWeights()
    if deadline is not None and deadline.expired():
        depot = instance.depot_index
        empty = [([depot, depot], [None, None]) for _ in range(fleet_size)]
        return empty, SearchScore.worst(), False

    if warm_start is not None:
        # Classic drop fewest-customer routes then regret repair.
        routes = _clone_routes(warm_start)
        if len(routes) > fleet_size:
            scored = []
            for i, (nodes, _) in enumerate(routes):
                n_c = sum(1 for n in nodes if instance.nodes[n].is_customer)
                scored.append((n_c, i))
            scored.sort()
            drop = {i for _, i in scored[: len(routes) - fleet_size]}
            routes = [r for i, r in enumerate(routes) if i not in drop]
        while len(routes) < fleet_size:
            depot = instance.depot_index
            routes.append(([depot, depot], [None, None]))
        routes = routes[:fleet_size]
        routes, res = repair_solution(
            instance, routes, config=cfg, weights=w, m=fleet_size, max_rounds=15, mode="regret",
            deadline=deadline, station_candidates=2,
        )
        return routes, res.score or SearchScore.worst(), bool(res.feasible)

    from src.search.fast_construct import matrix_construct
    from src.search.portfolio import run_variant_episode

    prefer_expert = bool((cfg.get("fleet_search") or {}).get("prefer_expert_construct", True))
    if use_expert is None:
        use_expert = prefer_expert

    if use_expert and (deadline is None or deadline.remaining() > 20):
        result = run_variant_episode(
            instance,
            fleet_size,
            variant=expert_variant,  # type: ignore[arg-type]
            seed=seed,
            config=cfg,
            weights=w,
            deadline=deadline,
            max_steps_factor=3,
        )
        return result.routes, result.score, bool(result.feasible)

    return matrix_construct(
        instance, fleet_size, seed=seed, config=cfg, weights=w,
        deadline=deadline, variant=expert_variant,
    )


def _variant_seed_schedule(variants: list[str], seeds: list[int], n: int) -> list[tuple[str, int]]:
    """Cover all variants before repeating any (round-robin)."""
    out: list[tuple[str, int]] = []
    i = 0
    while len(out) < n:
        v = variants[i % len(variants)]
        s = seeds[(i // len(variants)) % len(seeds)] + (i // (len(variants) * max(1, len(seeds))))
        out.append((v, s))
        i += 1
    return out


def solve_with_fleet_search(
    instance,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    budget_seconds: float | None = None,
    deadline: SearchDeadline | None = None,
    construct: Callable | None = None,
    route_pool: RoutePool | None = None,
) -> FleetSearchResult:
    """Official fleet-minimization entrypoint (no fixed m=30/40)."""
    cfg = config or {}
    fleet_cfg = dict(cfg.get("fleet_search") or {})
    hs_fleet = ((cfg.get("hybrid_search") or {}).get("fleet_search") or {})
    w = weights or (CostWeights.from_eval_config(cfg) if cfg else CostWeights())

    upper_retries = int(fleet_cfg.get("upper_bound_retries", hs_fleet.get("diversified_retries", 5)))
    retries_per = int(fleet_cfg.get("retries_per_fleet", 8))
    failures_before_stop = int(fleet_cfg.get("failures_before_stop", 3))
    warm = bool(fleet_cfg.get("warm_start_from_larger_fleet", True))
    cold_after_warm = bool(fleet_cfg.get("cold_retry_after_warm_failure", True))
    stop_upper_on_feasible = bool(fleet_cfg.get("stop_upper_on_first_feasible", True))
    use_elim = bool(fleet_cfg.get("route_elimination", True))
    use_sp = bool(fleet_cfg.get("set_partitioning_during_descent", True))
    variants = list(
        ((cfg.get("hybrid_search") or {}).get("construction") or {}).get("expert_variants_list")
        or ["edf", "regret", "energy", "fleet", "hybrid"]
    )
    seeds = list(fleet_cfg.get("seeds") or [0, 1, 2, 3])

    lo, hi0 = compute_fleet_bounds(instance, cfg)
    parent_deadline = deadline or SearchDeadline.from_seconds(budget_seconds, stage="fleet_search")
    traces: list[FleetAttemptTrace] = []
    pool = route_pool if route_pool is not None else RoutePool()
    elim_count = 0
    sp_hits = 0
    bound_ann = {
        "mathematical_lower_bound": int(capacity_lower_bound(instance)),
        "heuristic_lower_estimate": int(lo),
        "initial_upper_probe": int(hi0),
    }

    attempt = 0
    best_routes = None
    best_score = SearchScore.worst()
    best_feasible = False
    hi = hi0

    def _alive() -> bool:
        return parent_deadline.check()

    def _record(**kwargs):
        nonlocal attempt
        attempt += 1
        traces.append(FleetAttemptTrace(attempt=attempt, **kwargs))

    # --- Upper bound probes ---
    # Always lead with a canonical (edf, seed=0) probe. Validation-seed offsets
    # diversify descent; they must not erase the only reliable upper constructor.
    upper_order: list[tuple[str, int]] = [("edf", 0)]
    for v, s in _variant_seed_schedule(variants, seeds, max(len(variants), upper_retries)):
        if (v, s) not in upper_order:
            upper_order.append((v, s))
    upper_order = upper_order[: max(2, min(upper_retries + 1, 5))]
    for ui, (v, s) in enumerate(upper_order):
        if not _alive():
            break
        t0 = time.perf_counter()
        # Canonical first probe may need several minutes on type-1 R; later
        # diversified probes get shorter slices so a bad seed cannot burn the budget.
        if ui == 0:
            slice_s = min(380.0, max(60.0, parent_deadline.remaining() * 0.55))
        else:
            slice_s = min(150.0, max(40.0, parent_deadline.remaining() * 0.22))
        if construct is not None:
            routes = construct(instance, hi)
            routes, res = repair_solution(instance, routes, config=cfg, weights=w, m=hi)
            score = res.score or SearchScore.worst()
            feas = bool(res.feasible)
        else:
            routes, score, feas = construct_solution(
                instance, hi, expert_variant=v, seed=s, config=cfg, weights=w,
                deadline=parent_deadline.child(slice_s, stage="upper_construct"),
                use_expert=True,
            )
            # If expert with a non-canonical seed collapses, try fast construct once.
            if (not feas) and (v, s) != ("edf", 0) and parent_deadline.remaining() > 25:
                alt_routes, alt_score, alt_feas = construct_solution(
                    instance, hi, expert_variant=v, seed=s, config=cfg, weights=w,
                    deadline=parent_deadline.child(
                        min(90.0, parent_deadline.remaining()), stage="upper_matrix"
                    ),
                    use_expert=False,
                )
                if alt_feas or alt_score.dominates(score):
                    routes, score, feas = alt_routes, alt_score, alt_feas
                    v = f"{v}+matrix"
        n_cust = int(instance.num_customers)
        served = n_cust - int(score.unserved_customer_count)
        _record(
            fleet_size=hi, variant=v, seed=s, feasible=feas, served=served,
            used_vehicles=int(score.used_evs), distance=float(score.total_distance),
            runtime=time.perf_counter() - t0,
            failure_reason=None if feas else "infeasible_or_unserved",
            score=score, warm_start_source="upper_bound_probe", bound_kind="expanded_upper",
        )
        pool.add_from_solution(
            instance, routes, score=score, source_method=f"upper_{v}", source_fleet_size=hi, config=cfg,
        )
        if score.dominates(best_score):
            best_score, best_routes, best_feasible = score, routes, feas
        if feas and stop_upper_on_feasible:
            break

    if best_routes is None:
        depot = instance.depot_index
        best_routes = [([depot, depot], [None, None]) for _ in range(hi)]

    # Expand upper if needed
    expand_step = int(fleet_cfg.get("upper_expand_step", 4))
    expand_max = min(int(instance.num_customers), int(fleet_cfg.get("upper_expand_max", max(hi + 20, 45))))
    expand_hi = hi
    while (not best_feasible) and expand_hi + expand_step <= expand_max and _alive():
        expand_hi = min(expand_max, expand_hi + expand_step)
        v, s = variants[attempt % len(variants)], seeds[attempt % len(seeds)] + attempt
        t0 = time.perf_counter()
        routes, score, feas = construct_solution(
            instance, expand_hi, expert_variant=v, seed=s, config=cfg, weights=w,
            deadline=parent_deadline.child(
                min(350.0, parent_deadline.remaining()), stage="upper_expand"
            ),
            use_expert=True,
        )
        n_cust = int(instance.num_customers)
        served = n_cust - int(score.unserved_customer_count)
        _record(
            fleet_size=expand_hi, variant=v, seed=s, feasible=feas, served=served,
            used_vehicles=int(score.used_evs), distance=float(score.total_distance),
            runtime=time.perf_counter() - t0,
            failure_reason=None if feas else "upper_expand_infeasible",
            score=score, warm_start_source="upper_bound_expand", bound_kind="expanded_upper",
        )
        pool.add_from_solution(
            instance, routes, score=score, source_method="upper_expand", source_fleet_size=expand_hi, config=cfg,
        )
        if score.dominates(best_score):
            best_score, best_routes, best_feasible = score, routes, feas
        if feas:
            hi = expand_hi
            break

    bound_ann["first_feasible_upper_bound"] = int(hi) if best_feasible else None
    bound_ann["known_feasible_incumbent"] = int(hi) if best_feasible else None

    m = hi
    consecutive_failures = 0
    m_star = hi if best_feasible else hi
    star_routes = _clone_routes(best_routes)
    star_score = best_score
    successful_at_mstar = 1 if best_feasible else 0

    def _try_target(target: int, *, mode: str) -> bool:
        nonlocal star_routes, star_score, m_star, best_feasible, successful_at_mstar, elim_count, sp_hits
        found = False
        local_best = None
        local_routes = None
        schedule = _variant_seed_schedule(variants, seeds, max(retries_per, len(variants)))
        # Shrink retries when time is tight
        rem = parent_deadline.remaining()
        if rem < 45:
            schedule = schedule[:2]
        elif rem < 90:
            schedule = schedule[:4]
        elif rem < 150:
            schedule = schedule[:6]
        for trial, (v, s) in enumerate(schedule):
            if not _alive() or parent_deadline.remaining() < 3:
                break
            t0 = time.perf_counter()
            attempt_deadline = parent_deadline.child(
                min(30.0, max(3.0, parent_deadline.remaining() - 5.0)), stage=f"{mode}_attempt",
            )
            warm_routes = star_routes if (warm and best_feasible and mode == "warm") else None
            warm_src = f"warm_from_m={m_star}" if warm_routes is not None else "cold_construct"
            bound_kind = "known_feasible" if warm_routes is not None else (
                "independent_probe" if mode == "independent" else "heuristic_estimate"
            )
            if warm_routes is not None and use_elim and parent_deadline.remaining() > 90:
                routes, score, atts = eliminate_one_route(
                    instance, warm_routes, config=cfg, weights=w, m=m_star,
                    deadline=parent_deadline.child(
                        min(15.0, parent_deadline.remaining() * 0.15), stage="route_elim"
                    ),
                    top_k=2,
                    focused_alns=False,
                )
                elim_count += len(atts)
                feas = score is not None and bool(
                    replay_solution(instance, routes, config=cfg, weights=w, m=target).feasible
                )
                used = sum(1 for nodes, _ in routes if any(instance.nodes[n].is_customer for n in nodes))
                if used > target or not feas:
                    routes, score, feas = construct_solution(
                        instance, target, expert_variant=v, seed=s, config=cfg, weights=w,
                        warm_start=warm_routes, deadline=attempt_deadline,
                    )
                    warm_src = f"warm_drop_repair_from_m={m_star}"
            else:
                routes, score, feas = construct_solution(
                    instance, target, expert_variant=v, seed=s, config=cfg, weights=w,
                    warm_start=warm_routes, deadline=attempt_deadline,
                )

            if (not feas) and cold_after_warm and warm_routes is not None and parent_deadline.remaining() > 20:
                routes2, score2, feas2 = construct_solution(
                    instance, target, expert_variant=v, seed=s + 17, config=cfg, weights=w,
                    warm_start=None, deadline=attempt_deadline, use_expert=False,
                )
                if feas2 or score2.dominates(score):
                    routes, score, feas = routes2, score2, feas2
                    warm_src = "cold_construct_after_warm_fail"
                    bound_kind = "independent_probe"

            # Route-pool add is exact-replay heavy; only archive feasible attempts.
            if feas:
                pool.add_from_solution(
                    instance, routes, score=score, source_method=f"{mode}_{v}",
                    source_fleet_size=target, config=cfg,
                )
            n_cust = int(instance.num_customers)
            served = n_cust - int(score.unserved_customer_count)
            _record(
                fleet_size=target, variant=v, seed=s, feasible=feas, served=served,
                used_vehicles=int(score.used_evs), distance=float(score.total_distance),
                runtime=time.perf_counter() - t0,
                failure_reason=None if feas else f"{mode}_failed",
                score=score, warm_start_source=warm_src, bound_kind=bound_kind,
            )
            if feas and (local_best is None or score.dominates(local_best)):
                local_best, local_routes, found = score, routes, True

        # Route-pool recombination at this target
        if use_sp and _alive() and len(pool) > 0 and parent_deadline.remaining() > 15:
            rebuilt, pool_score, meta = solve_set_partitioning(
                instance, pool, config=cfg, weights=w, m=target,
            )
            if rebuilt and pool_score is not None and meta.get("feasible"):
                sp_hits += 1
                if local_best is None or pool_score.dominates(local_best):
                    local_best, local_routes, found = pool_score, rebuilt, True
                    _record(
                        fleet_size=target, variant="set_partition", seed=0, feasible=True,
                        served=int(instance.num_customers) - int(pool_score.unserved_customer_count),
                        used_vehicles=int(pool_score.used_evs), distance=float(pool_score.total_distance),
                        runtime=0.0, score=pool_score,
                        warm_start_source="route_pool_recombination", bound_kind="known_feasible",
                    )

        if found and local_routes is not None and local_best is not None:
            star_routes, star_score = local_routes, local_best
            m_star = target
            best_feasible = True
            successful_at_mstar = 1
            bound_ann["known_feasible_incumbent"] = int(m_star)
            return True
        return False

    # Warm descending mode
    while m - 1 >= lo and _alive():
        target = m - 1
        ok = _try_target(target, mode="warm")
        if ok:
            m = target
            consecutive_failures = 0
            continue
        consecutive_failures += 1
        if consecutive_failures >= failures_before_stop:
            break
        m = target

    # Independent target probes (heuristic feasibility is not monotonic)
    if best_feasible and _alive() and parent_deadline.remaining() > 60:
        probes = sorted({
            max(lo, (lo + m_star) // 2),
            max(lo, m_star - 2),
            max(lo, m_star - 4),
            max(lo, m_star - 1),
        })
        for target in probes:
            if target >= m_star or not _alive() or parent_deadline.remaining() < 30:
                continue
            if _try_target(target, mode="independent"):
                pass

    if _alive() and best_feasible:
        routes, res = repair_solution(
            instance, star_routes, config=cfg, weights=w, m=m_star, max_rounds=15, mode="regret",
        )
        if res.score is not None and res.score.dominates(star_score):
            star_routes, star_score = routes, res.score

    final = replay_solution(instance, star_routes, config=cfg, weights=w, m=m_star)
    if final.score is not None:
        star_score = final.score

    return FleetSearchResult(
        m_star=int(m_star),
        best_routes=star_routes,
        best_score=star_score,
        benchmark=star_score.to_benchmark(),
        attempts=traces,
        upper_bound_m=int(hi),
        lower_bound_m=int(lo),
        fleet_minimization_performed=True,
        route_pool_size=len(pool),
        route_elimination_attempts=elim_count,
        set_partitioning_hits=sp_hits,
        bound_annotations=bound_ann,
    )
