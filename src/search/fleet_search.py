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
    """Soft TW heuristic only — never used as a hard search floor.

    Historical bug: treating this as a hard lo blocked type-2 BKS fleets (K≈2–4)
    by forcing lo≈9. Kept for diagnostics / probe ordering.
    """
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
    """Hard floor = capacity (+ instance.m_min). TW pressure is annotation-only."""
    cfg = (config or {}).get("fleet_search") or {}
    lo_cap = capacity_lower_bound(instance)
    lo = max(lo_cap, int(getattr(instance, "m_min", 1) or 1))
    if bool(cfg.get("use_tw_pressure_as_hard_floor", False)):
        lo = max(lo, tw_pressure_lower_bound(instance))
    _, hi = lexicographic_fleet_range(instance, cfg)
    n = int(getattr(instance, "num_customers", 100) or 100)
    # Type-1 (tight TW) needs a higher band than capacity alone suggests.
    tw_soft = tw_pressure_lower_bound(instance)
    type1_hi = int(math.ceil(n / 5.0))  # ~20 on n=100
    hi_cap = int(cfg.get("max_upper_fleet", 0) or 0)
    if hi_cap > 0:
        hi = min(max(hi, type1_hi), hi_cap) if n >= 400 else min(hi, hi_cap)
    else:
        hi = min(hi, max(lo + 12, int(math.ceil(2.2 * lo)), type1_hi, tw_soft + 4))
        if n >= 400:
            # Do not crush Hanif upper bound below ~n/5.
            hi = max(hi, type1_hi, int(math.ceil(n / 5.0)))
    hi = max(lo, int(hi))
    return lo, hi


def adaptive_start_fleet(lo: int, hi: int, instance=None) -> int:
    """First probe between capacity LB and hi — not the padded maximum."""
    start = int(min(hi, max(lo + 2, int(math.ceil(1.55 * lo)) + 1)))
    if instance is not None:
        n = int(getattr(instance, "num_customers", 0) or 0)
        if n >= 400:
            # Hanif: enough for type-1 packing; type-2 still expands down via compact.
            tw = tw_pressure_lower_bound(instance)
            start = max(lo + 8, int(math.ceil(1.75 * lo)), int(math.ceil(n / 6.5)), tw)
            return int(min(hi, start))
        if n >= 80:
            # Avoid starting too low on tight type-1 packs (R1/RC1 failure mode).
            start = max(start, min(hi, int(math.ceil(n / 7.0))))
    return int(min(hi, start))


def _nonempty_route_count(instance, routes: list) -> int:
    return sum(
        1 for nodes, _ in routes
        if any(instance.nodes[n].is_customer for n in nodes)
    )


def _compact_to_used(instance, routes: list) -> list:
    """Drop empty routes so fleet slots match used vehicles."""
    kept = [
        (list(nodes), list(labels) if labels else [None] * len(nodes))
        for nodes, labels in routes
        if any(instance.nodes[n].is_customer for n in nodes)
    ]
    if not kept:
        depot = instance.depot_index
        return [([depot, depot], [None, None])]
    return kept


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

    from src.search.fast_construct import dense_construct, matrix_construct
    from src.search.portfolio import run_variant_episode

    prefer_expert = bool((cfg.get("fleet_search") or {}).get("prefer_expert_construct", False))
    backend = str((cfg.get("fleet_search") or {}).get("construction_backend", "dense") or "dense").lower()
    if use_expert is None:
        use_expert = prefer_expert and backend == "expert"

    if backend == "matrix":
        return matrix_construct(
            instance, fleet_size, seed=seed, config=cfg, weights=w,
            deadline=deadline, variant=expert_variant,
        )
    if backend == "expert" and use_expert and (deadline is None or deadline.remaining() > 20):
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

    # Default: dense sequential insertion (paper-oriented fleet packing, Hanif-fast).
    return dense_construct(
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
        "tw_pressure_soft": int(tw_pressure_lower_bound(instance)),
        "capacity_lower_bound": int(capacity_lower_bound(instance)),
    }

    attempt = 0
    best_routes = None
    best_score = SearchScore.worst()
    best_feasible = False
    large_n = int(getattr(instance, "num_customers", 0) or 0) >= 400
    # Probe from an adaptive mid/low band — not the padded maximum.
    hi = adaptive_start_fleet(lo, hi0, instance)
    if fleet_cfg.get("initial_upper_m") is not None:
        hi = min(hi0, max(lo, int(fleet_cfg["initial_upper_m"])))
    tw_variant = str((cfg.get("dataset") or {}).get("tw_variant") or "A").upper()
    if large_n and tw_variant in ("B", "RELAXED", "RELAXED_TW", "VARIANT_B"):
        # Relaxed TW: start near capacity band, not type-1 padded hi.
        n = int(getattr(instance, "num_customers", 0) or 0)
        hi = min(hi0, max(lo + 5, int(math.ceil(1.40 * lo)), int(math.ceil(n / 14.0))))
    bound_ann["adaptive_start_m"] = int(hi)
    bound_ann["tw_variant"] = tw_variant

    def _alive() -> bool:
        return parent_deadline.check()

    def _record(**kwargs):
        nonlocal attempt
        attempt += 1
        traces.append(FleetAttemptTrace(attempt=attempt, **kwargs))

    fast_mode = bool(fleet_cfg.get("fast_mode", False))
    if large_n:
        # Hanif: feasibility first, then cold shrink. No elim / SP / multi-seed.
        use_elim = False
        use_sp = False
        cold_after_warm = False
        upper_retries = 0
        retries_per = 1
        failures_before_stop = 1
        seeds = [0]
        variants = ["edf"]
    total_fleet = max(1.0, parent_deadline.remaining())
    if fast_mode:
        # Leave most of the budget for post-feasibility fleet shrink on Hanif.
        upper_deadline = parent_deadline.child(
            min(55.0 if large_n else 45.0, total_fleet * (0.35 if large_n else 0.30)),
            stage="upper_phase",
        )
        upper_retries = min(upper_retries, 1)
        retries_per = min(retries_per, 1)
        failures_before_stop = max(failures_before_stop, 3)
    else:
        upper_deadline = parent_deadline.child(min(120.0, total_fleet * 0.32), stage="upper_phase")

    # --- Upper probes with dense construction (expand upward if needed) ---
    upper_order: list[tuple[str, int]] = [("edf", 0)]
    if not fast_mode:
        for v, s in _variant_seed_schedule(variants, seeds, max(len(variants), upper_retries)):
            if (v, s) not in upper_order:
                upper_order.append((v, s))
    upper_order = upper_order[: max(1, min(upper_retries + 1, 2 if fast_mode else 3))]
    for ui, (v, s) in enumerate(upper_order):
        if not upper_deadline.check() or not _alive():
            break
        t0 = time.perf_counter()
        slice_s = min(
            70.0 if large_n else (40.0 if fast_mode else 70.0),
            max(20.0 if large_n else 15.0, upper_deadline.remaining() * 0.90),
        )
        if construct is not None:
            routes = construct(instance, hi)
            routes, res = repair_solution(instance, routes, config=cfg, weights=w, m=hi)
            score = res.score or SearchScore.worst()
            feas = bool(res.feasible)
        else:
            routes, score, feas = construct_solution(
                instance, hi, expert_variant=v, seed=s, config=cfg, weights=w,
                deadline=upper_deadline.child(slice_s, stage="upper_dense"),
                use_expert=False,
            )
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

    # Expand upward with dense construct until feasible (type-1 often needs this).
    expand_step = int(fleet_cfg.get("upper_expand_step", 2))
    expand_max = min(
        int(instance.num_customers),
        max(int(hi0), int(fleet_cfg.get("upper_expand_max", hi0)), adaptive_start_fleet(lo, hi0, instance) + 10),
    )
    expand_hi = hi
    while (not best_feasible) and expand_hi < expand_max and _alive() and parent_deadline.remaining() > 8:
        expand_hi = min(expand_max, expand_hi + expand_step)
        v, s = variants[attempt % len(variants)], seeds[attempt % len(seeds)] + attempt
        t0 = time.perf_counter()
        # Near-miss packs: spend more time repairing at the next fleet size.
        rem = parent_deadline.remaining()
        near = best_score is not None and int(best_score.unserved_customer_count) <= 5
        expand_slice = min(
            55.0 if near else (40.0 if fast_mode else 55.0),
            max(18.0, rem * (0.28 if near else 0.18)),
        )
        routes, score, feas = construct_solution(
            instance, expand_hi, expert_variant="edf", seed=s % 3, config=cfg, weights=w,
            deadline=parent_deadline.child(expand_slice, stage="upper_expand"),
            use_expert=False,
        )
        # If almost complete, one dedicated repair burst at this m (small-n only).
        if (
            (not large_n)
            and (not feas)
            and score is not None
            and int(score.unserved_customer_count) <= 3
            and rem > 20
        ):
            routes2, res2 = repair_solution(
                instance, routes, config=cfg, weights=w, m=expand_hi,
                max_rounds=20, mode="regret", regret_k=3, station_candidates=2,
                deadline=parent_deadline.child(min(25.0, rem * 0.2), stage="upper_expand_repair"),
            )
            if res2.score is not None and (
                res2.feasible or res2.score.dominates(score)
            ):
                routes, score, feas = routes2, res2.score, bool(res2.feasible)
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
        # Shrink retries when time is tight; stop after first success (budget killer was
        # re-proving the same m with every variant).
        rem = parent_deadline.remaining()
        if rem < 45:
            schedule = schedule[:1]
        elif rem < 90:
            schedule = schedule[:2]
        elif rem < 180:
            schedule = schedule[:3]
        else:
            schedule = schedule[: max(2, min(3, len(schedule)))]
        stop_on_first = bool(fleet_cfg.get("stop_target_on_first_feasible", True))
        for trial, (v, s) in enumerate(schedule):
            if not _alive() or parent_deadline.remaining() < 3:
                break
            t0 = time.perf_counter()
            attempt_deadline = parent_deadline.child(
                min(45.0, max(5.0, parent_deadline.remaining() - 5.0)), stage=f"{mode}_attempt",
            )
            warm_routes = star_routes if (warm and best_feasible and mode in ("warm", "elim")) else None
            warm_src = f"warm_from_m={m_star}" if warm_routes is not None else "cold_construct"
            bound_kind = "known_feasible" if warm_routes is not None else (
                "independent_probe" if mode in ("independent", "jump", "binary") else "heuristic_estimate"
            )
            if warm_routes is not None and use_elim and parent_deadline.remaining() > 25:
                elim_budget = min(35.0, max(8.0, parent_deadline.remaining() * 0.22))
                routes, score, atts = eliminate_one_route(
                    instance, warm_routes, config=cfg, weights=w, m=m_star,
                    deadline=parent_deadline.child(elim_budget, stage="route_elim"),
                    top_k=int(fleet_cfg.get("elim_top_k", 6)),
                    focused_alns=bool(fleet_cfg.get("elim_focused_alns", True)),
                )
                elim_count += len(atts)
                feas = score is not None and bool(
                    replay_solution(instance, routes, config=cfg, weights=w, m=target).feasible
                )
                used = _nonempty_route_count(instance, routes)
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

            if (not feas) and cold_after_warm and warm_routes is not None and parent_deadline.remaining() > 15:
                routes2, score2, feas2 = construct_solution(
                    instance, target, expert_variant=v, seed=s + 17, config=cfg, weights=w,
                    warm_start=None, deadline=attempt_deadline, use_expert=False,
                )
                if feas2 or score2.dominates(score):
                    routes, score, feas = routes2, score2, feas2
                    warm_src = "cold_construct_after_warm_fail"
                    bound_kind = "independent_probe"

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
                if stop_on_first:
                    break

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
            star_routes = _compact_to_used(instance, local_routes)
            # Prefer actual used count when solution compacted below target slots.
            used_now = _nonempty_route_count(instance, star_routes)
            star_score = local_best
            m_star = min(int(target), int(used_now)) if used_now > 0 else int(target)
            best_feasible = True
            successful_at_mstar = 1
            bound_ann["known_feasible_incumbent"] = int(m_star)
            return True
        return False

    # Compact first feasible upper to used vehicles (avoid descending from padded hi).
    if best_feasible:
        used0 = max(lo, _nonempty_route_count(instance, best_routes))
        if used0 < m:
            star_routes = _compact_to_used(instance, star_routes)
            final0 = replay_solution(instance, star_routes, config=cfg, weights=w, m=used0)
            if final0.feasible and final0.score is not None:
                star_score = final0.score
                m = used0
                m_star = used0
                hi = used0
                bound_ann["compacted_upper_used"] = int(used0)
                _record(
                    fleet_size=used0, variant="compact_used", seed=0, feasible=True,
                    served=int(instance.num_customers) - int(star_score.unserved_customer_count),
                    used_vehicles=int(star_score.used_evs), distance=float(star_score.total_distance),
                    runtime=0.0, score=star_score,
                    warm_start_source="compact_empty_routes", bound_kind="known_feasible",
                )

    # Hanif fleet shrink: cold dense construct at lower m (no expensive elimination).
    if large_n and best_feasible and _alive() and parent_deadline.remaining() > 25:
        # Big jump first (helps type-2 / relaxed TW where start fleet is padded).
        jump = max(lo, (lo + m_star) // 2)
        if jump < m_star - 3 and parent_deadline.remaining() > 20:
            t0 = time.perf_counter()
            slice_s = min(45.0, max(15.0, parent_deadline.remaining() * 0.35))
            routes, score, feas = construct_solution(
                instance, jump, expert_variant="edf", seed=0,
                config=cfg, weights=w,
                deadline=parent_deadline.child(slice_s, stage="hanif_jump"),
                use_expert=False,
            )
            used = 0
            if feas and score is not None:
                routes = _compact_to_used(instance, routes)
                used = _nonempty_route_count(instance, routes) or jump
                final_s = replay_solution(instance, routes, config=cfg, weights=w, m=used)
                feas = bool(final_s.feasible)
                if feas and final_s.score is not None:
                    score = final_s.score
            _record(
                fleet_size=jump, variant="hanif_jump", seed=0, feasible=bool(feas),
                served=int(instance.num_customers) - int((score or star_score).unserved_customer_count),
                used_vehicles=int(used if feas else (score.used_evs if score else 0)),
                distance=float((score or star_score).total_distance),
                runtime=time.perf_counter() - t0, score=score,
                warm_start_source="cold_dense_jump", bound_kind="known_feasible",
                failure_reason=None if feas else "hanif_jump_infeasible",
            )
            if feas and score is not None:
                star_routes = routes
                star_score = score
                m_star = min(int(jump), int(used)) if used > 0 else int(jump)
                m = m_star
                bound_ann["known_feasible_incumbent"] = int(m_star)

        shrink_failures = 0
        step = max(4, (m_star - lo) // 6)
        shrink_rounds = 0
        while (
            m_star > lo
            and shrink_failures < 3
            and shrink_rounds < 8
            and parent_deadline.remaining() > 18
        ):
            target = max(lo, m_star - step)
            if target >= m_star:
                target = m_star - 1
            if target < lo:
                break
            shrink_rounds += 1
            t0 = time.perf_counter()
            slice_s = min(40.0, max(12.0, parent_deadline.remaining() * 0.30))
            routes, score, feas = construct_solution(
                instance, target, expert_variant="edf", seed=shrink_rounds,
                config=cfg, weights=w,
                deadline=parent_deadline.child(slice_s, stage="hanif_shrink"),
                use_expert=False,
            )
            used = _nonempty_route_count(instance, routes) if feas else 0
            if feas and score is not None:
                routes = _compact_to_used(instance, routes)
                used = _nonempty_route_count(instance, routes) or target
                final_s = replay_solution(instance, routes, config=cfg, weights=w, m=used)
                feas = bool(final_s.feasible)
                if feas and final_s.score is not None:
                    score = final_s.score
            _record(
                fleet_size=target, variant="hanif_shrink", seed=shrink_rounds,
                feasible=bool(feas),
                served=int(instance.num_customers) - int((score or star_score).unserved_customer_count),
                used_vehicles=int(used if feas else (score.used_evs if score else 0)),
                distance=float((score or star_score).total_distance),
                runtime=time.perf_counter() - t0, score=score,
                warm_start_source="cold_dense_shrink", bound_kind="known_feasible",
                failure_reason=None if feas else "hanif_shrink_infeasible",
            )
            if feas and score is not None:
                star_routes = routes
                star_score = score
                m_star = min(int(target), int(used)) if used > 0 else int(target)
                m = m_star
                best_feasible = True
                bound_ann["known_feasible_incumbent"] = int(m_star)
                shrink_failures = 0
                step = max(2, step // 2) if m_star - lo < 2 * step else step
            else:
                shrink_failures += 1
                step = max(2, step // 2)

    # Jump probe: try mid-band before slow linear descent burns the budget.
    if best_feasible and (not large_n) and _alive() and parent_deadline.remaining() > 40:
        jump_targets = sorted({
            max(lo, int(round(0.55 * m_star + 0.45 * lo))),
            max(lo, (lo + m_star) // 2),
            max(lo, m_star - max(3, m_star // 5)),
        }, reverse=True)
        for target in jump_targets:
            if target >= m_star or not _alive():
                continue
            if _try_target(target, mode="jump"):
                m = m_star

    # Warm / elimination descending mode — only after a feasible incumbent exists.
    if best_feasible and not large_n:
        while m - 1 >= lo and _alive():
            target = m - 1
            ok = _try_target(target, mode="warm")
            if ok:
                m = m_star
                consecutive_failures = 0
                continue
            consecutive_failures += 1
            if consecutive_failures >= failures_before_stop:
                break
            m = target

    # Intensive elimination loop while time remains
    elim_rounds = 0 if large_n else int(fleet_cfg.get("intensive_elim_rounds", 8))
    for _ in range(elim_rounds):
        if not (best_feasible and m_star > lo and _alive() and parent_deadline.remaining() > 35):
            break
        t0 = time.perf_counter()
        routes, score, atts = eliminate_one_route(
            instance, star_routes, config=cfg, weights=w, m=m_star,
            deadline=parent_deadline.child(
                min(40.0, parent_deadline.remaining() * 0.25), stage="intensive_elim"
            ),
            top_k=int(fleet_cfg.get("elim_top_k", 8)),
            focused_alns=True,
        )
        elim_count += len(atts)
        feas = score is not None and bool(
            replay_solution(instance, routes, config=cfg, weights=w, m=m_star - 1).feasible
        )
        used = _nonempty_route_count(instance, routes)
        _record(
            fleet_size=m_star - 1, variant="intensive_elim", seed=0, feasible=feas,
            served=int(instance.num_customers) - int((score or star_score).unserved_customer_count),
            used_vehicles=used, distance=float((score or star_score).total_distance),
            runtime=time.perf_counter() - t0, score=score,
            warm_start_source=f"elim_from_m={m_star}", bound_kind="known_feasible",
            failure_reason=None if feas else "intensive_elim_failed",
        )
        if feas and score is not None:
            star_routes = _compact_to_used(instance, routes)
            star_score = score
            m_star = min(m_star - 1, used) if used > 0 else m_star - 1
            m = m_star
            bound_ann["known_feasible_incumbent"] = int(m_star)
        else:
            break

    # Hanif: light leftover recovery only (never full station regret on 1000 nodes).
    if large_n and (not best_feasible) and _alive() and parent_deadline.remaining() > 5:
        from src.search.fast_construct import _recover_unserved_fast
        t0 = time.perf_counter()
        final0 = replay_solution(instance, star_routes, config=cfg, weights=w, m=m_star)
        unserved = [
            i for i, n in enumerate(instance.nodes)
            if n.is_customer and i not in set(final0.served_customers)
        ]
        if unserved:
            star_routes, unserved = _recover_unserved_fast(
                instance, star_routes, unserved, m=m_star, config=cfg, weights=w,
                deadline=parent_deadline.child(
                    min(parent_deadline.remaining() * 0.85, 45.0), stage="hanif_recover"
                ),
            )
            final1 = replay_solution(instance, star_routes, config=cfg, weights=w, m=m_star)
            if final1.score is not None:
                star_score = final1.score
                best_feasible = bool(final1.feasible)
                if best_feasible:
                    bound_ann["known_feasible_incumbent"] = int(m_star)
        _record(
            fleet_size=m_star, variant="hanif_recover", seed=0,
            feasible=bool(best_feasible),
            served=int(instance.num_customers) - int(star_score.unserved_customer_count),
            used_vehicles=int(star_score.used_evs), distance=float(star_score.total_distance),
            runtime=time.perf_counter() - t0, score=star_score,
            warm_start_source="station_recover", bound_kind="expanded_upper",
        )

    # Independent target probes (feasibility is not monotonic)
    if best_feasible and (not large_n) and _alive() and parent_deadline.remaining() > 25:
        probes = sorted({
            max(lo, (lo + m_star) // 2),
            max(lo, m_star - 2),
            max(lo, m_star - 4),
            max(lo, m_star - 1),
            max(lo, int(round(1.15 * lo))),
            max(lo, int(round(1.35 * lo))),
        })
        for target in probes:
            if target >= m_star or not _alive() or parent_deadline.remaining() < 15:
                continue
            if _try_target(target, mode="independent"):
                m = m_star

    if _alive() and best_feasible:
        routes, res = repair_solution(
            instance, star_routes, config=cfg, weights=w, m=m_star, max_rounds=15, mode="regret",
        )
        if res.score is not None and res.score.dominates(star_score):
            star_routes, star_score = routes, res.score

    star_routes = _compact_to_used(instance, star_routes)
    final = replay_solution(instance, star_routes, config=cfg, weights=w, m=m_star)
    if final.score is not None:
        star_score = final.score
        used_final = _nonempty_route_count(instance, star_routes)
        if final.feasible and used_final > 0:
            m_star = min(int(m_star), int(used_final))

    bound_ann["remaining_after_search"] = float(parent_deadline.remaining())
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
