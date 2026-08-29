"""Deterministic-FPS: budgeted fleet search → charging → repair → ALNS → pool → rescue → exact accept."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.search.alns import run_alns
from src.search.charging_dp import optimize_charging_plan
from src.search.critical_rescue import critical_rescue
from src.search.deadline import SearchDeadline, StageBudgetPlan, StageRuntimeLedger
from src.search.exact_checks import ExactAcceptanceError, exact_accept_solution
from src.search.fleet_search import solve_with_fleet_search
from src.search.repair import repair_solution
from src.search.route_pool import RoutePool
from src.search.route_replay import replay_solution
from src.search.score import BenchmarkScore, SearchScore
from src.search.set_partitioning import solve_set_partitioning


METHOD_DETERMINISTIC_FPS = "Deterministic-FPS"


@dataclass
class DeterministicFPSResult:
    method: str
    routes: list
    score: SearchScore
    benchmark: BenchmarkScore
    m_star: int
    feasible: bool
    runtime_seconds: float
    fleet_minimization_performed: bool
    stages: dict[str, Any] = field(default_factory=dict)
    fleet_trace: list[dict] = field(default_factory=list)
    exact_check: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = ""
    lower_bound_m: int = 0
    upper_bound_m: int = 0
    charging_visits: int = 0
    unserved_customers: list[int] = field(default_factory=list)
    failure_attribution: list[dict] = field(default_factory=list)
    stage_runtimes: dict[str, Any] = field(default_factory=dict)
    budget_adherence: dict[str, Any] = field(default_factory=dict)
    bound_annotations: dict[str, Any] = field(default_factory=dict)


def _trace_dict(t) -> dict:
    return {
        "m": t.fleet_size,
        "attempt": t.attempt,
        "expert_variant": t.variant,
        "seed": t.seed,
        "warm_start_source": t.warm_start_source,
        "bound_kind": getattr(t, "bound_kind", None),
        "feasible": t.feasible,
        "served": t.served,
        "used_vehicles": t.used_vehicles,
        "distance": t.distance,
        "runtime": t.runtime,
        "failure_reason": t.failure_reason,
    }


def _optimize_routes_charging(instance, routes, *, config, m, deadline=None) -> list:
    out = []
    for nodes, labels in routes:
        if deadline is not None and deadline.expired():
            out.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
            continue
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        if not custs:
            out.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
            continue
        plan = optimize_charging_plan(instance, custs, config=config, m=m, deadline=deadline)
        if plan.feasible and plan.nodes:
            out.append((list(plan.nodes), list(plan.labels)))
        else:
            out.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
    return out


def solve_deterministic_fps(
    instance,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    budget_seconds: float | None = 720.0,
    warm_start_routes: list | None = None,
) -> DeterministicFPSResult:
    cfg = config or {}
    w = weights or (CostWeights.from_eval_config(cfg) if cfg else CostWeights())
    plan = StageBudgetPlan.from_config(cfg)
    global_deadline = SearchDeadline.from_seconds(budget_seconds, stage="deterministic_fps")
    ledger = StageRuntimeLedger()
    t0 = time.perf_counter()
    pool = RoutePool()
    stages: dict[str, Any] = {}
    termination = "completed"
    carry = 0.0  # unused seconds flowing forward
    large_n = int(getattr(instance, "num_customers", 0) or 0) >= 400

    def _stage_deadline(name: str) -> SearchDeadline:
        nonlocal carry
        base = plan.seconds_for(budget_seconds, name)
        if base is None:
            return global_deadline.child(None, stage=name)
        allotted = base + carry
        # Never exceed remaining global budget
        rem = global_deadline.remaining()
        # Reserve minimum tails for later mandatory stages
        reserved_tail = 0.0
        order = ["upper_bound", "fleet_descent", "route_pool", "charging", "repair_alns", "rescue"]
        if name in order:
            idx = order.index(name)
            for later in order[idx + 1 :]:
                # Keep a floor so later stages are not starved (half of their plan, capped)
                sec = plan.seconds_for(budget_seconds, later)
                if sec is not None:
                    reserved_tail += 0.5 * sec
        usable = max(1.0, min(allotted, rem - reserved_tail)) if budget_seconds else allotted
        return global_deadline.child(usable, stage=name)

    # Combined fleet search uses upper + descent fractions
    competitor = str((cfg.get("experiment") or {}).get("competitor") or "").strip().lower()
    if competitor in {"alns", "vns_ts"}:
        intensify_cfg_early = dict((cfg.get("hanif_fleet_intensify") or {}))
        intensify_cfg_early["enabled"] = False
        cfg = dict(cfg)
        cfg["hanif_fleet_intensify"] = intensify_cfg_early
    fleet_secs = None
    if budget_seconds is not None:
        fleet_secs = (plan.seconds_for(budget_seconds, "upper_bound") or 0) + (
            plan.seconds_for(budget_seconds, "fleet_descent") or 0
        )
    # A→B warm start: skip cold upper construct when a feasible A incumbent is provided.
    if warm_start_routes is not None and large_n:
        from src.search.fleet_search import _compact_to_used, _nonempty_route_count, capacity_lower_bound
        t_fleet = time.perf_counter()
        routes = _compact_to_used(instance, warm_start_routes)
        m_star = _nonempty_route_count(instance, routes) or len(routes)
        final_w = replay_solution(instance, routes, config=cfg, weights=w, m=m_star)
        if not final_w.feasible:
            # Fall back to cold search if warm start is not feasible under this TW.
            fleet_deadline = global_deadline.child(fleet_secs, stage="fleet_search")
            fleet = solve_with_fleet_search(
                instance, config=cfg, weights=w, deadline=fleet_deadline, route_pool=pool,
            )
            routes, score, m_star = fleet.best_routes, fleet.best_score, fleet.m_star
            stages["warm_start"] = {"accepted": False, "fallback": "cold_fleet_search"}
            stages["fleet_search"] = {
                "m_star": m_star,
                "lower_bound": fleet.lower_bound_m,
                "upper_bound": fleet.upper_bound_m,
                "n_attempts": len(fleet.attempts),
                "fleet_minimization_performed": True,
                "bound_annotations": fleet.bound_annotations,
            }
            lower_bound_m = int(fleet.lower_bound_m)
            upper_bound_m = int(fleet.upper_bound_m)
            fleet_attempts = fleet.attempts
            fleet_bound_ann = dict(fleet.bound_annotations)
        else:
            score = final_w.score
            stages["warm_start"] = {"accepted": True, "k": int(m_star)}
            stages["fleet_search"] = {
                "m_star": m_star,
                "lower_bound": capacity_lower_bound(instance),
                "upper_bound": m_star,
                "n_attempts": 0,
                "fleet_minimization_performed": False,
                "warm_start": True,
            }
            lower_bound_m = int(capacity_lower_bound(instance))
            upper_bound_m = int(m_star)
            fleet_attempts = []
            fleet_bound_ann = {"warm_start": True}
        fleet_elapsed = time.perf_counter() - t_fleet
        ledger.mark("fleet_search", fleet_elapsed, invoked=True)
        if fleet_secs is not None:
            carry = max(0.0, fleet_secs - fleet_elapsed)
    else:
        fleet_deadline = global_deadline.child(fleet_secs, stage="fleet_search")
        t_fleet = time.perf_counter()
        fleet = solve_with_fleet_search(
            instance, config=cfg, weights=w, deadline=fleet_deadline, route_pool=pool,
        )
        fleet_elapsed = time.perf_counter() - t_fleet
        ledger.mark("fleet_search", fleet_elapsed, invoked=True)
        if fleet_secs is not None:
            carry = max(0.0, fleet_secs - fleet_elapsed)

        routes = fleet.best_routes
        score = fleet.best_score
        m_star = fleet.m_star
        stages["fleet_search"] = {
            "m_star": m_star,
            "lower_bound": fleet.lower_bound_m,
            "upper_bound": fleet.upper_bound_m,
            "n_attempts": len(fleet.attempts),
            "fleet_minimization_performed": True,
            "attempted_fleet_sizes": sorted({a.fleet_size for a in fleet.attempts}),
            "route_pool_size": fleet.route_pool_size,
            "route_elimination_attempts": fleet.route_elimination_attempts,
            "set_partitioning_hits": fleet.set_partitioning_hits,
            "bound_annotations": fleet.bound_annotations,
        }
        lower_bound_m = int(fleet.lower_bound_m)
        upper_bound_m = int(fleet.upper_bound_m)
        fleet_attempts = fleet.attempts
        fleet_bound_ann = dict(fleet.bound_annotations)

    # Early fleet re-descent — skip on Hanif-scale instances (too expensive).
    from src.search.fleet_search import _compact_to_used, _nonempty_route_count
    from src.search.route_elimination import eliminate_one_route

    routes = _compact_to_used(instance, routes)
    used0 = _nonempty_route_count(instance, routes)
    if used0 > 0:
        m_star = min(int(m_star), int(used0))
    early_logs: list[dict] = []
    if not large_n:
        early_frac = 0.18 if (budget_seconds is not None and budget_seconds <= 200) else 0.30
        early_cap = 80.0 if (budget_seconds is not None and budget_seconds <= 200) else 220.0
        early_deadline = global_deadline.child(
            min(early_cap, max(20.0, global_deadline.remaining() * early_frac)), stage="early_redescent"
        )
        while early_deadline.remaining() > 20 and m_star > max(1, int(lower_bound_m)):
            t_el = time.perf_counter()
            new_routes, new_score, atts = eliminate_one_route(
                instance, routes, config=cfg, weights=w, m=m_star,
                deadline=early_deadline.child(
                    min(40.0, early_deadline.remaining()), stage="early_elim"
                ),
                top_k=6,
                focused_alns=False,
            )
            feas = new_score is not None and bool(
                replay_solution(instance, new_routes, config=cfg, weights=w, m=m_star - 1).feasible
            )
            early_logs.append({
                "from_m": m_star, "feasible": feas, "n_attempts": len(atts),
                "runtime": time.perf_counter() - t_el,
            })
            if not feas or new_score is None:
                break
            routes = _compact_to_used(instance, new_routes)
            score = new_score
            m_star = _nonempty_route_count(instance, routes) or (m_star - 1)
            pool.add_from_solution(
                instance, routes, score=score, source_method="early_elim",
                source_fleet_size=m_star, config=cfg,
            )
    stages["early_fleet_redescent"] = {
        "invoked": not large_n, "rounds": len(early_logs), "final_m": m_star, "log": early_logs,
        "skipped_large_n": large_n,
    }
    ledger.mark("early_redescent", sum(x["runtime"] for x in early_logs), invoked=not large_n)

    # Hanif fleet intensification (scalable route elimination after feasible construct).
    # The runner records --disable-intensify; this path must not force it back on.
    intensify_cfg = dict((cfg.get("hanif_fleet_intensify") or {}))
    do_intensify = large_n and bool(intensify_cfg.get("enabled", True))
    if do_intensify and global_deadline.remaining() > 10:
        from src.search.hanif_fleet_intensify import intensify_hanif_fleet
        t_i = time.perf_counter()
        intens_budget = min(
            float(intensify_cfg.get("time_limit_seconds", 90)),
            max(8.0, global_deadline.remaining() * float(intensify_cfg.get("budget_fraction", 0.55))),
        )
        intens_deadline = global_deadline.child(intens_budget, stage="hanif_intensify")
        routes, score, ilog = intensify_hanif_fleet(
            instance, routes, config=cfg, weights=w, m=m_star,
            deadline=intens_deadline,
            max_rounds=int(intensify_cfg.get("max_rounds", 60)),
            top_k=int(intensify_cfg.get("top_k", 10)),
            seeds=list(intensify_cfg.get("seeds") or [0, 1, 2, 3]),
            lower_bound=int(lower_bound_m),
        )
        routes = _compact_to_used(instance, routes)
        m_star = _nonempty_route_count(instance, routes) or m_star
        stages["hanif_fleet_intensify"] = ilog.as_dict()
        ledger.mark("hanif_intensify", time.perf_counter() - t_i, invoked=True)
    else:
        stages["hanif_fleet_intensify"] = {"invoked": False, "skipped": True}

    # Portfolio (short): skip when budget is tight or instance is large
    t_p = time.perf_counter()
    if large_n or global_deadline.remaining() < 15:
        stages["portfolio"] = {"n": 0, "skipped_large_or_short": True}
        ledger.mark("portfolio", 0.0, invoked=False)
    else:
        port_deadline = global_deadline.child(min(carry + 20.0, global_deadline.remaining() * 0.12), stage="portfolio")
        if port_deadline.remaining() > 1.0:
            from src.search.fast_construct import matrix_construct
            variants = ["edf", "regret", "energy", "fleet", "hybrid"]
            elites = []
            for i, v in enumerate(variants):
                if port_deadline.expired():
                    break
                r, sc, feas = matrix_construct(
                    instance, m_star, seed=i, config=cfg, weights=w,
                    deadline=port_deadline.child(min(8.0, port_deadline.remaining()), stage=f"port_{v}"),
                    variant=v,
                )
                pool.add_from_solution(
                    instance, r, score=sc, source_method=f"portfolio_{v}", source_fleet_size=m_star, config=cfg,
                )
                elites.append((sc, r, v))
                if feas and sc.dominates(score):
                    routes, score = r, sc
            stages["portfolio"] = {
                "n": len(elites),
                "elites": len(elites),
                "variants": sorted({v for _, _, v in elites}),
                "seeds": list(range(len(elites))),
                "backend": "matrix_construct",
            }
        else:
            stages["portfolio"] = {"n": 0, "skipped_short": True}
        ledger.mark("portfolio", time.perf_counter() - t_p, invoked=True)

    # Charging (skip full per-route DP on Hanif-scale — keep construction labels)
    t_c = time.perf_counter()
    if large_n:
        stages["charging_optimization"] = {"invoked": False, "skipped_large_n": True}
        ledger.mark("charging", 0.0, invoked=False)
    else:
        ch_deadline = _stage_deadline("charging")
        routes = _optimize_routes_charging(instance, routes, config=cfg, m=m_star, deadline=ch_deadline)
        final_ch = replay_solution(instance, routes, config=cfg, weights=w, m=m_star)
        if final_ch.score is not None and final_ch.score.dominates(score):
            score = final_ch.score
        stages["charging_optimization"] = {"invoked": True, "feasible": bool(final_ch.feasible)}
        ch_elapsed = time.perf_counter() - t_c
        ledger.mark("charging", ch_elapsed, invoked=True)
        if budget_seconds is not None:
            carry = max(0.0, (plan.seconds_for(budget_seconds, "charging") or 0) + carry - ch_elapsed)

    # Repair + ALNS / VNS-TS (literature competitors keep ALNS/VNS even on large n)
    t_ra = time.perf_counter()
    ra_deadline = _stage_deadline("repair_alns")
    if large_n:
        routes, rep = repair_solution(
            instance, routes, config=cfg, weights=w, m=m_star,
            max_rounds=8, mode="regret", regret_k=2, station_candidates=1,
            deadline=ra_deadline,
        )
    else:
        routes, rep = repair_solution(
            instance, routes, config=cfg, weights=w, m=m_star,
            max_rounds=15, mode="regret",
        )
    if rep.score is not None and rep.score.dominates(score):
        score = rep.score
    stages["repair"] = {"feasible": bool(rep.feasible), "unserved": int(rep.unserved_customers), "invoked": True}
    run_alns_now = (not large_n) or competitor == "alns"
    run_vns_now = competitor == "vns_ts"
    if run_vns_now:
        from src.search.vns_ts import run_vns_ts
        vns_budget = max(3.0, min(ra_deadline.remaining(), global_deadline.remaining() - 2.0))
        vns = run_vns_ts(
            instance, routes, config=cfg, weights=w, m=m_star,
            time_limit_seconds=vns_budget, seed=int((cfg.get("seed") or 0)),
        )
        routes, score = vns.routes, vns.score
        stages["alns"] = {"invoked": False, "skipped_for": "vns_ts"}
        stages["vns_ts"] = {
            "iterations": vns.iterations,
            "accepted": vns.accepted,
            "budget_s": vns_budget,
            "invoked": True,
            "telemetry": vns.telemetry,
        }
    elif run_alns_now:
        alns_cfg = dict((cfg.get("hybrid_search") or {}).get("alns") or {})
        if cfg.get("alns"):
            alns_cfg.update(dict(cfg.get("alns") or {}))
        alns_budget = max(3.0, min(ra_deadline.remaining(), float(alns_cfg.get("time_limit_seconds", 40))))
        if competitor == "alns":
            alns_budget = max(alns_budget, min(90.0, global_deadline.remaining() * 0.45))
        alns = run_alns(
            instance, routes, config=cfg, weights=w, m=m_star,
            iterations=int(alns_cfg.get("iterations", 60 if not large_n else 200)),
            time_limit_seconds=alns_budget,
            guide=None,
        )
        routes, score = alns.routes, alns.score
        pool.add_from_solution(
            instance, routes, score=score, source_method="alns", source_fleet_size=m_star, config=cfg,
        )
        stages["alns"] = {
            "iterations": alns.iterations,
            "accepted": alns.accepted,
            "budget_s": alns_budget,
            "invoked": True,
            "neural_guided": False,
            "telemetry": alns.telemetry,
        }
    else:
        stages["alns"] = {"invoked": False, "skipped_large_n": True}

    # Post-ALNS fleet re-descent (skip on Hanif-scale — too slow).
    routes = _compact_to_used(instance, routes)
    used_now = _nonempty_route_count(instance, routes)
    if used_now > 0:
        m_star = min(int(m_star), int(used_now))
    redesc_logs: list[dict] = []
    if not large_n:
        while (
            global_deadline.remaining() > 25
            and m_star > max(1, int(lower_bound_m))
        ):
            t_el = time.perf_counter()
            new_routes, new_score, atts = eliminate_one_route(
                instance, routes, config=cfg, weights=w, m=m_star,
                deadline=global_deadline.child(
                    min(35.0, global_deadline.remaining() * 0.25), stage="post_alns_elim"
                ),
                top_k=6,
                focused_alns=False,
            )
            feas = new_score is not None and bool(
                replay_solution(instance, new_routes, config=cfg, weights=w, m=m_star - 1).feasible
            )
            redesc_logs.append({
                "from_m": m_star,
                "feasible": feas,
                "n_attempts": len(atts),
                "runtime": time.perf_counter() - t_el,
            })
            if not feas or new_score is None:
                break
            routes = _compact_to_used(instance, new_routes)
            score = new_score
            m_star = _nonempty_route_count(instance, routes) or (m_star - 1)
            pool.add_from_solution(
                instance, routes, score=score, source_method="post_alns_elim",
                source_fleet_size=m_star, config=cfg,
            )
    stages["post_alns_fleet_redescent"] = {
        "invoked": not large_n,
        "rounds": len(redesc_logs),
        "final_m": m_star,
        "log": redesc_logs,
        "skipped_large_n": large_n,
    }

    ra_elapsed = time.perf_counter() - t_ra
    ledger.mark("repair_alns", ra_elapsed, invoked=True)

    # Route pool (skip on Hanif-scale)
    t_rp = time.perf_counter()
    rp_deadline = _stage_deadline("route_pool")
    if large_n:
        stages["route_pool"] = {"invoked": False, "skipped_large_n": True, "pool_size": len(pool)}
        ledger.mark("route_pool", 0.0, invoked=False)
    else:
        rebuilt, pool_score, pool_meta = solve_set_partitioning(
            instance, pool, config=cfg, weights=w, m=m_star,
        )
        invoked = bool(rebuilt)
        if rebuilt and pool_score is not None and pool_score.dominates(score):
            routes, score = rebuilt, pool_score
        stages["route_pool"] = {
            **pool_meta,
            "pool_size": len(pool),
            "invoked": True,
            "recombined": invoked,
            "budget_remaining": rp_deadline.remaining(),
        }
        ledger.mark("route_pool", time.perf_counter() - t_rp, invoked=True)

    # Rescue — skip heavy rescue on Hanif-scale
    t_rs = time.perf_counter()
    rescue_deadline = _stage_deadline("rescue")
    unserved_now = [
        i for i, n in enumerate(instance.nodes)
        if n.is_customer and i not in set(replay_solution(instance, routes, config=cfg, weights=w, m=m_star).served_customers)
    ]
    rescue_logs: list[dict] = []
    if unserved_now and not large_n:
        routes, score, rescue_logs = critical_rescue(
            instance, routes, config=cfg, weights=w, m=m_star,
        )
        stages["critical_rescue"] = {
            "invoked_check": True,
            "action_required": True,
            "n_records": len(rescue_logs),
            "invoked": True,
        }
    elif unserved_now and large_n and rescue_deadline.remaining() > 2:
        from src.search.fast_construct import _recover_unserved_fast
        routes, still = _recover_unserved_fast(
            instance, routes, unserved_now, m=m_star, config=cfg, weights=w,
            deadline=rescue_deadline,
        )
        final_r = replay_solution(instance, routes, config=cfg, weights=w, m=m_star)
        if final_r.score is not None:
            score = final_r.score
        stages["critical_rescue"] = {
            "invoked_check": True,
            "action_required": True,
            "n_records": len(still),
            "invoked": True,
            "mode": "hanif_recover",
        }
    else:
        stages["critical_rescue"] = {
            "invoked_check": True,
            "action_required": bool(unserved_now),
            "n_records": 0,
            "invoked": True,
            "skipped_large_n": large_n and bool(unserved_now),
        }
    ledger.mark("rescue", time.perf_counter() - t_rs, invoked=True, note="check_always")

    # Exact final replay
    check, final, missing = exact_accept_solution(
        instance, routes, config=cfg, weights=w, m=m_star,
        claimed_score=score, fail_loud=False,
    )
    if final.score is not None:
        score = final.score
    if check.errors:
        stages["exact_check_errors"] = list(check.errors)
        if any("mismatch" in e or "duplicate" in e or "illegal_charge" in e for e in check.errors):
            termination = "exact_check_failed"
            try:
                exact_accept_solution(
                    instance, routes, config=cfg, weights=w, m=m_star,
                    claimed_score=score, fail_loud=True,
                )
            except ExactAcceptanceError as exc:
                stages["exact_acceptance_error"] = str(exc)

    runtime = time.perf_counter() - t0
    within = global_deadline.within_tolerance(runtime, frac=0.05)
    if not within:
        termination = "budget_overrun" if termination == "completed" else termination

    bench = score.to_benchmark()
    used = int(check.used_vehicles if check.used_vehicles else score.used_evs)
    return DeterministicFPSResult(
        method=METHOD_DETERMINISTIC_FPS,
        routes=routes,
        score=score,
        benchmark=bench,
        m_star=used if final.feasible else int(m_star),
        feasible=bool(final.feasible) and int(score.unserved_customer_count) == 0 and int(score.hard_violation_count) == 0,
        runtime_seconds=runtime,
        fleet_minimization_performed=True,
        stages=stages,
        fleet_trace=[_trace_dict(t) for t in fleet_attempts],
        exact_check=check.as_dict(),
        termination_reason=termination,
        lower_bound_m=int(lower_bound_m),
        upper_bound_m=int(upper_bound_m),
        charging_visits=int(check.charging_visits),
        unserved_customers=list(missing),
        failure_attribution=list(rescue_logs),
        stage_runtimes=ledger.as_dict(),
        budget_adherence={
            "budget_seconds": budget_seconds,
            "runtime_seconds": runtime,
            "within_5pct": within,
            "cleanup_allowance": global_deadline.cleanup_allowance_seconds,
        },
        bound_annotations=dict(fleet_bound_ann),
    )
