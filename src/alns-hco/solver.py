"""ALNS-HCO solver entry points for the clean revision study."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal

from src.alns_hco.checksum import assert_same_checksum, customer_checksum
from src.alns_hco.hco import solution_from_hco
from src.alns_hco.hco_activity import OracleCache
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.phase1 import solve_phase1
from src.alns_hco.phase2 import solve_phase2
from src.common.config import PHASE1_S, PHASE2_S
from src.hanif.replay import replay_solution

HcoMode = Literal["plain", "nohco", "detour", "heterogeneous"]

MODES = (
    "alns_pr",
    "fixed_route_hco",
    "phase2",
    "alns_hco_post",
    "alns_hco",
)


@dataclass
class AlnsHcoResult:
    routes: list
    metrics: Any
    runtime_s: float
    mode: str
    stages: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)


def _metrics_dict(m) -> dict:
    return {
        "feasible": bool(m.feasible),
        "served": int(len(m.customers)),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
        "k": int(m.used_k),
        "distance": float(m.distance),
        "duration": float(m.duration),
        "charging_visits": int(m.charging_visits),
        "charged_energy": float(m.charged_energy),
        "charging_time": float(m.charging_time),
        "charging_expense": float(m.charging_expense),
        "station_waiting": float(m.station_waiting),
        "customer_waiting": float(getattr(m, "customer_waiting", 0.0) or 0.0),
    }


def solve_alns_hco(
    instance,
    *,
    mode: str = "phase2",
    phase1_s: float | None = None,
    phase2_s: float | None = None,
    seed: int = 0,
    start_routes: list | None = None,
    hco_mode: HcoMode = "heterogeneous",
    d_pr: float | None = None,
    k_star: int | None = None,
    distance_tolerance: float | None = None,
    candidates_per_criterion: int | None = None,
    phase1_checksum: str | None = None,
) -> AlnsHcoResult:
    if mode == "alns":
        mode = "alns_pr"
    if mode == "alns_hco_fast":
        mode = "phase2"
    if mode == "alns_hco":
        mode = "phase2"
    if mode == "alns_hco_post":
        mode = "fixed_route_hco"
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}; expected one of {MODES}")

    hp = HYPERPARAMETERS
    p1 = PHASE1_S if phase1_s is None else float(phase1_s)
    p2 = PHASE2_S if phase2_s is None else float(phase2_s)
    k_cand = int(hp["hco_candidates_per_criterion"] if candidates_per_criterion is None else candidates_per_criterion)
    t0 = time.perf_counter()
    cache = OracleCache()
    stages: dict[str, Any] = {}

    if start_routes is None:
        phase1 = solve_phase1(instance, time_limit_s=p1, seed=seed)
        routes = phase1["routes"]
        ref = phase1["metrics"]
        k_star_v = int(phase1["k_star"])
        d_pr_v = float(ref.distance)
        chk = customer_checksum(instance, routes)
        stages["phase1"] = {
            **_metrics_dict(ref),
            "runtime_s": phase1["runtime_s"],
            "iterations": phase1["iterations"],
            "accepted": phase1["accepted"],
            "customer_checksum": chk,
        }
    else:
        routes = start_routes
        ref = replay_solution(instance, routes)
        k_star_v = int(k_star if k_star is not None else ref.used_k)
        d_pr_v = float(d_pr if d_pr is not None else ref.distance)
        chk = phase1_checksum or customer_checksum(instance, routes)
        stages["phase1"] = {
            **_metrics_dict(ref),
            "runtime_s": 0.0,
            "customer_checksum": chk,
            "from_shared": True,
        }

    if phase1_checksum is not None and chk != phase1_checksum:
        raise AssertionError(f"Phase-I checksum mismatch: {chk} != {phase1_checksum}")

    if mode == "alns_pr":
        return AlnsHcoResult(
            routes=routes, metrics=ref, runtime_s=time.perf_counter() - t0,
            mode=mode, stages=stages,
            validation={"k_star": k_star_v, "k_frozen_ok": True, "mode": mode, "customer_checksum": chk},
        )

    if mode == "fixed_route_hco":
        planned, metrics = solution_from_hco(
            instance, routes, cache=cache, mode=hco_mode, candidates_per_criterion=k_cand,
            non_worsening=True,
        )
        if planned and metrics.feasible and int(metrics.used_k) == k_star_v:
            # Customer sequence must be unchanged.
            assert_same_checksum(instance, routes, planned, context="fixed_route_hco")
            routes, ref = planned, metrics
        stages["postprocess"] = {
            **_metrics_dict(ref), "hco_mode": hco_mode,
            "hco_activity": cache.activity.as_dict(),
        }
        return AlnsHcoResult(
            routes=routes, metrics=ref, runtime_s=time.perf_counter() - t0,
            mode=mode, stages=stages,
            validation={
                "k_star": k_star_v,
                "k_frozen_ok": int(ref.used_k) == k_star_v,
                "sequences_frozen": True,
                "customer_checksum": customer_checksum(instance, routes),
                "phase1_checksum": chk,
                "hco_mode": hco_mode,
                "mode": mode,
                "hco_activity": cache.activity.as_dict(),
                "non_worsening": True,
                "fixed_route_objective": "lex(C_ch, T_wait, T_ch, D)",
            },
        )

    # Phase II charging-aware ALNS (or plain fixed-fleet ALNS)
    phase2 = solve_phase2(
        instance, routes, k_star=k_star_v, d_pr=d_pr_v, time_limit_s=p2, seed=seed,
        cache=cache, hco_mode=hco_mode, distance_tolerance=distance_tolerance,
        candidates_per_criterion=k_cand,
    )
    routes2, metrics = phase2["routes"], phase2["metrics"]
    if (not metrics.feasible) or int(metrics.used_k) != k_star_v:
        routes2, metrics = routes, ref
    stages["phase2"] = {
        **_metrics_dict(metrics),
        "runtime_s": phase2["runtime_s"],
        "iterations": phase2["iterations"],
        "accepted": phase2["accepted"],
        "hco_evals": phase2.get("hco_evals"),
        "rejected_distance": phase2.get("rejected_distance"),
        "hco_mode": hco_mode,
        "eps": phase2.get("eps"),
        "hco_activity": phase2.get("hco_activity"),
        "cache_hits": phase2.get("cache_hits"),
        "cache_calls": phase2.get("cache_calls"),
    }
    return AlnsHcoResult(
        routes=routes2, metrics=metrics, runtime_s=time.perf_counter() - t0,
        mode=mode, stages=stages,
        validation={
            "k_star": k_star_v,
            "k_ok": int(metrics.used_k) == k_star_v,
            "k_frozen_ok": int(metrics.used_k) == k_star_v,
            "feasible": bool(metrics.feasible),
            "mode": mode,
            "hco_mode": hco_mode,
            "phase1_checksum": chk,
            "customer_checksum": customer_checksum(instance, routes2),
            "hco_cache_hits": phase2.get("cache_hits"),
            "hco_activity": phase2.get("hco_activity"),
        },
    )


def solve_alns_pr_budget(instance, *, time_limit_s: float, seed: int = 0) -> AlnsHcoResult:
    return solve_alns_hco(instance, mode="alns_pr", phase1_s=time_limit_s, phase2_s=0.0, seed=seed)
