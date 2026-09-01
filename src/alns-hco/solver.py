"""ALNS-HCO: Phase I ALNS-PR, then Phase II at frozen K=K* with one HCO."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.alns_hco.hco import OracleCache, solution_from_hco
from src.alns_hco.phase1 import solve_phase1
from src.alns_hco.phase2 import solve_phase2
from src.common.config import PHASE1_S, PHASE2_S

MODES = ("alns_pr", "alns_hco_post", "alns_hco")


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
    mode: str = "alns_hco",
    phase1_s: float | None = None,
    phase2_s: float | None = None,
    seed: int = 0,
    start_routes: list | None = None,
) -> AlnsHcoResult:
    if mode == "alns":
        mode = "alns_pr"
    if mode == "alns_hco_fast":
        mode = "alns_hco"
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}; expected one of {MODES}")
    p1 = PHASE1_S if phase1_s is None else float(phase1_s)
    p2 = PHASE2_S if phase2_s is None else float(phase2_s)
    t0 = time.perf_counter()
    cache = OracleCache()

    phase1 = solve_phase1(instance, time_limit_s=p1, seed=seed, start_routes=start_routes)
    routes = phase1["routes"]
    ref = phase1["metrics"]
    k_star = int(phase1["k_star"])
    stages = {
        "phase1": {
            **_metrics_dict(ref),
            "runtime_s": phase1["runtime_s"],
            "iterations": phase1["iterations"],
            "accepted": phase1["accepted"],
            "alns_k": phase1.get("alns_k"),
        }
    }

    if mode == "alns_pr":
        return AlnsHcoResult(
            routes=routes, metrics=ref, runtime_s=time.perf_counter() - t0,
            mode=mode, stages=stages,
            validation={"k_star": k_star, "k_frozen_ok": True, "mode": mode},
        )

    if mode == "alns_hco_post":
        planned, metrics = solution_from_hco(instance, routes, cache=cache)
        if planned and metrics.feasible and int(metrics.used_k) == k_star:
            routes, ref = planned, metrics
        stages["postprocess"] = _metrics_dict(ref)
        return AlnsHcoResult(
            routes=routes, metrics=ref, runtime_s=time.perf_counter() - t0,
            mode=mode, stages=stages,
            validation={
                "k_star": k_star,
                "k_frozen_ok": int(ref.used_k) == k_star,
                "sequences_frozen": True,
                "mode": mode,
            },
        )

    phase2 = solve_phase2(
        instance, routes, k_star=k_star, time_limit_s=p2, seed=seed, cache=cache,
    )
    routes, metrics = phase2["routes"], phase2["metrics"]
    if (not metrics.feasible) or int(metrics.used_k) != k_star:
        routes, metrics = phase1["routes"], ref
    stages["phase2"] = {
        **_metrics_dict(metrics),
        "runtime_s": phase2["runtime_s"],
        "iterations": phase2["iterations"],
        "accepted": phase2["accepted"],
        "hco_evals": phase2.get("hco_evals"),
    }
    return AlnsHcoResult(
        routes=routes, metrics=metrics, runtime_s=time.perf_counter() - t0,
        mode=mode, stages=stages,
        validation={
            "k_star": k_star,
            "k_ok": int(metrics.used_k) == k_star,
            "k_frozen_ok": int(metrics.used_k) == k_star,
            "feasible": bool(metrics.feasible),
            "mode": mode,
            "hco_cache_hits": phase2.get("cache_hits"),
        },
    )


def solve_alns_pr_budget(instance, *, time_limit_s: float, seed: int = 0) -> AlnsHcoResult:
    return solve_alns_hco(instance, mode="alns_pr", phase1_s=time_limit_s, phase2_s=0.0, seed=seed)
