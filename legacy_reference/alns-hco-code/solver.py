"""ALNS-HCO entry point.

Phase I is the same GI + routing ALNS as the 15 s ALNS baseline, then just-enough
fleet compression. Phase II keeps searching with lex (K, distance, charging);
the fleet may decrease and may not increase.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.oracle import OracleCache, solution_from_oracle
from src.alns_hco.phase1 import solve_phase1
from src.alns_hco.phase2 import solve_phase2
from src.hanif.replay import replay_solution

MODES = ("alns", "alns_hco_post", "alns_hco_fast", "alns_hco")


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
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode}; expected one of {MODES}")
    hp = HYPERPARAMETERS
    p1 = float(hp["phase1_time_limit_s"] if phase1_s is None else phase1_s)
    p2 = float(hp["phase2_time_limit_s"] if phase2_s is None else phase2_s)
    t0 = time.perf_counter()
    cache = OracleCache()

    compress_s = float(hp.get("fleet_compress_s", 8.0))
    phase1 = solve_phase1(
        instance, time_limit_s=p1, seed=seed, start_routes=start_routes, compress_s=compress_s,
    )
    routes = phase1["routes"]
    ref = phase1["metrics"]
    k_star = int(phase1["k_star"])
    d_star = float(phase1["d_star"])
    t_star = float(phase1["t_star"])
    stages = {
        "phase1": {
            **_metrics_dict(ref),
            "runtime_s": phase1["runtime_s"],
            "iterations": phase1["iterations"],
            "accepted": phase1["accepted"],
            "alns_k": phase1.get("alns_k"),
            "eliminations": phase1.get("eliminations"),
        }
    }

    if mode == "alns":
        metrics = ref
        validation = {"k_star": k_star, "d_star": d_star, "t_star": t_star, "mode": mode}
        return AlnsHcoResult(
            routes=routes, metrics=metrics, runtime_s=time.perf_counter() - t0,
            mode=mode, stages=stages, validation=validation,
        )

    if mode == "alns_hco_post":
        planned, metrics = solution_from_oracle(instance, routes, exact=True, cache=cache)
        if planned and metrics.feasible and metrics.used_k <= k_star:
            routes = planned
        else:
            metrics = ref
        stages["postprocess"] = _metrics_dict(metrics)
        validation = {
            "k_frozen": int(metrics.used_k) <= k_star,
            "sequences_frozen": True,
            "k_star": k_star,
            "d_star": d_star,
            "mode": mode,
        }
        return AlnsHcoResult(
            routes=routes, metrics=metrics, runtime_s=time.perf_counter() - t0,
            mode=mode, stages=stages, validation=validation,
        )

    use_exact = mode == "alns_hco"
    phase2 = solve_phase2(
        instance, routes,
        k_star=k_star, d_star=d_star, t_star=t_star,
        time_limit_s=p2, seed=seed, use_exact=use_exact, cache=cache,
    )
    routes = phase2["routes"]
    metrics = phase2["metrics"]
    if not metrics.feasible or metrics.used_k > k_star:
        routes, metrics = phase1["routes"], ref
    stages["phase2"] = {
        **_metrics_dict(metrics),
        "runtime_s": phase2["runtime_s"],
        "iterations": phase2["iterations"],
        "accepted": phase2["accepted"],
        "exact_evals": phase2.get("exact_evals"),
        "d_cap": phase2.get("d_cap"),
        "t_cap": phase2.get("t_cap"),
    }
    eps_d = float(hp["epsilon_distance"])
    eps_t = float(hp["epsilon_duration"])
    validation = {
        "k_star": k_star,
        "d_star": d_star,
        "t_star": t_star,
        "k_ok": int(metrics.used_k) <= k_star,
        "k_improved": int(metrics.used_k) < int(phase1.get("alns_k") or k_star),
        "distance_ok": float(metrics.distance) <= (1.0 + eps_d) * d_star + 1e-6
        or int(metrics.used_k) < k_star,
        "duration_ok": float(metrics.duration) <= (1.0 + eps_t) * t_star + 1e-6
        or int(metrics.used_k) < k_star,
        "feasible": bool(metrics.feasible),
        "mode": mode,
        "cache_fast_hits": phase2.get("cache_fast_hits"),
        "cache_exact_hits": phase2.get("cache_exact_hits"),
    }
    return AlnsHcoResult(
        routes=routes, metrics=metrics, runtime_s=time.perf_counter() - t0,
        mode=mode, stages=stages, validation=validation,
    )
