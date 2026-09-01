"""One (instance, TW, method, seed) solve for the Hanif paper master CSV."""
from __future__ import annotations

from src.alns_hco.solver import solve_alns_hco
from src.common.config import (
    GI_DEADLINE_S, METHOD_ALNS_FR, METHOD_ALNS_HCO, METHOD_ALNS_PR,
    METHOD_ALNS_PR_HCO, METHOD_GI, METHOD_SA, METHOD_VNS, PHASE1_S, PHASE2_S,
    TOTAL_BUDGET_S,
)
from src.hanif.benchmarks.alns import solve_alns
from src.hanif.benchmarks.greedy import solve_greedy
from src.hanif.benchmarks.sa import solve_sa
from src.hanif.benchmarks.vns import solve_vns
from src.hanif.loader import load_hanif
from src.hanif.replay import replay_solution


def _family(name: str) -> str:
    return name.split("_")[0]


def _row(instance_name, tw, method, seed, metrics, runtime_s, *, extra=None) -> dict:
    extra = extra or {}
    return {
        "instance": instance_name,
        "family": _family(instance_name),
        "tw": tw,
        "method": method,
        "seed": int(seed),
        "feasible": bool(metrics.feasible),
        "served": int(len(metrics.customers)),
        "unserved": int(metrics.unserved),
        "hard_violations": int(metrics.hard_violations),
        "k": int(metrics.used_k),
        "distance": float(metrics.distance),
        "duration": float(metrics.duration),
        "customer_waiting": float(getattr(metrics, "customer_waiting", 0.0) or 0.0),
        "charging_visits": int(metrics.charging_visits),
        "charged_energy": float(metrics.charged_energy),
        "station_waiting": float(metrics.station_waiting),
        "charging_time": float(metrics.charging_time),
        "charging_expense": float(metrics.charging_expense),
        "runtime_s": float(runtime_s),
        "termination_reason": extra.get("termination_reason", "time_limit"),
        "phase1_s": extra.get("phase1_s"),
        "phase2_s": extra.get("phase2_s"),
        "phase1_k": extra.get("phase1_k"),
        "phase1_distance": extra.get("phase1_distance"),
        "phase1_charging_expense": extra.get("phase1_charging_expense"),
        "k_star": extra.get("k_star"),
        "k_frozen_ok": extra.get("k_frozen_ok"),
        "routes": extra.get("routes"),
    }


def solve_job(job: dict) -> dict:
    name, tw, method, seed = job["name"], job["tw"], job["method"], int(job["seed"])
    inst = load_hanif(name, tw)
    save_routes = bool(job.get("save_routes"))

    if method == METHOD_GI:
        out = solve_greedy(inst, deadline=GI_DEADLINE_S)
        row = _row(name, tw, method, 0, out["metrics"], out["runtime_s"], extra={
            "termination_reason": "construction",
            "routes": out["routes"] if save_routes else None,
        })
        return row

    if method == METHOD_ALNS_FR:
        gi = solve_greedy(inst, deadline=GI_DEADLINE_S)
        out = solve_alns(inst, gi["routes"], time_limit_s=TOTAL_BUDGET_S, seed=seed)
        return _row(name, tw, method, seed, out["metrics"], out["runtime_s"] + gi["runtime_s"], extra={
            "routes": out["routes"] if save_routes else None,
        })

    if method == METHOD_VNS:
        gi = solve_greedy(inst, deadline=GI_DEADLINE_S)
        out = solve_vns(inst, gi["routes"], time_limit_s=TOTAL_BUDGET_S, seed=seed)
        return _row(name, tw, method, seed, out["metrics"], out["runtime_s"] + gi["runtime_s"], extra={
            "routes": out["routes"] if save_routes else None,
        })

    if method == METHOD_SA:
        gi = solve_greedy(inst, deadline=GI_DEADLINE_S)
        out = solve_sa(inst, gi["routes"], time_limit_s=TOTAL_BUDGET_S, seed=seed)
        return _row(name, tw, method, seed, out["metrics"], out["runtime_s"] + gi["runtime_s"], extra={
            "routes": out["routes"] if save_routes else None,
        })

    if method == METHOD_ALNS_PR:
        result = solve_alns_hco(inst, mode="alns_pr", phase1_s=TOTAL_BUDGET_S, phase2_s=0.0, seed=seed)
        p1 = result.stages.get("phase1") or {}
        return _row(name, tw, method, seed, result.metrics, result.runtime_s, extra={
            "phase1_s": p1.get("runtime_s"),
            "phase1_k": p1.get("k"),
            "phase1_distance": p1.get("distance"),
            "phase1_charging_expense": p1.get("charging_expense"),
            "k_star": result.validation.get("k_star"),
            "k_frozen_ok": True,
            "routes": result.routes if save_routes else None,
        })

    if method == METHOD_ALNS_PR_HCO:
        result = solve_alns_hco(inst, mode="alns_hco_post", phase1_s=TOTAL_BUDGET_S, phase2_s=0.0, seed=seed)
        p1 = result.stages.get("phase1") or {}
        return _row(name, tw, method, seed, result.metrics, result.runtime_s, extra={
            "phase1_s": p1.get("runtime_s"),
            "phase1_k": p1.get("k"),
            "phase1_distance": p1.get("distance"),
            "phase1_charging_expense": p1.get("charging_expense"),
            "k_star": result.validation.get("k_star"),
            "k_frozen_ok": result.validation.get("k_frozen_ok"),
            "termination_reason": "alns_pr_then_hco",
            "routes": result.routes if save_routes else None,
        })

    result = solve_alns_hco(
        inst, mode="alns_hco", phase1_s=PHASE1_S, phase2_s=PHASE2_S, seed=seed,
    )
    p1 = result.stages.get("phase1") or {}
    p2 = result.stages.get("phase2") or {}
    return _row(name, tw, method, seed, result.metrics, result.runtime_s, extra={
        "phase1_s": p1.get("runtime_s"),
        "phase2_s": p2.get("runtime_s"),
        "phase1_k": p1.get("k"),
        "phase1_distance": p1.get("distance"),
        "phase1_charging_expense": p1.get("charging_expense"),
        "k_star": result.validation.get("k_star"),
        "k_frozen_ok": result.validation.get("k_frozen_ok"),
        "routes": result.routes if save_routes else None,
    })
