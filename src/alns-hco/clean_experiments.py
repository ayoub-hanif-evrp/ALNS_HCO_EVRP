"""Solve one (instance, TW, method, seed) job for the clean Hanif revision study."""
from __future__ import annotations

import time

from src.alns_hco.checksum import customer_checksum
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.shared_phase1 import ensure_shared_phase1
from src.alns_hco.solver import solve_alns_hco
from src.alns_hco.station_log import extract_route_rows, extract_station_visits, solution_row
from src.common.config import (
    GI_DEADLINE_S, METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_FIXED_DETOUR,
    METHOD_FIXED_HET, METHOD_GI, METHOD_PHASE1_PR15, METHOD_PHASE2_DETOUR,
    METHOD_PHASE2_HET, METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_SA, METHOD_VNS,
    PHASE1_S, PHASE2_S, TOTAL_BUDGET_S,
)
from src.common.energy_accounting import attach_energy_reporting, attach_search_diagnostics
from src.common.paths import hanif_dirs
from src.hanif.benchmarks.alns import solve_alns
from src.hanif.benchmarks.greedy import solve_greedy
from src.hanif.benchmarks.sa import solve_sa
from src.hanif.benchmarks.vns import solve_vns
from src.hanif.loader import load_hanif


def _family(name: str) -> str:
    return name.split("_")[0]


def _finalize_reporting(
    row: dict,
    instance,
    routes: list | None,
    *,
    stages: dict | None = None,
    validation: dict | None = None,
    shared_phase1_iterations: int | None = None,
) -> dict:
    """Attach post-hoc energy + search diagnostics. Never feeds back into solvers.

    Separates solver wall-clock from reporting overhead / end-to-end time.
    Avoids recomputing energy when ``solution_row`` already filled it.
    """
    t0 = time.perf_counter()
    if row.get("solver_runtime_s") is None and row.get("runtime_s") is not None:
        row["solver_runtime_s"] = float(row["runtime_s"])
    attach_energy_reporting(row, instance, routes)  # no-op if already filled
    attach_search_diagnostics(
        row,
        stages=stages,
        validation=validation,
        shared_phase1_iterations=shared_phase1_iterations,
    )
    overhead = time.perf_counter() - t0
    row["reporting_overhead_s"] = float(overhead)
    solver_rt = row.get("solver_runtime_s")
    if solver_rt is not None:
        row["end_to_end_runtime_s"] = float(solver_rt) + float(overhead)
    else:
        row["end_to_end_runtime_s"] = None
    row["execution_time_s"] = row.get("end_to_end_runtime_s")
    return row


def _base_meta(instance_name, tw, method, seed, phase="final") -> dict:
    return {
        "instance": instance_name,
        "family": _family(instance_name),
        "tw": tw,
        "method": method,
        "seed": int(seed),
        "phase": phase,
    }


def _error_row(instance_name, tw, method, seed, err: str) -> dict:
    return {
        "instance": instance_name,
        "family": _family(instance_name),
        "tw": tw,
        "method": method,
        "seed": int(seed),
        "feasible": False,
        "served": 0,
        "unserved": -1,
        "hard_violations": -1,
        "k": None,
        "distance": None,
        "duration": None,
        "customer_waiting": None,
        "charging_visits": None,
        "charged_energy": None,
        "station_waiting": None,
        "charging_time": None,
        "charging_expense": None,
        "runtime_s": 0.0,
        "termination_reason": "error",
        "phase1_s": None,
        "phase2_s": None,
        "phase1_k": None,
        "phase1_distance": None,
        "phase1_charging_expense": None,
        "k_star": None,
        "k_frozen_ok": False,
        "phase1_checksum": None,
        "hco_mode": None,
        "eps": None,
        "error": str(err),
        "station_visits": [],
        "route_rows": [],
    }


def solve_job(job: dict) -> dict:
    name, tw, method, seed = job["name"], job["tw"], job["method"], int(job["seed"])
    shared_root = hanif_dirs()["shared_phase1"]
    eps = float(job.get("eps", HYPERPARAMETERS["phase2_distance_tolerance"]))
    k_cand = int(job.get("candidates_per_criterion", HYPERPARAMETERS["hco_candidates_per_criterion"]))
    phase2_budget = float(job["phase2_s"]) if job.get("phase2_s") is not None else float(PHASE2_S)

    try:
        if method == METHOD_GI:
            inst = load_hanif(name, tw)
            out = solve_greedy(inst, deadline=GI_DEADLINE_S)
            meta = _base_meta(name, tw, method, 0)
            row = solution_row(
                out["metrics"], meta=meta, runtime_s=out["runtime_s"],
                instance=inst, routes=out["routes"],
            )
            row.update({
                "termination_reason": "construction",
                "phase1_s": None, "phase2_s": None,
                "phase1_k": None, "phase1_distance": None, "phase1_charging_expense": None,
                "k_star": None, "k_frozen_ok": True, "hco_mode": None, "eps": None, "error": None,
            })
            row["station_visits"] = extract_station_visits(inst, out["routes"], meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, out["routes"], meta={**meta, "phase": "final"})
            return _finalize_reporting(row, inst, out["routes"])

        if method == METHOD_ALNS_FR:
            inst = load_hanif(name, tw)
            gi = solve_greedy(inst, deadline=GI_DEADLINE_S)
            out = solve_alns(inst, gi["routes"], time_limit_s=TOTAL_BUDGET_S, seed=seed)
            meta = _base_meta(name, tw, method, seed)
            row = solution_row(
                out["metrics"], meta=meta, runtime_s=out["runtime_s"] + gi["runtime_s"],
                instance=inst, routes=out["routes"],
            )
            row.update({
                "termination_reason": "time_limit",
                "phase1_s": None, "phase2_s": None,
                "phase1_k": None, "phase1_distance": None, "phase1_charging_expense": None,
                "k_star": None, "k_frozen_ok": True, "hco_mode": None, "eps": None, "error": None,
                "alns_iterations": out.get("iterations"),
            })
            row["station_visits"] = extract_station_visits(inst, out["routes"], meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, out["routes"], meta={**meta, "phase": "final"})
            return _finalize_reporting(row, inst, out["routes"])

        if method == METHOD_VNS:
            inst = load_hanif(name, tw)
            gi = solve_greedy(inst, deadline=GI_DEADLINE_S)
            out = solve_vns(inst, gi["routes"], time_limit_s=TOTAL_BUDGET_S, seed=seed)
            meta = _base_meta(name, tw, method, seed)
            row = solution_row(
                out["metrics"], meta=meta, runtime_s=out["runtime_s"] + gi["runtime_s"],
                instance=inst, routes=out["routes"],
            )
            row.update({
                "termination_reason": "time_limit",
                "phase1_s": None, "phase2_s": None,
                "phase1_k": None, "phase1_distance": None, "phase1_charging_expense": None,
                "k_star": None, "k_frozen_ok": True, "hco_mode": None, "eps": None, "error": None,
            })
            row["station_visits"] = extract_station_visits(inst, out["routes"], meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, out["routes"], meta={**meta, "phase": "final"})
            return _finalize_reporting(row, inst, out["routes"])

        if method == METHOD_SA:
            inst = load_hanif(name, tw)
            gi = solve_greedy(inst, deadline=GI_DEADLINE_S)
            out = solve_sa(inst, gi["routes"], time_limit_s=TOTAL_BUDGET_S, seed=seed)
            meta = _base_meta(name, tw, method, seed)
            row = solution_row(
                out["metrics"], meta=meta, runtime_s=out["runtime_s"] + gi["runtime_s"],
                instance=inst, routes=out["routes"],
            )
            row.update({
                "termination_reason": "time_limit",
                "phase1_s": None, "phase2_s": None,
                "phase1_k": None, "phase1_distance": None, "phase1_charging_expense": None,
                "k_star": None, "k_frozen_ok": True, "hco_mode": None, "eps": None, "error": None,
            })
            row["station_visits"] = extract_station_visits(inst, out["routes"], meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, out["routes"], meta={**meta, "phase": "final"})
            return _finalize_reporting(row, inst, out["routes"])

        if method == METHOD_ALNS_PR:
            inst = load_hanif(name, tw)
            result = solve_alns_hco(inst, mode="alns_pr", phase1_s=TOTAL_BUDGET_S, phase2_s=0.0, seed=seed)
            meta = _base_meta(name, tw, method, seed)
            p1 = result.stages.get("phase1") or {}
            row = solution_row(
                result.metrics, meta=meta, runtime_s=result.runtime_s,
                phase1_checksum=p1.get("customer_checksum"),
                instance=inst, routes=result.routes,
            )
            row.update({
                "termination_reason": "time_limit",
                "phase1_s": p1.get("runtime_s"), "phase2_s": None,
                "phase1_k": p1.get("k"), "phase1_distance": p1.get("distance"),
                "phase1_charging_expense": p1.get("charging_expense"),
                "k_star": result.validation.get("k_star"), "k_frozen_ok": True,
                "hco_mode": None, "eps": None, "error": None,
            })
            row["station_visits"] = extract_station_visits(inst, result.routes, meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, result.routes, meta={**meta, "phase": "final"})
            return _finalize_reporting(
                row, inst, result.routes, stages=result.stages, validation=result.validation,
            )

        # Shared Phase I and all dependents
        shared = ensure_shared_phase1(shared_root, instance_name=name, tw=tw, seed=seed)
        inst = shared["instance"]
        rec = shared["record"]
        routes = shared["routes"]
        chk = rec["customer_checksum"]
        k_star = int(rec["k_star"])
        d_pr = float(rec["metrics"]["distance"])

        if method == METHOD_PHASE1_PR15:
            meta = _base_meta(name, tw, method, seed, phase="phase1")
            row = solution_row(
                shared["metrics"], meta=meta, runtime_s=rec["phase1_s"], phase1_checksum=chk,
                instance=inst, routes=routes,
            )
            row.update({
                "termination_reason": "shared_phase1",
                "phase1_s": rec["phase1_s"], "phase2_s": 0.0,
                "phase1_k": rec["metrics"]["k"], "phase1_distance": d_pr,
                "phase1_charging_expense": rec["metrics"]["charging_expense"],
                "k_star": k_star, "k_frozen_ok": True,
                "hco_mode": None, "eps": None, "error": None,
                "alns_phase1_iterations": rec.get("iterations"),
                "alns_iterations": rec.get("iterations"),
            })
            row["station_visits"] = extract_station_visits(inst, routes, meta={**meta, "phase": "phase1"})
            row["route_rows"] = extract_route_rows(inst, routes, meta={**meta, "phase": "phase1"})
            return _finalize_reporting(
                row, inst, routes,
                shared_phase1_iterations=rec.get("iterations"),
            )

        hco_mode = {
            METHOD_FIXED_DETOUR: "detour",
            METHOD_PHASE2_DETOUR: "detour",
            METHOD_FIXED_HET: "heterogeneous",
            METHOD_PHASE2_HET: "heterogeneous",
            METHOD_PHASE2_PLAIN: "plain",
            METHOD_PHASE2_NOHCO: "nohco",
        }.get(method, "heterogeneous")
        if method in (METHOD_FIXED_DETOUR, METHOD_FIXED_HET):
            result = solve_alns_hco(
                inst, mode="fixed_route_hco", seed=seed, start_routes=routes,
                hco_mode=hco_mode, k_star=k_star, d_pr=d_pr, phase1_checksum=chk,
                candidates_per_criterion=k_cand,
            )
            # Assert shared checksum identity
            if result.validation.get("phase1_checksum") != chk:
                raise AssertionError("fixed-route branch used different Phase-I checksum")
            if result.validation.get("customer_checksum") != chk:
                raise AssertionError("fixed-route HCO changed customer sequence")
            meta = _base_meta(name, tw, method, seed)
            row = solution_row(
                result.metrics, meta=meta, runtime_s=result.runtime_s + float(rec["phase1_s"]),
                phase1_checksum=chk, instance=inst, routes=result.routes,
            )
            act = (result.stages.get("postprocess") or {}).get("hco_activity") or result.validation.get("hco_activity")
            row.update({
                "termination_reason": "fixed_route_hco",
                "phase1_s": rec["phase1_s"], "phase2_s": 0.0,
                "phase1_k": k_star, "phase1_distance": d_pr,
                "phase1_charging_expense": rec["metrics"]["charging_expense"],
                "k_star": k_star, "k_frozen_ok": result.validation.get("k_frozen_ok"),
                "hco_mode": hco_mode, "eps": None, "error": None,
                "hco_activity": act,
                "alns_phase1_iterations": rec.get("iterations"),
            })
            row["station_visits"] = extract_station_visits(inst, result.routes, meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, result.routes, meta={**meta, "phase": "final"})
            return _finalize_reporting(
                row, inst, result.routes, stages=result.stages, validation=result.validation,
                shared_phase1_iterations=rec.get("iterations"),
            )

        if method in (METHOD_PHASE2_PLAIN, METHOD_PHASE2_NOHCO, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET):
            result = solve_alns_hco(
                inst, mode="phase2", phase2_s=phase2_budget, seed=seed, start_routes=routes,
                hco_mode=hco_mode, k_star=k_star, d_pr=d_pr, phase1_checksum=chk,
                distance_tolerance=eps, candidates_per_criterion=k_cand,
            )
            if result.validation.get("phase1_checksum") != chk:
                raise AssertionError("Phase-II branch used different Phase-I checksum")
            if int(result.metrics.used_k) != k_star:
                raise AssertionError(f"Phase II changed K: {result.metrics.used_k} != {k_star}")
            meta = _base_meta(name, tw, method, seed)
            p2 = result.stages.get("phase2") or {}
            row = solution_row(
                result.metrics, meta=meta,
                runtime_s=result.runtime_s + float(rec["phase1_s"]),
                phase1_checksum=chk, instance=inst, routes=result.routes,
            )
            term = {
                METHOD_PHASE2_PLAIN: "phase2_plain",
                METHOD_PHASE2_NOHCO: "phase2_nohco",
            }.get(method, "phase2")
            row.update({
                "termination_reason": term,
                "phase1_s": rec["phase1_s"], "phase2_s": p2.get("runtime_s"),
                "phase1_k": k_star, "phase1_distance": d_pr,
                "phase1_charging_expense": rec["metrics"]["charging_expense"],
                "k_star": k_star, "k_frozen_ok": result.validation.get("k_frozen_ok"),
                "hco_mode": hco_mode, "eps": eps, "error": None,
                "hco_activity": p2.get("hco_activity") or result.validation.get("hco_activity"),
                "alns_phase1_iterations": rec.get("iterations"),
            })
            row["station_visits"] = extract_station_visits(inst, result.routes, meta={**meta, "phase": "final"})
            row["route_rows"] = extract_route_rows(inst, result.routes, meta={**meta, "phase": "final"})
            return _finalize_reporting(
                row, inst, result.routes, stages=result.stages, validation=result.validation,
                shared_phase1_iterations=rec.get("iterations"),
            )

        raise ValueError(f"unknown method {method}")
    except Exception as exc:  # noqa: BLE001 — log failures explicitly
        return _error_row(name, tw, method, seed, repr(exc))
