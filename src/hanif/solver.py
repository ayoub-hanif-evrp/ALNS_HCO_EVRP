"""Two-stage CARE-FI for the Hanif 1000-customer benchmark."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.hanif.output import serialize_routes, validate_record, write_json
from src.hanif.paths import FROZEN_HANIF_CLASSICAL, TMP_OUT
from src.hanif import BENCHMARK, SOLVER_VERSION
from src.hanif.charging_refinement import refine_charging
from src.hanif.construction import construct_feasible
from src.hanif.fleet_reduction import reduce_fleet
from src.hanif.replay import customer_sequence, replay_solution, sequences_match


@dataclass
class CareFiResult:
    routes: list
    stage: str
    metrics: Any
    runtime_s: float
    stages: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    termination_reason: str = "completed"


def load_archived_classical(name: str, tw: str) -> list | None:
    variant = "relaxed_tw" if str(tw).lower().startswith("relax") else "original_tw"
    path = FROZEN_HANIF_CLASSICAL / variant / "routes" / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    routes = []
    for r in payload.get("routes") or []:
        routes.append((
            [int(n) for n in r["nodes"]],
            [None if x is None else float(x) for x in (r.get("labels") or [None] * len(r["nodes"]))],
        ))
    return routes or None


def solve_stage1(
    instance,
    *,
    budget_s: float = 180.0,
    nearby_routes: int = 8,
    insert_positions: int = 3,
    nearby_stations: int = 3,
    max_elim_candidates: int = 8,
    warm_routes: list | None = None,
) -> CareFiResult:
    t0 = time.perf_counter()
    stages: dict[str, Any] = {}
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


def solve_stage2(instance, classical_routes: list, *, max_passes: int = 20) -> CareFiResult:
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
    stage2_res = solve_stage2(instance, stage1.routes)
    return stage1, stage2_res


def result_record(
    instance,
    result: CareFiResult,
    *,
    tw: str,
    configuration: dict,
) -> dict[str, Any]:
    m = result.metrics
    row = {
        "benchmark": BENCHMARK,
        "instance": instance.instance_name,
        "solver_version": SOLVER_VERSION,
        "configuration": configuration,
        "runtime_s": result.runtime_s,
        "battery_capacity": float(instance.vehicle.battery_capacity),
        "energy_consumption_rate": float(instance.vehicle.energy_consumption_rate),
        "used_k": int(m.used_k),
        "distance": float(m.distance),
        "feasible": bool(m.feasible),
        "termination_reason": result.termination_reason,
        "tw_variant": tw,
        "stage": result.stage,
        "duration": float(m.duration),
        "charging_visits": int(m.charging_visits),
        "charged_energy": float(m.charged_energy),
        "station_waiting": float(m.station_waiting),
        "charging_time": float(m.charging_time),
        "charging_expense": float(m.charging_expense),
        "phi": (result.stages or {}).get("phi_proposed") or (result.stages or {}).get("phi_classical"),
        "served": len(m.customers),
        "hard_violations": int(m.hard_violations),
        "validation": result.validation,
        "stages": result.stages,
    }
    validate_record(row, stage2=result.stage == "proposed")
    return row


def write_result(
    instance,
    result: CareFiResult,
    *,
    tw: str,
    out_root: Path | None = None,
    configuration: dict | None = None,
) -> Path:
    root = out_root or (TMP_OUT / "hanif")
    variant = "relaxed_tw" if str(tw).lower().startswith("relax") else "original_tw"
    folder = root / ("proposed" if result.stage == "proposed" else "classical") / variant
    routes_dir = folder / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    row = result_record(instance, result, tw=tw, configuration=configuration or {})
    payload = {
        **row,
        "routes": serialize_routes(result.routes),
        "customer_sequences": [
            list(customer_sequence(instance, n)) for n, _ in result.routes if customer_sequence(instance, n)
        ],
    }
    path = routes_dir / f"{instance.instance_name}.json"
    write_json(path, payload)
    return path
