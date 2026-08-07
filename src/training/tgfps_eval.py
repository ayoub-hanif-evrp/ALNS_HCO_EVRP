"""TG-FPS evaluation: explicit methods, 6-instance validation gate, full-9."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.search.diagnostics import attribute_unserved, write_failure_attribution
from src.search.hybrid import (
    METHOD_EXPERT,
    METHOD_EXPERT_REPAIR,
    METHOD_EXPERT_REPAIR_ALNS,
    METHOD_NEURAL_REPAIR,
    METHOD_NEURAL_REPAIR_ALNS,
    NEURAL_METHODS,
    solve_method,
)
from src.search.neural_guide import NeuralCheckpointError, NeuralSearchGuide
from src.search.score import SearchScore
from src.training.gates import (
    aggregate_instance_scores,
    neural_no_regression_gate,
    no_regression_gate,
)
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

FULL9 = [
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
]

VALIDATION6 = [
    "c103_21", "c201_21",
    "r103_21", "r201_21",
    "rc103_21", "rc201_21",
]

BASELINE_DIR = Path("outputs/baselines/full9_20260731")
DET_BASELINE_DIR = Path("outputs/baselines/expert_repair_alns_20260801")


def load_frozen_baseline_rows() -> list[dict]:
    """Load Expert / BC / RL-best per-instance rows from the immutable baseline."""
    rows: list[dict] = []
    root = BASELINE_DIR / "evaluation" / "holdout_full9"
    mapping = {
        "Expert": root / "expert" / "summary.json",
        "Static BC": root / "bc" / "summary.json",
        "Old RL-best": root / "rl_best" / "summary.json",
    }
    for method, path in mapping.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items: list = []
        if isinstance(payload, list):
            items = payload
        elif isinstance(payload, dict):
            pi = payload.get("per_instance")
            if isinstance(pi, list):
                items = pi
            elif isinstance(pi, dict):
                items = list(pi.values())
            else:
                items = payload.get("instances") or payload.get("results") or payload.get("rows") or []
        for it in items:
            if not isinstance(it, dict):
                continue
            name = str(it.get("instance") or it.get("instance_name") or it.get("name") or "")
            rows.append({
                "method": method,
                "instance": name,
                "feasible": bool(it.get("feasible", False)),
                "served_ratio": float(it.get("served_ratio", 0) or 0),
                "unserved": int(it.get("unserved", 0) or 0),
                "used_evs": float(it.get("used_evs", 0) or 0),
                "distance": float(it.get("distance", it.get("total_distance", 0)) or 0),
                "duration": float(it.get("duration", it.get("total_duration", 0)) or 0),
                "charging_time": float(it.get("charging_time", it.get("total_charging_time", 0)) or 0),
                "objective": float(it.get("objective", it.get("paper_objective", it.get("J", 0))) or 0),
                "hard_violations": int(it.get("hard_violations", 0) or 0),
                "runtime": float(it.get("runtime", it.get("runtime_seconds", it.get("seconds", 0))) or 0),
                "m": int(it.get("m", 0) or 0),
                "source": "frozen_baseline",
            })
    return rows


def _row_from_result(name: str, method: str, result) -> dict:
    s = result.score
    n_cust = max(int(result.n_customers or 1), 1)
    return {
        "method": method,
        "instance": name,
        "feasible": bool(result.feasible),
        "complete_feasibility": bool(result.feasible),
        "served_ratio": float(result.served_ratio),
        "served_customers": int(result.served_customers),
        "unserved": int(s.unserved_customer_count),
        "used_evs": int(s.used_evs),
        "distance": float(s.total_distance),
        "duration": float(s.total_duration),
        "charging_time": float(s.total_charging_time),
        "objective": float(s.paper_objective),
        "paper_J": float(s.paper_objective),
        "hard_violations": int(s.hard_violation_count),
        "runtime": float(result.runtime_seconds),
        "m": int(result.m),
        "source": "live",
        "telemetry": result.telemetry or {},
        "bks_gap": None,
    }


def evaluate_methods_on_instances(
    config: dict,
    instance_names: list[str],
    *,
    methods: list[str] | None = None,
    m_default: int = 30,
    out_dir: Path | None = None,
    guide: NeuralSearchGuide | None = None,
    include_frozen_baseline: bool = False,
) -> dict[str, Any]:
    methods = methods or [
        METHOD_EXPERT,
        METHOD_EXPERT_REPAIR,
        METHOD_EXPERT_REPAIR_ALNS,
    ]
    root = Path(config["dataset"]["path"])
    cap = load_capacity_from_config(config)
    weights = CostWeights.from_eval_config(config)
    rows: list[dict] = []
    by_method: dict[str, list[SearchScore]] = {m: [] for m in methods}
    failure_rows: list[dict] = []
    method_errors: dict[str, str] = {}

    # Load guide once if any neural method requested.
    need_neural = any(m in NEURAL_METHODS for m in methods)
    active_guide = guide
    if need_neural and active_guide is None:
        try:
            active_guide = NeuralSearchGuide(config)
        except NeuralCheckpointError as exc:
            for m in methods:
                if m in NEURAL_METHODS:
                    method_errors[m] = str(exc)
            methods = [m for m in methods if m not in NEURAL_METHODS]
            LOGGER.error("Neural methods disabled — checkpoint error: %s", exc)

    for name in instance_names:
        inst = load_instance(root / name, load_capacity=cap)
        for method in list(methods):
            LOGGER.info("Eval | %s | %s", method, name)
            try:
                if method in NEURAL_METHODS and active_guide is None:
                    raise NeuralCheckpointError(f"{method} requires a loaded neural checkpoint")
                res = solve_method(
                    method, inst, m_default, config=config, weights=weights, guide=active_guide,
                )
            except NeuralCheckpointError as exc:
                method_errors[method] = str(exc)
                LOGGER.error("%s failed loudly: %s", method, exc)
                continue
            row = _row_from_result(name, method, res)
            rows.append(row)
            by_method.setdefault(method, []).append(res.score)
            if not res.feasible or res.score.unserved_customer_count > 0:
                attrs = attribute_unserved(inst, res.routes, config=config, m=m_default)
                for a in attrs:
                    if isinstance(a, dict):
                        failure_rows.append({"instance": name, "method": method, **a})

    if include_frozen_baseline:
        rows.extend(load_frozen_baseline_rows())

    summaries = {}
    for method in sorted({r["method"] for r in rows}):
        method_rows = [r for r in rows if r["method"] == method and r.get("instance") != "_aggregate"]
        if not method_rows:
            method_rows = [r for r in rows if r["method"] == method]
        n = max(len(method_rows), 1)
        summaries[method] = {
            "n": len(method_rows),
            "mean_served": sum(float(r["served_ratio"]) for r in method_rows) / n,
            "feasibility_rate": sum(1.0 if r["feasible"] else 0.0 for r in method_rows) / n,
            "mean_objective": sum(float(r["objective"]) for r in method_rows) / n,
            "mean_used_evs": sum(float(r["used_evs"]) for r in method_rows) / n,
            "mean_distance": sum(float(r.get("distance", 0)) for r in method_rows) / n,
            "mean_duration": sum(float(r.get("duration", 0)) for r in method_rows) / n,
            "mean_charging": sum(float(r.get("charging_time", 0)) for r in method_rows) / n,
            "mean_runtime": sum(float(r["runtime"]) for r in method_rows) / n,
            "aggregate_score": list(aggregate_instance_scores(by_method[method]).as_tuple())
            if by_method.get(method) else None,
        }

    report = {
        "summaries": summaries,
        "rows": rows,
        "failure_attribution": failure_rows,
        "method_errors": method_errors,
        "selected_method": None,
        "gates": [],
    }
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "method_comparison.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8",
        )
        if rows:
            flat_rows = []
            for r in rows:
                fr = {k: v for k, v in r.items() if k != "telemetry"}
                flat_rows.append(fr)
            with (out_dir / "method_comparison.csv").open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
                w.writeheader()
                for r in flat_rows:
                    w.writerow(r)
        if failure_rows:
            write_failure_attribution(failure_rows, out_dir / "failure_attribution.csv")
        # Telemetry dump for neural rows
        tel_rows = [r for r in rows if r.get("telemetry")]
        if tel_rows:
            (out_dir / "inference_telemetry.json").write_text(
                json.dumps([{"instance": r["instance"], "method": r["method"], **r["telemetry"]} for r in tel_rows], indent=2),
                encoding="utf-8",
            )
    return report


def run_validation_gate(
    config: dict,
    *,
    validation_instances: list[str] | None = None,
    m_default: int = 30,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate all six reserved validation instances; gate neural vs deterministic."""
    val = list(validation_instances or (config.get("dataset") or {}).get("validation_instances") or VALIDATION6)
    if len(val) < 6:
        LOGGER.warning("Expected 6 validation instances, got %d: %s", len(val), val)
    out_dir = Path(out_dir or "outputs/tgfps_neural_validation")
    methods = [
        METHOD_EXPERT_REPAIR_ALNS,
        METHOD_NEURAL_REPAIR,
        METHOD_NEURAL_REPAIR_ALNS,
    ]
    report = evaluate_methods_on_instances(
        config, val, methods=methods, m_default=m_default, out_dir=out_dir / "raw",
    )
    det_rows = [r for r in report["rows"] if r["method"] == METHOD_EXPERT_REPAIR_ALNS]
    selected = METHOD_EXPERT_REPAIR_ALNS
    gates = []
    for neural_method in (METHOD_NEURAL_REPAIR, METHOD_NEURAL_REPAIR_ALNS):
        chal_rows = [r for r in report["rows"] if r["method"] == neural_method]
        if not chal_rows:
            gates.append({
                "baseline": METHOD_EXPERT_REPAIR_ALNS,
                "challenger": neural_method,
                "accepted": False,
                "reason": report.get("method_errors", {}).get(neural_method, "missing_rows"),
                "selected": METHOD_EXPERT_REPAIR_ALNS,
            })
            continue
        g = neural_no_regression_gate(
            METHOD_EXPERT_REPAIR_ALNS, det_rows, neural_method, chal_rows,
        )
        gates.append({
            "baseline": METHOD_EXPERT_REPAIR_ALNS,
            "challenger": neural_method,
            "accepted": g.accepted,
            "reason": g.reason,
            "selected": g.winner,
            "details": g.details,
        })
        if g.accepted and g.winner == neural_method:
            selected = neural_method

    # Deployed fallback remains Expert+Repair+ALNS unless neural passes.
    if selected in NEURAL_METHODS:
        deployed = selected
    else:
        deployed = METHOD_EXPERT_REPAIR_ALNS

    result = {
        "validation_instances": val,
        "summaries": report["summaries"],
        "gates": gates,
        "selected_method": deployed,
        "neural_passed_gate": deployed in NEURAL_METHODS,
        "rows": report["rows"],
        "method_errors": report.get("method_errors") or {},
        "question": (
            "Does the Transformer improve the validated deterministic solver "
            "through better solution quality or lower search cost?"
        ),
        "answer": (
            "YES — neural method passed the validation gate and is selected."
            if deployed in NEURAL_METHODS
            else "NO — keep Expert+Repair+ALNS; neural did not pass the no-regression gate "
            "(or failed to load/participate)."
        ),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "hybrid_best_validation.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8",
    )
    return result
