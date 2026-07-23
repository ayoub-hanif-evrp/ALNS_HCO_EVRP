"""Multi-checkpoint evaluation and final report for Hanif warm-start pipeline."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.data.instance_class import CLASSES, instance_class
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def _macro_mean(per_class: dict[str, float | None]) -> float | None:
    vals = [per_class.get(c) for c in CLASSES if per_class.get(c) is not None]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def summarize_eval_by_class(eval_summary: dict) -> dict[str, Any]:
    """Build C/R/RC + macro metrics from an evaluate() summary."""
    rows = eval_summary.get("per_instance") or eval_summary.get("results") or []
    if not rows and isinstance(eval_summary.get("holdout_report"), dict):
        # Fall back to aggregate-only report.
        hold = eval_summary["holdout_report"]
        return {
            "served": {c: None for c in CLASSES},
            "feasibility": {c: None for c in CLASSES},
            "objective": {c: None for c in CLASSES},
            "unserved": {c: None for c in CLASSES},
            "hard_violations": {c: None for c in CLASSES},
            "macro_served": hold.get("mean_served_ratio") or hold.get("served_rate"),
            "macro_feasibility": hold.get("feasibility_rate") or hold.get("feasibility"),
            "macro_objective": hold.get("mean_objective") or hold.get("objective"),
            "aggregate": hold,
        }

    buckets: dict[str, list[dict]] = {c: [] for c in CLASSES}
    for row in rows:
        name = str(row.get("instance_name") or row.get("instance") or "")
        buckets[instance_class(name)].append(row)

    def _mean(items: list[dict], *keys: str) -> float | None:
        vals = []
        for it in items:
            for k in keys:
                if k in it and it[k] is not None:
                    vals.append(float(it[k]))
                    break
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    served = {c: _mean(buckets[c], "served_ratio", "served_rate") for c in CLASSES}
    feas = {c: _mean(buckets[c], "feasible", "feasibility") for c in CLASSES}
    # feasible may be 0/1
    for c in CLASSES:
        if feas[c] is None:
            feas[c] = _mean(buckets[c], "is_feasible")
    obj = {c: _mean(buckets[c], "objective", "cost") for c in CLASSES}
    unserved = {c: _mean(buckets[c], "unserved_customers", "unserved") for c in CLASSES}
    hard = {}
    for c in CLASSES:
        batt = _mean(buckets[c], "battery_violations", "mean_battery_violations") or 0.0
        tw = _mean(buckets[c], "time_window_violations", "tw_violations") or 0.0
        cap = _mean(buckets[c], "capacity_violations") or 0.0
        hard[c] = float(batt + tw + cap) if buckets[c] else None

    return {
        "served": served,
        "feasibility": feas,
        "objective": obj,
        "unserved": unserved,
        "hard_violations": hard,
        "macro_served": _macro_mean(served),
        "macro_feasibility": _macro_mean(feas),
        "macro_objective": _macro_mean(obj),
        "counts": {c: len(buckets[c]) for c in CLASSES},
    }


def select_final_checkpoint(candidates: dict[str, dict]) -> tuple[str, dict]:
    """Prefer validation macro-feasibility, then served, then objective (lower better)."""
    best_name = None
    best_row = None
    best_key = None
    for name, row in candidates.items():
        if not row:
            continue
        feas = row.get("macro_feasibility")
        served = row.get("macro_served")
        obj = row.get("macro_objective")
        if feas is None and served is None:
            continue
        key = (
            float(feas) if feas is not None else -1.0,
            float(served) if served is not None else -1.0,
            -float(obj) if obj is not None else float("-inf"),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_name = name
            best_row = row
    if best_name is None:
        # Fallback preference order.
        for name in (
            "bc_best", "rl_best_feasibility", "rl_best_served",
            "rl_best_objective", "bc_last", "rl_last",
        ):
            if name in candidates and candidates[name]:
                return name, candidates[name]
        raise RuntimeError("No checkpoint candidates available for final selection")
    return best_name, best_row or {}


def write_final_report(
    run_dir: Path,
    *,
    dataset_report: dict,
    split_meta: dict,
    stages: dict[str, Any],
    checkpoint_comparison: dict,
    selected: str,
    runtime: dict,
) -> tuple[Path, Path]:
    report = {
        "run_dir": str(run_dir),
        "dataset": dataset_report,
        "split": {
            "train": split_meta.get("train_instances"),
            "validation": split_meta.get("validation_instances"),
            "test": split_meta.get("test_instances"),
            "split_by_class": split_meta.get("split_by_class"),
            "pilot_enabled": split_meta.get("pilot_enabled"),
        },
        "expert": stages.get("expert"),
        "behavior_cloning": stages.get("bc"),
        "rl": stages.get("rl"),
        "checkpoint_comparison": checkpoint_comparison,
        "chosen_final_checkpoint": selected,
        "runtime": runtime,
        "remaining_limitations": [
            "B3 pilot uses a deterministic small subset per class; scale to full Hanif for final paper numbers.",
            "Station wait uses E[wait]=wait_mu only; wait_sigma is not sampled.",
            "density_score is a dynamic feature, not a physics constraint.",
            "Exact masks on 1000-customer instances remain computationally heavy.",
        ],
    }
    json_path = run_dir / "final_report.json"
    json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Hanif warm-start final report",
        "",
        f"- Run: `{run_dir}`",
        f"- Chosen checkpoint: **{selected}**",
        f"- Pilot: {split_meta.get('pilot_enabled')}",
        "",
        "## Split",
        f"- Train: {split_meta.get('train_instances')}",
        f"- Validation: {split_meta.get('validation_instances')}",
        f"- Test: {split_meta.get('test_instances')}",
        "",
        "## Checkpoint comparison (test)",
        "",
    ]
    methods = (checkpoint_comparison.get("methods") or {})
    for name, row in methods.items():
        lines.append(
            f"- **{name}**: served={row.get('macro_served')} "
            f"feas={row.get('macro_feasibility')} obj={row.get('macro_objective')}"
        )
    lines.extend([
        "",
        "## Runtime",
        json.dumps(runtime, indent=2),
        "",
        "## Limitations",
    ])
    for lim in report["remaining_limitations"]:
        lines.append(f"- {lim}")
    md_path = run_dir / "final_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Final report -> %s / %s", json_path, md_path)
    return json_path, md_path


def evaluate_checkpoint_on_split(
    config: dict,
    checkpoint: Path,
    *,
    out_dir: Path,
    label: str,
) -> dict:
    """Run greedy evaluate() for one checkpoint; write metrics under out_dir."""
    from src.evaluation.evaluate import evaluate

    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(json.dumps(config, default=str))
    cfg["outputs"] = {
        **(cfg.get("outputs") or {}),
        "results": str(out_dir),
        "routes": str(out_dir / "routes"),
        "figures": str(out_dir / "figures"),
    }
    t0 = time.time()
    summary = evaluate(cfg, checkpoint=str(checkpoint))
    elapsed = time.time() - t0
    by_class = summarize_eval_by_class(summary if isinstance(summary, dict) else {})
    payload = {
        "label": label,
        "checkpoint": str(checkpoint),
        "elapsed_seconds": elapsed,
        "summary": summary,
        "by_class": by_class,
        "macro_served": by_class.get("macro_served"),
        "macro_feasibility": by_class.get("macro_feasibility"),
        "macro_objective": by_class.get("macro_objective"),
    }
    (out_dir / f"{label}_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )
    return payload
