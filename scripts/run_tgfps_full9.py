"""One untouched full-9 eval with capped search budgets.

Uses smoke/validation gate result when available; otherwise runs a cheap
2-instance validation, then evaluates the selected solver once on the paper-9.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.search.hybrid import solve_expert_repair_alns, solve_tgfps
from src.training.tgfps_eval import FULL9, load_frozen_baseline_rows, evaluate_methods_on_instances
from src.utils.config import load_config


def _cap_config(config: dict) -> dict:
    config = dict(config)
    hs = dict(config.get("hybrid_search") or {})
    alns = dict(hs.get("alns") or {})
    alns["iterations"] = 15
    alns["time_limit_seconds"] = 12
    hs["alns"] = alns
    cons = dict(hs.get("construction") or {})
    cons["expert_variants_list"] = ["hybrid"]
    hs["construction"] = cons
    fleet = dict(hs.get("fleet_search") or {})
    fleet["descending_search"] = False
    fleet["diversified_retries"] = 1
    hs["fleet_search"] = fleet
    config["hybrid_search"] = hs
    return config


def main() -> None:
    config = _cap_config(load_config(str(ROOT / "configs" / "train_tgfps.yaml")))
    out = ROOT / "outputs" / "tgfps_full9"
    out.mkdir(parents=True, exist_ok=True)

    # Prefer prior smoke selection if present (no-regression already applied).
    smoke_sel = None
    smoke_json = ROOT / "outputs" / "tgfps_smoke" / "smoke_methods" / "method_comparison.json"
    if smoke_json.is_file():
        smoke_sel = json.loads(smoke_json.read_text(encoding="utf-8")).get("selected_method")

    val = {"summaries": {}, "gates": [], "selected_method": smoke_sel}
    if smoke_sel is None:
        val_names = list((config.get("dataset") or {}).get("validation_instances") or [])[:2]
        print("Validation (cheap):", val_names, flush=True)
        t0 = time.perf_counter()
        val = evaluate_methods_on_instances(
            config,
            val_names,
            methods=["Expert+Repair", "Expert+Repair+ALNS"],
            m_default=30,
            out_dir=out / "hybrid_validation",
        )
        val["validation_runtime_s"] = time.perf_counter() - t0
    else:
        print(f"Reusing smoke validation selection: {smoke_sel}", flush=True)
        # Record carved validation names for audit (not evaluated live here).
        (out / "hybrid_validation").mkdir(parents=True, exist_ok=True)
        (out / "hybrid_validation" / "note.json").write_text(
            json.dumps({
                "selected_from_smoke": smoke_sel,
                "validation_instances_reserved": list(
                    (config.get("dataset") or {}).get("validation_instances") or []
                ),
                "holdouts_untouched_until_now": FULL9,
            }, indent=2),
            encoding="utf-8",
        )

    selected = smoke_sel or val.get("selected_method") or "Expert+Repair+ALNS"
    print(f"Selected solver: {selected}", flush=True)
    (out / "hybrid_best_validation.json").write_text(
        json.dumps({
            "selected": selected,
            "smoke_selected": smoke_sel,
            "validation": val.get("summaries"),
            "gates": val.get("gates"),
        }, indent=2),
        encoding="utf-8",
    )

    # Untouched full-9 once
    print("Full-9 holdouts:", FULL9, flush=True)
    weights = CostWeights.from_eval_config(config)
    root = Path(config["dataset"]["path"])
    cap = load_capacity_from_config(config)
    rows = []
    for name in FULL9:
        print(f"FULL9 | {selected} | {name}", flush=True)
        inst = load_instance(root / name, load_capacity=cap)
        m = 40
        if selected == "TG-FPS":
            res = solve_tgfps(
                inst, config=config, weights=weights, m=m,
                use_alns=True, use_portfolio=True, time_limit_seconds=12.0,
            )
        elif selected == "Expert+Repair":
            res = solve_expert_repair_alns(inst, m, config=config, weights=weights, with_alns=False)
        else:
            res = solve_expert_repair_alns(inst, m, config=config, weights=weights, with_alns=True)
        n_cust = max(int(inst.num_customers), 1)
        rows.append({
            "method": selected,
            "instance": name,
            "feasible": bool(res.feasible),
            "served_ratio": (n_cust - int(res.score.unserved_customer_count)) / n_cust,
            "unserved": int(res.score.unserved_customer_count),
            "used_evs": int(res.score.used_evs),
            "objective": float(res.score.paper_objective),
            "distance": float(res.score.total_distance),
            "duration": float(res.score.total_duration),
            "charging_time": float(res.score.total_charging_time),
            "hard_violations": int(res.score.hard_violation_count),
            "runtime": float(res.runtime_seconds),
            "m": int(res.m),
            "source": "live_full9",
        })
        print(
            f"  feas={res.feasible} served={rows[-1]['served_ratio']:.3f} "
            f"J={rows[-1]['objective']:.1f} t={rows[-1]['runtime']:.1f}s",
            flush=True,
        )

    n = max(len(rows), 1)
    live_summary = {
        "n": len(rows),
        "mean_served": sum(r["served_ratio"] for r in rows) / n,
        "feasibility_rate": sum(1.0 if r["feasible"] else 0.0 for r in rows) / n,
        "mean_objective": sum(r["objective"] for r in rows) / n,
        "mean_used_evs": sum(r["used_evs"] for r in rows) / n,
        "mean_runtime": sum(r["runtime"] for r in rows) / n,
    }
    frozen = load_frozen_baseline_rows()
    hold9 = out / "holdout_full9_tgfps"
    hold9.mkdir(parents=True, exist_ok=True)
    report = {
        "selected_method": selected,
        "live_summary": live_summary,
        "rows": rows,
        "frozen_rows": frozen,
    }
    (hold9 / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Final comparison markdown
    final = out / "final_method_table"
    final.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Final method comparison (full-9 holdout)",
        "",
        f"Selected after validation/smoke gate: **{selected}**",
        "",
        "## Frozen baseline (immutable `outputs/baselines/full9_20260731`)",
        "",
        "| Method | Served | Feas | Mean J | Source |",
        "|---|---:|---:|---:|---|",
    ]
    by = {}
    for r in frozen:
        by.setdefault(r["method"], []).append(r)
    for method, rs in by.items():
        nn = max(len(rs), 1)
        lines.append(
            f"| {method} | {sum(float(x['served_ratio']) for x in rs)/nn:.3f} | "
            f"{sum(1.0 if x['feasible'] else 0.0 for x in rs)/nn:.3f} | "
            f"{sum(float(x['objective']) for x in rs)/nn:.1f} | frozen |"
        )
    lines += [
        "",
        "## Live selected solver (this run)",
        "",
        f"| {selected} | {live_summary['mean_served']:.3f} | {live_summary['feasibility_rate']:.3f} | "
        f"{live_summary['mean_objective']:.1f} | live_full9 |",
        "",
        "Directional goals (not hard-tuned): feas>6/9, served≥0.984, J<96951.",
    ]
    (final / "method_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (final / "method_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "run_summary.json").write_text(
        json.dumps({"selected": selected, "live_summary": live_summary}, indent=2),
        encoding="utf-8",
    )
    print("Selected:", selected, flush=True)
    print("Live summary:", json.dumps(live_summary), flush=True)
    print("Wrote", final / "method_comparison.md", flush=True)


if __name__ == "__main__":
    main()
