"""Smoke script: TG-FPS validation on small instances + frozen table merge.

Usage:
  python scripts/run_tgfps_smoke_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.search.hybrid import solve_tgfps, solve_expert_repair_alns
from src.search.diagnostics import attribute_unserved, write_failure_attribution
from src.training.tgfps_eval import (
    FULL9,
    build_full_comparison_report,
    evaluate_methods_on_instances,
    load_frozen_baseline_rows,
)
from src.utils.config import load_config


def main() -> None:
    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    # Prefer tiny instances for a fast local smoke; fall back to carved validation.
    small = [
        p.name for p in (ROOT / "dataset" / "Schneider" / "evrptw_instances").iterdir()
        if p.is_dir() and p.name.endswith("C5")
    ][:3]
    out = ROOT / "outputs" / "tgfps_smoke"
    out.mkdir(parents=True, exist_ok=True)

    if small:
        report = evaluate_methods_on_instances(
            config,
            small,
            methods=["Expert", "Expert+Repair", "Expert+Repair+ALNS", "TG-FPS"],
            m_default=6,
            out_dir=out / "smoke_methods",
        )
        print("Smoke selected:", report.get("selected_method"))
        print(json.dumps(report.get("summaries"), indent=2))
    else:
        val = list((config.get("dataset") or {}).get("validation_instances") or [])[:2]
        report = build_full_comparison_report(
            config, out, validation_names=val, run_full9=False,
        )
        print("Validation selected:", report.get("selected"))

    # Always emit frozen baseline merge table for the paper-9 comparison surface.
    frozen = load_frozen_baseline_rows()
    final = out / "final_method_table"
    final.mkdir(parents=True, exist_ok=True)
    methods = {}
    for r in frozen:
        methods.setdefault(r["method"], []).append(r)
    lines = [
        "# Method comparison (frozen baseline full-9 + TG-FPS smoke)",
        "",
        "| Method | Served | Feas | Mean J | Source |",
        "|---|---:|---:|---:|---|",
    ]
    for method, rs in methods.items():
        n = max(len(rs), 1)
        served = sum(float(x["served_ratio"]) for x in rs) / n
        feas = sum(1.0 if x["feasible"] else 0.0 for x in rs) / n
        obj = sum(float(x["objective"]) for x in rs) / n
        lines.append(f"| {method} | {served:.3f} | {feas:.3f} | {obj:.1f} | frozen |")
    if small and report.get("summaries"):
        for method, s in report["summaries"].items():
            lines.append(
                f"| {method} (smoke) | {s['mean_served']:.3f} | {s['feasibility_rate']:.3f} | "
                f"{s['mean_objective']:.1f} | live_smoke |"
            )
    (final / "method_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (final / "frozen_rows.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print("Wrote", final / "method_comparison.md")
    print("Holdouts remain untouched until run_full9_test on validation-selected solver.")
    print("FULL9:", FULL9)


if __name__ == "__main__":
    main()
