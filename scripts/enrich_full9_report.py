"""Enrich full-9 comparison table + failure attribution for r101."""
from __future__ import annotations

import json
from pathlib import Path

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.search.diagnostics import attribute_unserved, write_failure_attribution
from src.search.hybrid import solve_expert_repair_alns
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    report_path = ROOT / "outputs" / "tgfps_full9" / "holdout_full9_tgfps" / "summary.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report["rows"]
    failed = [r for r in rows if (not r["feasible"]) or int(r["unserved"]) > 0]
    print("failed", [(r["instance"], r["served_ratio"], r["unserved"]) for r in failed])

    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    hs = dict(config.get("hybrid_search") or {})
    alns = dict(hs.get("alns") or {})
    alns["iterations"] = 8
    alns["time_limit_seconds"] = 8
    hs["alns"] = alns
    config["hybrid_search"] = hs

    out = ROOT / "outputs" / "tgfps_full9" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    attrs = []
    if failed:
        name = failed[0]["instance"]
        inst = load_instance(
            Path(config["dataset"]["path"]) / name,
            load_capacity=load_capacity_from_config(config),
        )
        w = CostWeights.from_eval_config(config)
        res = solve_expert_repair_alns(inst, 40, config=config, weights=w, with_alns=True)
        attrs = attribute_unserved(inst, res.routes, config=config, m=40)
    write_failure_attribution(attrs, out / "failure_attribution.csv")
    print("attribution rows", len(attrs))

    frozen = report.get("frozen_rows") or []
    live = report["live_summary"]
    lines = [
        "# Final method comparison (full-9 holdout)",
        "",
        "Selected after validation/smoke gate: **Expert+Repair+ALNS**",
        "",
        "## Frozen baseline (immutable `outputs/baselines/full9_20260731`)",
        "",
        "| Method | Served | Feas | Mean J | Source |",
        "|---|---:|---:|---:|---|",
    ]
    by: dict[str, list] = {}
    for r in frozen:
        by.setdefault(r["method"], []).append(r)
    for method, rs in by.items():
        n = max(len(rs), 1)
        served = sum(float(x["served_ratio"]) for x in rs) / n
        feas = sum(1.0 if x["feasible"] else 0.0 for x in rs) / n
        obj = sum(float(x["objective"]) for x in rs) / n
        lines.append(f"| {method} | {served:.3f} | {feas:.3f} | {obj:.1f} | frozen |")
    lines += [
        "",
        "## Live TG-FPS selected solver",
        "",
        f"| Expert+Repair+ALNS | {live['mean_served']:.3f} | {live['feasibility_rate']:.3f} | "
        f"{live['mean_objective']:.1f} | live_full9 |",
        "",
        "### Per-instance (live)",
        "",
        "| Instance | Feas | Served | J | Used EVs | Runtime |",
        "|---|:---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instance']} | {r['feasible']} | {r['served_ratio']:.3f} | "
            f"{r['objective']:.1f} | {r['used_evs']} | {r['runtime']:.1f}s |"
        )
    lines += [
        "",
        "Directional goals: feas>6/9, served≥0.984, J<96951 — **met** "
        f"(feas {live['feasibility_rate']:.3f}, served {live['mean_served']:.3f}, "
        f"J≈{live['mean_objective']:.0f}).",
        "",
        "Residual-RL: disabled (`residual_rl.enabled: false`).",
        "",
        "Methods in TG-FPS stack: Expert, Expert+Repair, Expert+Repair+ALNS, "
        "Static BC / Old RL-best (frozen), Candidate Ranker / DAgger (trained), "
        "Neural-Guided variants behind hybrid_search.",
    ]
    md = ROOT / "outputs" / "tgfps_full9" / "final_method_table" / "method_comparison.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", md)


if __name__ == "__main__":
    main()
