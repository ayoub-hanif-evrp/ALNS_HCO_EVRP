"""Rebuild summary tables from saved per-instance rows (no training)."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _rows_from_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("rows", "per_instance", "instances", "results"):
        v = payload.get(key)
        if isinstance(v, list):
            return v
        if isinstance(v, dict):
            return list(v.values())
    return []


def summarize(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    def f(key, default=0.0):
        return sum(float(r.get(key, default) or 0) for r in rows) / n
    return {
        "n": len(rows),
        "mean_served": f("served_ratio"),
        "feasibility_rate": sum(1.0 if r.get("feasible") else 0.0 for r in rows) / n,
        "mean_used_vehicles": f("used_evs", f("used_vehicles")),
        "mean_distance": f("distance", f("total_distance")),
        "mean_charging_time": f("charging_time"),
        "mean_duration": f("duration"),
        "mean_paper_j": f("objective", f("paper_j", f("paper_J"))),
        "mean_runtime": f("runtime", f("seconds")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        default="outputs/baselines/fixed_fleet_expert_repair_alns/holdout_full9_tgfps/summary.json",
    )
    ap.add_argument("--out", default="outputs/baselines/fixed_fleet_expert_repair_alns/recomputed_table.json")
    args = ap.parse_args()
    path = Path(args.input)
    rows = _rows_from_json(path)
    # Prefer live method rows only
    live = [r for r in rows if r.get("source") in (None, "live", "live_full9") or "method" in r]
    if not live:
        live = rows
    by_method: dict[str, list] = {}
    for r in live:
        by_method.setdefault(str(r.get("method", "unknown")), []).append(r)
    tables = {m: summarize(rs) for m, rs in by_method.items()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tables": tables, "n_rows": len(live)}, indent=2), encoding="utf-8")
    # Markdown
    md = out.with_suffix(".md")
    lines = [
        "# Recomputed final table",
        "",
        "| Method | n | Served | Feas | Used EVs | Distance | Charging | Duration | Paper J | Runtime |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for m, s in tables.items():
        lines.append(
            f"| {m} | {s['n']} | {s['mean_served']:.3f} | {s['feasibility_rate']:.3f} | "
            f"{s['mean_used_vehicles']:.1f} | {s['mean_distance']:.1f} | {s['mean_charging_time']:.1f} | "
            f"{s['mean_duration']:.1f} | {s['mean_paper_j']:.1f} | {s['mean_runtime']:.1f} |"
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(tables, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
