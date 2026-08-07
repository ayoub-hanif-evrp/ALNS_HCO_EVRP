"""Build full 9-holdout paper table scaffold (Expert/BC/RL) once evals exist."""
from __future__ import annotations

import csv
import json
from pathlib import Path

FULL_HOLDOUTS = [
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
]
TYPE2 = {"c208_21", "r211_21", "rc208_21"}


def _md_table(rows: list[dict], title: str) -> str:
    lines = [
        f"# {title}",
        "",
        "| Method | Family | Served | Feas | Mean J | n |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['family']} | {r['served']:.3f} | "
            f"{r['feas']:.3f} | {r['J']:.1f} | {r['n']} |"
        )
    lines.append("")
    lines.append(
        "Do **not** claim full Schneider success from type-2-only rows."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    run = Path("outputs/runs/run_20260729_140606")
    type2 = run / "evaluation" / "holdout_type2" / "all_summaries.json"
    rows: list[dict] = []
    if type2.is_file():
        data = json.loads(type2.read_text(encoding="utf-8"))
        for label, s in data.items():
            rows.append({
                "method": label,
                "family": "type2",
                "served": float(s["mean_served"]),
                "feas": float(s["feasibility_rate"]),
                "J": float(s["mean_objective"]),
                "n": int(s["n"]),
            })

    # Placeholder rows for full-9 once curriculum eval lands.
    full_eval = Path("outputs/evaluation/holdout_full9/all_summaries.json")
    if full_eval.is_file():
        data = json.loads(full_eval.read_text(encoding="utf-8"))
        for label, s in data.items():
            rows.append({
                "method": label,
                "family": "full9",
                "served": float(s["mean_served"]),
                "feas": float(s["feasibility_rate"]),
                "J": float(s["mean_objective"]),
                "n": int(s["n"]),
            })
    else:
        rows.append({
            "method": "expert_pending",
            "family": "full9",
            "served": 0.0,
            "feas": 0.0,
            "J": 0.0,
            "n": 9,
        })

    out_dir = Path("outputs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "paper_holdout_tables.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "family", "served", "feas", "J", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    md_path = out_dir / "paper_holdout_tables.md"
    md_path.write_text(
        _md_table(rows, "Schneider holdout tables (type-2 ready; full-9 pending curriculum)")
        + "\n## Holdout instances\n\n"
        + "- Type-2: "
        + ", ".join(sorted(TYPE2))
        + "\n- Full-9: "
        + ", ".join(FULL_HOLDOUTS)
        + "\n",
        encoding="utf-8",
    )
    # Also mirror under the type-2 run for dashboard convenience.
    (run / "evaluation" / "paper_holdout_tables.md").write_text(
        md_path.read_text(encoding="utf-8"), encoding="utf-8",
    )
    print("Wrote", md_path)
    print("Wrote", csv_path)


if __name__ == "__main__":
    main()
