"""Summarize Hanif's bundled CON/HC CSV. Context only — not a matched baseline."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV = (
    ROOT / "dataset/Hanif/EVRPTW_ChargingStations_Dataset/code/baseline_solver_results.csv"
)
DESCRIPTOR_REFERENCE = {"A": 74.2, "B": 63.7}
TARGET_Q = (600, 700, 800)
TARGET_METHODS = ("CON", "HC")
VARIANTS = ("A", "B")


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def summarize_hanif_baseline_csv(path: Path | None = None) -> dict[str, Any]:
    csv_path = Path(path or DEFAULT_CSV)
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant") or "").strip().upper()
        method = str(row.get("method") or "").strip().upper()
        q = int(float(row.get("Q") or 0))
        if variant not in VARIANTS or method not in TARGET_METHODS or q not in TARGET_Q:
            continue
        grouped[(variant, q, method)].append({
            "instance": row.get("instance"),
            "feasible": str(row.get("feasible") or "").strip() in {"1", "true", "True"},
            "num_routes": _float_or_none(row.get("num_routes")),
            "num_vehicles_used": _float_or_none(row.get("num_vehicles_used")),
        })

    table: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for q in TARGET_Q:
            for method in TARGET_METHODS:
                group = grouped.get((variant, q, method), [])
                n = len(group)
                feas = [r for r in group if r["feasible"]]
                all_routes = [r["num_routes"] for r in group if r["num_routes"] is not None]
                feas_routes = [
                    (r["num_vehicles_used"] if r["num_vehicles_used"] is not None else r["num_routes"])
                    for r in feas
                ]
                feas_routes = [x for x in feas_routes if x is not None]
                table.append({
                    "source": f"Bundled_Hanif_{method}_CSV",
                    "variant": variant,
                    "Q": q,
                    "method": method,
                    "n": n,
                    "n_feasible": len(feas),
                    "feasible_instances": f"{len(feas)}/{n}" if n else "0/0",
                    "mean_routes_all_partitions": (
                        sum(all_routes) / len(all_routes) if all_routes else None
                    ),
                    "mean_routes_feasible_only": (
                        sum(feas_routes) / len(feas_routes) if feas_routes else None
                    ),
                    "scope": "context_only_not_matched_baseline",
                    "semantics": "hanif_con_hc_ignores_wait_mu",
                })

    descriptor_rows = [
        {
            "source": "Reported_Data_Descriptor_reference",
            "variant": variant,
            "Q": 600,
            "method": "descriptor_transcribed",
            "n": None,
            "n_feasible": None,
            "feasible_instances": "not_reported_in_local_artifact",
            "mean_routes_all_partitions": DESCRIPTOR_REFERENCE[variant],
            "mean_routes_feasible_only": None,
            "scope": "context_only_not_matched_baseline",
            "semantics": "transcribed_reference_not_reproduced_in_bundled_csv",
        }
        for variant in VARIANTS
    ]
    return {
        "csv": csv_path.as_posix(),
        "warning": (
            "Context only. Do not place these numbers in a method ablation. "
            "Hanif CON/HC ignore wait_mu; Q and simulator semantics differ from EXTENDED."
        ),
        "descriptor_reference": DESCRIPTOR_REFERENCE,
        "rows": descriptor_rows + table,
    }


def context_table_rows(summary: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    data = summary or summarize_hanif_baseline_csv()
    out = []
    for row in data["rows"]:
        out.append({
            "source": row["source"],
            "variant": row["variant"],
            "Q": row["Q"],
            "method": row["method"],
            "feasible_instances": row["feasible_instances"],
            "mean_routes_all_partitions": row["mean_routes_all_partitions"],
            "mean_routes_feasible_only": row["mean_routes_feasible_only"],
            "scope": row["scope"],
        })
    return out
