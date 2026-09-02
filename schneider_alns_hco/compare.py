"""Compare ALNS-HCO Schneider results to published EVRPTW-PR BKS."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from bks import CITATION, compare_row
from config import RESULT_FIELDS, RESULTS, TIE_TOL


def _f(raw, default=None):
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _lex_key(row: dict) -> tuple:
    feas = 0 if str(row.get("feasible")).lower() in {"1", "true"} else 1
    return (
        feas,
        int(_f(row.get("unserved"), 0) or 0),
        int(_f(row.get("hard_violations"), 0) or 0),
        int(_f(row.get("k"), 10**9) or 10**9),
        float(_f(row.get("distance"), 1e18) or 1e18),
    )


def load_results(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def best_per_instance(rows: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups[row["instance"]].append(row)
    best = []
    for name in sorted(groups):
        pick = min(groups[name], key=_lex_key)
        best.append(pick)
    return best


def compile_comparison(results_csv: Path | None = None) -> dict:
    results_csv = results_csv or (RESULTS / "results.csv")
    rows = load_results(results_csv)
    best = best_per_instance(rows)
    compared = []
    for row in best:
        rec = compare_row(row["instance"], int(_f(row["k"], 0) or 0), float(_f(row["distance"], 0) or 0))
        rec.update({
            "family": row.get("family"),
            "seed": int(_f(row.get("seed"), 0) or 0),
            "feasible": str(row.get("feasible")).lower() in {"1", "true"},
            "unserved": int(_f(row.get("unserved"), 0) or 0),
            "runtime_s": _f(row.get("runtime_s")),
            "k_star": _f(row.get("k_star")),
            "k_frozen_ok": str(row.get("k_frozen_ok")).lower() in {"1", "true"},
        })
        compared.append(rec)

    feas = [r for r in compared if r["feasible"] and r["unserved"] == 0]
    wins = sum(1 for r in feas if r["delta_k"] < 0)
    ties = sum(1 for r in feas if r["delta_k"] == 0)
    losses = sum(1 for r in feas if r["delta_k"] > 0)
    dist_wins = sum(1 for r in feas if r["delta_k"] == 0 and (r["distance_gap_pct"] or 0) < -TIE_TOL)
    dist_ties = sum(
        1 for r in feas
        if r["delta_k"] == 0 and r["distance_gap_pct"] is not None and abs(r["distance_gap_pct"]) < 0.05
    )
    gaps = [r["distance_gap_pct"] for r in feas if r["delta_k"] == 0 and r["distance_gap_pct"] is not None]
    families: dict[str, dict] = {}
    for r in feas:
        fam = r.get("family") or "?"
        slot = families.setdefault(fam, {"n": 0, "wins": 0, "ties": 0, "losses": 0, "delta_k": []})
        slot["n"] += 1
        slot["delta_k"].append(r["delta_k"])
        if r["delta_k"] < 0:
            slot["wins"] += 1
        elif r["delta_k"] == 0:
            slot["ties"] += 1
        else:
            slot["losses"] += 1
    for slot in families.values():
        slot["mean_delta_k"] = sum(slot["delta_k"]) / len(slot["delta_k"]) if slot["delta_k"] else None
        del slot["delta_k"]

    summary = {
        "n_runs": len(rows),
        "n_instances": len(compared),
        "feasible": len(feas),
        "fleet_wins": wins,
        "fleet_ties": ties,
        "fleet_losses": losses,
        "distance_wins_when_k_tied": dist_wins,
        "near_bks_equal_k_gap_lt_0.05pct": dist_ties,
        "mean_delta_k": (sum(r["delta_k"] for r in feas) / len(feas)) if feas else None,
        "mean_distance_gap_pct_when_k_tied": (sum(gaps) / len(gaps)) if gaps else None,
        "new_best_instances": [r["instance"] for r in feas if r["delta_k"] < 0],
        "families": families,
        "ranking": "feasible, then min K, then min distance",
        "bks_rule": "lex min of literature column and Erdelic ALNS from Erdelic & Caric 2022 Table 3 (EVRPTW-PR)",
        "citation": CITATION,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    best_path = RESULTS / "best_per_instance.csv"
    with best_path.open("w", newline="", encoding="utf-8") as fh:
        fields = list(RESULT_FIELDS)
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(best)

    cmp_fields = [
        "instance", "family", "bks_source_name", "bks_k", "bks_d",
        "our_k", "our_d", "delta_k", "distance_gap_pct", "verdict",
        "seed", "runtime_s", "feasible", "note",
    ]
    cmp_path = RESULTS / "vs_literature.csv"
    with cmp_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cmp_fields, extrasaction="ignore")
        w.writeheader()
        for r in compared:
            w.writerow(r)

    md = [
        "# ALNS-HCO vs published Schneider EVRPTW-PR BKS",
        "",
        "Hanif results are not included. Ranking is feasible, then min $K$, then min distance.",
        f"BKS: {CITATION}",
        "",
        f"Instances: {summary['n_instances']}. Feasible: {summary['feasible']}.",
        f"Fleet wins/ties/losses vs BKS: **{wins}/{ties}/{losses}**.",
        f"Equal-K distance wins: {dist_wins}. Near-BKS (|gap|<0.05% at equal K): {dist_ties}.",
        (
            f"Mean ΔK: {summary['mean_delta_k']:.2f}."
            if summary["mean_delta_k"] is not None else "Mean ΔK: n/a."
        ),
        (
            f"Equal-K mean distance gap: {summary['mean_distance_gap_pct_when_k_tied']:.2f}%."
            if summary["mean_distance_gap_pct_when_k_tied"] is not None
            else "Equal-K mean distance gap: n/a."
        ),
        "",
        "| instance | family | BKS source | BKS K | BKS D | our K | our D | ΔK | equal-K gap % | verdict | seed | s |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for r in compared:
        gap = "" if r["distance_gap_pct"] is None else f"{r['distance_gap_pct']:.2f}"
        md.append(
            f"| {r['instance']} | {r['family']} | {r['bks_source_name']} | {r['bks_k']} | "
            f"{r['bks_d']:.2f} | {r['our_k']} | {r['our_d']:.2f} | {r['delta_k']} | {gap} | "
            f"{r['verdict']} | {r['seed']} | {r['runtime_s']:.1f} |"
        )
    md.append("")
    if families:
        md += ["## By family", "", "| family | n | wins | ties | losses | mean ΔK |", "|---|---:|---:|---:|---:|---:|"]
        for fam in sorted(families):
            s = families[fam]
            md.append(
                f"| {fam} | {s['n']} | {s['wins']} | {s['ties']} | {s['losses']} | {s['mean_delta_k']:.2f} |"
            )
        md.append("")
    (RESULTS / "vs_literature.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
