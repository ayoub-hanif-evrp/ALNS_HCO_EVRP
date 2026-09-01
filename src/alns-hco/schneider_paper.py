"""Compile Schneider external-benchmark table from existing PR/FR archives."""
from __future__ import annotations

import csv
from pathlib import Path

from src.common.paths import ROOT, schneider_dirs
from src.schneider.evaluation.schneider_bks import bks_table


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(raw, default=None):
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def compile_schneider_paper() -> Path | None:
    dirs = schneider_dirs()
    pr = _load(ROOT / "outputs" / "schneider" / "pr" / "results.csv")
    if not pr:
        pr = _load(dirs["root"] / "pr" / "results.csv")
    if not pr:
        note = dirs["paper"] / "table_schneider_benchmark.md"
        note.write_text(
            "# Schneider benchmark\n\nNo `outputs/schneider/pr/results.csv` found.\n"
            "Published BKS live in `src/schneider/evaluation/schneider_bks.py`.\n",
            encoding="utf-8",
        )
        return note

    bks = bks_table("pr")
    out_rows = []
    for raw in pr:
        name = raw.get("instance")
        ref = bks.get(name) or {}
        our_k = int(_f(raw.get("used_k") or raw.get("k"), 0) or 0)
        our_d = _f(raw.get("distance"), 0.0) or 0.0
        pub_k = int(_f(raw.get("bks_k") or ref.get("bks_k"), 0) or 0)
        pub_d = _f(raw.get("bks_d") or ref.get("bks_d"))
        src = raw.get("bks_source") or ref.get("bks_source_name") or ref.get("bks_source") or ""
        dk = our_k - pub_k if pub_k else None
        dgap = None
        if dk == 0 and pub_d:
            dgap = 100.0 * (our_d - pub_d) / pub_d
        out_rows.append({
            "instance": name,
            "published_source": src,
            "published_k": pub_k,
            "published_distance": pub_d,
            "our_k": our_k,
            "our_distance": our_d,
            "delta_k": dk,
            "distance_gap_pct_equal_k": dgap,
            "runtime_s": _f(raw.get("runtime_s")),
            "feasible": raw.get("feasible"),
        })

    n = len(out_rows)
    wins = sum(1 for r in out_rows if r["delta_k"] is not None and r["delta_k"] < 0)
    ties = sum(1 for r in out_rows if r["delta_k"] == 0)
    losses = sum(1 for r in out_rows if r["delta_k"] is not None and r["delta_k"] > 0)
    equal = [r for r in out_rows if r["delta_k"] == 0 and r["distance_gap_pct_equal_k"] is not None]
    mean_gap = sum(r["distance_gap_pct_equal_k"] for r in equal) / len(equal) if equal else None
    matches = sum(
        1 for r in out_rows
        if r["delta_k"] == 0 and r["distance_gap_pct_equal_k"] is not None
        and abs(r["distance_gap_pct_equal_k"]) < 0.05
    )

    fields = list(out_rows[0].keys())
    csv_path = dirs["paper"] / "table_schneider_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    md = [
        "# Schneider EVRPTW external benchmark",
        "",
        "Hanif results are not included. Published columns come from the repository BKS file",
        "(`src/schneider/evaluation/schneider_bks.py`, Erdelić & Carić 2022 compilation).",
        "Our columns are the archived Schneider PR run in `outputs/schneider/pr/`.",
        "",
        f"Instances: {n}. Fleet wins/ties/losses: {wins}/{ties}/{losses}.",
        f"Equal-K mean distance gap: {mean_gap:.2f}%" if mean_gap is not None else "Equal-K mean distance gap: n/a",
        f"Near-BKS matches (|gap|<0.05% at equal K): {matches}. New best (ΔK<0): {wins}.",
        "",
        "| instance | source | pub K | pub D | our K | our D | ΔK | equal-K gap % | runtime s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in out_rows:
        gap = "" if r["distance_gap_pct_equal_k"] is None else f"{r['distance_gap_pct_equal_k']:.2f}"
        pub_d = "" if r["published_distance"] is None else f"{r['published_distance']:.2f}"
        md.append(
            f"| {r['instance']} | {r['published_source']} | {r['published_k']} | {pub_d} | "
            f"{r['our_k']} | {r['our_distance']:.2f} | {r['delta_k']} | {gap} | "
            f"{'' if r['runtime_s'] is None else f'{r['runtime_s']:.1f}'} |"
        )
    md.append("")
    (dirs["paper"] / "table_schneider_benchmark.md").write_text("\n".join(md), encoding="utf-8")

    tex = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"instance & source & pub $K$ & pub $D$ & our $K$ & our $D$ & $\Delta K$ & gap \% & s \\",
        r"\midrule",
    ]
    for r in out_rows[:12]:
        gap = "--" if r["distance_gap_pct_equal_k"] is None else f"{r['distance_gap_pct_equal_k']:.2f}"
        pub_d = "--" if r["published_distance"] is None else f"{r['published_distance']:.2f}"
        tex.append(
            f"{r['instance']} & {r['published_source']} & {r['published_k']} & {pub_d} & "
            f"{r['our_k']} & {r['our_distance']:.2f} & {r['delta_k']} & {gap} & "
            f"{0 if r['runtime_s'] is None else r['runtime_s']:.1f} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    (dirs["paper"] / "table_schneider_benchmark.tex").write_text("\n".join(tex), encoding="utf-8")
    return csv_path
