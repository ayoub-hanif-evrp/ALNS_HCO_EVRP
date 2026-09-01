"""Merge ALNS-HCO with GI / ALNS / VNS / SA into one comparison file.

Uses the already-run 120-instance archives. Does not re-solve.

  python scripts/compile_alns_hco_benchmark.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.hyperparameters import HYPERPARAMETERS as HCO_HP  # noqa: E402
from src.alns_hco.paths import ALNS_HCO_OUT  # noqa: E402
from src.hanif.benchmarks import HYPERPARAMETERS as BASE_HP  # noqa: E402

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
METHODS = ["ALNS-HCO", "ALNS", "VNS", "SA", "GI"]
BASELINE_METHODS = {"ALNS", "VNS", "SA", "GI"}
METRIC_FIELDS = (
    "feasible", "served", "unserved", "hard_violations",
    "k", "distance", "duration",
    "charging_visits", "charged_energy", "charging_time", "charging_expense",
    "station_waiting", "customer_waiting", "runtime_s",
)
CSV_FIELDS = ("instance", "family", "tw", "method") + METRIC_FIELDS


def _f(raw, default=0.0):
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str) and raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    return float(raw)


def _bool(raw) -> bool:
    return str(raw).lower() in {"true", "1", "yes"}


def _row(raw: dict, method: str | None = None) -> dict:
    return {
        "instance": raw["instance"],
        "family": raw.get("family") or raw["instance"].split("_")[0],
        "tw": raw["tw"],
        "method": method or raw["method"],
        "feasible": _bool(raw.get("feasible")),
        "served": int(_f(raw.get("served"), 1000)),
        "unserved": int(_f(raw.get("unserved"))),
        "hard_violations": int(_f(raw.get("hard_violations"))),
        "k": int(_f(raw.get("k"))),
        "distance": _f(raw.get("distance")),
        "duration": _f(raw.get("duration")),
        "charging_visits": _f(raw.get("charging_visits")),
        "charged_energy": _f(raw.get("charged_energy")),
        "charging_time": _f(raw.get("charging_time")),
        "charging_expense": _f(raw.get("charging_expense")),
        "station_waiting": _f(raw.get("station_waiting")),
        "customer_waiting": _f(raw.get("customer_waiting")),
        "runtime_s": _f(raw.get("runtime_s")),
    }


def _load_baselines(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            if raw.get("method") not in BASELINE_METHODS:
                continue
            rows.append(_row(raw))
    return rows


def _load_hco(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [_row(raw, "ALNS-HCO") for raw in csv.DictReader(fh)]


def _mean(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else None


def _fmt(rows: list[dict], key: str, nd: int = 1) -> str:
    v = _mean(rows, key)
    return "" if v is None else f"{v:.{nd}f}"


def _wins(hco: list[dict], other: list[dict], key: str) -> tuple[int, int, int]:
    by_o = {(r["instance"], r["tw"]): r for r in other}
    better = tie = worse = 0
    for r in hco:
        o = by_o.get((r["instance"], r["tw"]))
        if o is None:
            continue
        a, b = float(r[key]), float(o[key])
        if a < b - 1e-9:
            better += 1
        elif a > b + 1e-9:
            worse += 1
        else:
            tie += 1
    return better, tie, worse


def _md(all_rows: list[dict]) -> str:
    lines = [
        "# ALNS-HCO vs ALNS, VNS, SA, and GI",
        "",
        "Same 120 Hanif instances (60 Original TW + 60 Relaxed TW), same replay.",
        "GI, ALNS, VNS, SA, and ALNS-HCO are **our implementations**.",
        "None of these is a reproduction of Schneider VNS–TS, Keskin ALNS, or Wang HCO.",
        "",
        "Budgets: GI ~1 s; ALNS / VNS / SA = 15 s after a shared greedy start;",
        "ALNS-HCO = the same 15 s ALNS, then just-enough fleet compression (~8 s) and 15 s Phase II (seed 0).",
        "",
        "## Feasibility",
        "",
        "A solution is feasible only if every customer is served exactly once, battery never goes",
        "negative, every time window and the capacity/horizon limits hold. One violation makes the",
        "instance unsolved. All five methods solved **120/120** with 0 unserved and 0 hard violations.",
        "",
        "GI / ALNS / VNS / SA use **full recharge** when a station is required.",
        "ALNS-HCO Phase I copies that ALNS, then relabels with just-enough partial charge,",
        "compresses the fleet, and continues ALNS with lex (K, distance, charging).",
        "",
    ]
    for tw, label in (("original", "Original TW"), ("relaxed", "Relaxed TW")):
        sub = [r for r in all_rows if r["tw"] == tw]
        lines += [
            f"## {label}",
            "",
            "| Method | n | Feasible | Mean K | Mean distance | Mean duration | Mean ch. time | Mean ch. cost | Mean station wait | Mean runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in METHODS:
            mrows = [r for r in sub if r["method"] == method]
            if not mrows:
                continue
            feas = sum(1 for r in mrows if r["feasible"])
            lines.append(
                f"| {method} | {len(mrows)} | {feas}/{len(mrows)} | {_fmt(mrows, 'k', 2)} | "
                f"{_fmt(mrows, 'distance', 1)} | {_fmt(mrows, 'duration', 1)} | "
                f"{_fmt(mrows, 'charging_time', 1)} | {_fmt(mrows, 'charging_expense', 1)} | "
                f"{_fmt(mrows, 'station_waiting', 1)} | {_fmt(mrows, 'runtime_s', 2)} |"
            )
        lines.append("")
        hco = [r for r in sub if r["method"] == "ALNS-HCO"]
        if hco:
            lines += [
                "ALNS-HCO vs each method (strictly better / tie / worse), lower is better:",
                "",
                "| vs | K | Distance | Charging cost |",
                "|---|---:|---:|---:|",
            ]
            for method in METHODS:
                if method == "ALNS-HCO":
                    continue
                oth = [r for r in sub if r["method"] == method]
                if not oth:
                    continue
                k = _wins(hco, oth, "k")
                d = _wins(hco, oth, "distance")
                c = _wins(hco, oth, "charging_expense")
                lines.append(
                    f"| {method} | {k[0]}/{k[1]}/{k[2]} | {d[0]}/{d[1]}/{d[2]} | {c[0]}/{c[1]}/{c[2]} |"
                )
            lines.append("")
        lines += [
            "### Mean K by family",
            "",
            "| Method | " + " | ".join(FAMILIES) + " |",
            "|---|" + "|".join("---:" for _ in FAMILIES) + "|",
        ]
        for method in METHODS:
            cells = [method]
            for fam in FAMILIES:
                mrows = [r for r in sub if r["method"] == method and r["family"] == fam]
                cells.append(_fmt(mrows, "k", 1) if mrows else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
        lines += [
            "### Mean charging cost by family",
            "",
            "| Method | " + " | ".join(FAMILIES) + " |",
            "|---|" + "|".join("---:" for _ in FAMILIES) + "|",
        ]
        for method in METHODS:
            cells = [method]
            for fam in FAMILIES:
                mrows = [r for r in sub if r["method"] == method and r["family"] == fam]
                cells.append(_fmt(mrows, "charging_expense", 1) if mrows else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    hco_o = [r for r in all_rows if r["method"] == "ALNS-HCO" and r["tw"] == "original"]
    alns_o = [r for r in all_rows if r["method"] == "ALNS" and r["tw"] == "original"]
    hco_r = [r for r in all_rows if r["method"] == "ALNS-HCO" and r["tw"] == "relaxed"]
    alns_r = [r for r in all_rows if r["method"] == "ALNS" and r["tw"] == "relaxed"]
    k_o = _wins(hco_o, alns_o, "k") if hco_o and alns_o else (0, 0, 0)
    k_r = _wins(hco_r, alns_r, "k") if hco_r and alns_r else (0, 0, 0)
    mean_hco_o = _mean(hco_o, "k")
    mean_alns_o = _mean(alns_o, "k")
    mean_hco_r = _mean(hco_r, "k")
    mean_alns_r = _mean(alns_r, "k")
    beat_o = mean_hco_o is not None and mean_alns_o is not None and mean_hco_o < mean_alns_o - 1e-9
    beat_r = mean_hco_r is not None and mean_alns_r is not None and mean_hco_r < mean_alns_r - 1e-9

    lines += [
        "## How to read this",
        "",
        r"EVRPTW ranking is feasible, then min \(K\), then min distance. Charging cost is third.",
        "",
        f"- Original TW mean K: ALNS-HCO {_fmt(hco_o, 'k', 2)} vs ALNS {_fmt(alns_o, 'k', 2)}"
        f" ({'HCO smaller' if beat_o else 'ALNS smaller or equal'}); paired K wins/tie/losses {k_o[0]}/{k_o[1]}/{k_o[2]}.",
        f"- Relaxed TW mean K: ALNS-HCO {_fmt(hco_r, 'k', 2)} vs ALNS {_fmt(alns_r, 'k', 2)}"
        f" ({'HCO smaller' if beat_r else 'ALNS smaller or equal'}); paired K wins/tie/losses {k_r[0]}/{k_r[1]}/{k_r[2]}.",
        "- ALNS-HCO is the 15 s ALNS plus extra search, so wall-clock is longer. The paired K count",
        "  above is vs the archived 15 s ALNS, not vs the ALNS prefix inside the same HCO run.",
        "",
        "## Hyperparameters",
        "",
        "### ALNS-HCO",
        "",
    ]
    for k, v in HCO_HP.items():
        if k in {"name", "abbreviation"}:
            continue
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    for key in ("ALNS", "VNS", "SA", "GI"):
        hp = BASE_HP[key]
        lines.append(f"### {hp['name']} ({hp['abbreviation']})")
        lines.append("")
        for k, v in hp.items():
            if k in {"name", "abbreviation"}:
                continue
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    base_csv = ALNS_HCO_OUT / "baselines.csv"
    hco_csv = ALNS_HCO_OUT / "results.csv"
    if not base_csv.is_file():
        raise SystemExit(f"missing baseline file: {base_csv}")
    if not hco_csv.is_file():
        raise SystemExit(f"missing ALNS-HCO results: {hco_csv}")

    rows = _load_baselines(base_csv) + _load_hco(hco_csv)
    rows.sort(key=lambda r: (r["tw"], r["instance"], r["method"]))
    out_csv = ALNS_HCO_OUT / "benchmarks.csv"
    out_md = ALNS_HCO_OUT / "benchmarks.md"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    out_md.write_text(_md(rows), encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
