"""Run four matched Hanif metaheuristics and write one comparison file.

These are our implementations on the Hanif 1000-customer set. There is no
published numerical table on this dataset.

  python scripts/run_hanif_benchmarks.py --workers 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hanif.benchmarks import HYPERPARAMETERS  # noqa: E402
from src.hanif.benchmarks.alns import solve_alns  # noqa: E402
from src.hanif.benchmarks.greedy import solve_greedy  # noqa: E402
from src.hanif.benchmarks.sa import solve_sa  # noqa: E402
from src.hanif.benchmarks.vns import solve_vns  # noqa: E402
from src.hanif.loader import FAMILIES, REPRESENTATIVE, list_instances, load_hanif  # noqa: E402
from src.hanif.paths import HANIF_OUT  # noqa: E402

TIME_LIMIT = 15.0
SEED = 0

METRIC_FIELDS = (
    "feasible", "served", "unserved", "hard_violations", "duplicates",
    "k", "distance", "duration",
    "charging_visits", "charged_energy", "charging_time", "charging_expense",
    "station_waiting", "customer_waiting",
)

CSV_FIELDS = (
    "instance", "family", "tw", "method",
    *METRIC_FIELDS,
    "runtime_s", "iterations", "accepted",
)


def _family(name: str) -> str:
    return name.split("_")[0]


def _from_metrics(m) -> dict:
    return {
        "feasible": bool(m.feasible),
        "served": int(len(m.customers)),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
        "duplicates": int(getattr(m, "duplicates", 0) or 0),
        "k": int(m.used_k),
        "distance": float(m.distance),
        "duration": float(m.duration),
        "charging_visits": int(m.charging_visits),
        "charged_energy": float(m.charged_energy),
        "charging_time": float(m.charging_time),
        "charging_expense": float(m.charging_expense),
        "station_waiting": float(m.station_waiting),
        "customer_waiting": float(getattr(m, "customer_waiting", 0.0) or 0.0),
    }


def _row(name: str, tw: str, method: str, result: dict) -> dict:
    row = {
        "instance": name,
        "family": _family(name),
        "tw": tw,
        "method": method,
        **_from_metrics(result["metrics"]),
        "runtime_s": float(result["runtime_s"]),
        "iterations": int(result.get("iterations") or 0),
        "accepted": int(result.get("accepted") or 0),
    }
    return row


def _worker(job: dict) -> list[dict]:
    inst = load_hanif(job["name"], job["tw"])
    gi = solve_greedy(inst, deadline=40.0)
    rows = [_row(job["name"], job["tw"], "GI", gi)]
    start = gi["routes"]
    for method, fn in (("ALNS", solve_alns), ("VNS", solve_vns), ("SA", solve_sa)):
        res = fn(inst, start, time_limit_s=job["budget"], seed=SEED)
        rows.append(_row(job["name"], job["tw"], method, res))
    print(
        f"  [{job['tw']}] {job['name']}  "
        + "  ".join(
            f"{r['method']} K={r['k']} feas={int(r['feasible'])} viol={r['hard_violations']}"
            for r in rows
        ),
        flush=True,
    )
    return rows


def _mean(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def _fmt(rows: list[dict], key: str, nd: int = 1) -> str:
    v = _mean(rows, key)
    if v is None:
        return ""
    return f"{v:.{nd}f}"


def _fnum(raw, default=0.0):
    if raw is None or raw == "":
        return default
    return float(raw)


def _carefi_rows(path: Path) -> list[dict]:
    """CARE-FI official row is Stage 2 (K frozen, charging refined)."""
    if not path.is_file():
        return []
    out = []
    with path.open(encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            out.append({
                "instance": raw["instance"],
                "family": raw.get("family") or _family(raw["instance"]),
                "tw": raw["tw"],
                "method": "CARE-FI",
                "feasible": str(raw.get("feasible", "")).lower() in {"true", "1"},
                "served": int(_fnum(raw.get("served"), 1000)),
                "unserved": int(_fnum(raw.get("unserved"))),
                "hard_violations": int(_fnum(raw.get("hard_violations"))),
                "duplicates": 0,
                "k": int(_fnum(raw.get("k_s2") or raw.get("k"))),
                "distance": _fnum(raw.get("distance_s2") or raw.get("distance")),
                "duration": _fnum(raw.get("duration_s2") or raw.get("duration")),
                "charging_visits": int(_fnum(raw.get("charging_visits_s2") or raw.get("charging_visits"))),
                "charged_energy": _fnum(raw.get("charged_energy_s2") or raw.get("charged_energy")),
                "charging_time": _fnum(raw.get("charging_time_s2") or raw.get("charging_time")),
                "charging_expense": _fnum(raw.get("charging_expense_s2") or raw.get("charging_expense")),
                "station_waiting": _fnum(raw.get("station_waiting_s2") or raw.get("station_waiting")),
                "customer_waiting": _fnum(raw.get("customer_waiting_s2") or raw.get("customer_waiting")),
                "runtime_s": _fnum(raw.get("runtime_s")),
                "iterations": 0,
                "accepted": 0,
                # Stage-1 (full recharge) kept for the charge-policy note.
                "k_s1": int(_fnum(raw.get("k"))),
                "distance_s1": _fnum(raw.get("distance")),
                "duration_s1": _fnum(raw.get("duration")),
                "charging_visits_s1": int(_fnum(raw.get("charging_visits"))),
                "charged_energy_s1": _fnum(raw.get("charged_energy")),
                "charging_time_s1": _fnum(raw.get("charging_time")),
                "charging_expense_s1": _fnum(raw.get("charging_expense")),
                "station_waiting_s1": _fnum(raw.get("station_waiting")),
                "customer_waiting_s1": _fnum(raw.get("customer_waiting")),
                "phi": _fnum(raw.get("phi")),
            })
    return out


def _feas_line(rows: list[dict]) -> str:
    n = len(rows)
    feas = sum(1 for r in rows if r["feasible"])
    unserved = sum(int(r.get("unserved") or 0) for r in rows)
    hard = sum(int(r.get("hard_violations") or 0) for r in rows)
    dup = sum(int(r.get("duplicates") or 0) for r in rows)
    return f"{feas}/{n}", unserved, hard, dup


def _md(all_rows: list[dict]) -> str:
    methods = ["CARE-FI", "GI", "ALNS", "VNS", "SA"]
    lines = [
        "# Hanif metaheuristic benchmark",
        "",
        "No published solver has a numerical table on this 1000-customer Hanif set.",
        "GI, ALNS, VNS, and SA are **our implementations** on the same instances, replay, and vehicle data as CARE-FI.",
        "They are not reproductions of Schneider VNS–TS or Keskin ALNS.",
        "",
        f"Budget: {TIME_LIMIT:.0f} s per search method after a shared greedy start. Seed: {SEED}.",
        "",
        "## Feasibility",
        "",
        "Every accepted solution is checked by the same sequential replay as CARE-FI (`src/hanif/replay.py`).",
        "A solution is feasible only if **all** of the following hold:",
        "",
        "- every customer is served **exactly once** (no unserved, no duplicates)",
        "- battery never goes negative on any arc",
        "- every customer start is inside `[a_i, b_i]`",
        "- load never exceeds vehicle capacity",
        "- return to the depot is inside the planning horizon",
        "- charging labels are a portion in `[0, 1]`",
        "",
        "Moves that fail replay are discarded. GI, VNS, and SA never keep an infeasible incumbent.",
        "ALNS discards any repair with unserved customers or hard violations; simulated-annealing steps",
        "may accept a **worse feasible** distance, never a constraint violation.",
        "These methods are **not exact**: they can miss a better feasible solution, but they do not",
        "report a solution that violates the hard constraints above.",
        "",
        "GI / ALNS / VNS / SA use **full recharge** when a station is inserted.",
        "CARE-FI numbers below are **Stage 2** (customer sequences frozen, charge portions refined).",
        "CARE-FI Stage 1 uses the same full-recharge policy as the four methods; those charging",
        "figures are in the Stage-1 note after each TW block.",
        "",
    ]
    for tw, label in (("original", "Original TW"), ("relaxed", "Relaxed TW")):
        sub = [r for r in all_rows if r["tw"] == tw]
        lines += [
            f"## {label}",
            "",
            "| Method | n | Feasible | Unserved | Hard viol. | Mean K | Mean distance | Mean duration | Mean ch. visits | Mean charged energy | Mean ch. time | Mean ch. cost | Mean station wait | Mean customer wait | Mean runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in methods:
            mrows = [r for r in sub if r["method"] == method]
            if not mrows:
                continue
            feas, unserved, hard, _dup = _feas_line(mrows)
            lines.append(
                f"| {method} | {len(mrows)} | {feas} | {unserved} | {hard} | "
                f"{_fmt(mrows, 'k', 2)} | {_fmt(mrows, 'distance', 1)} | {_fmt(mrows, 'duration', 1)} | "
                f"{_fmt(mrows, 'charging_visits', 1)} | {_fmt(mrows, 'charged_energy', 1)} | "
                f"{_fmt(mrows, 'charging_time', 1)} | {_fmt(mrows, 'charging_expense', 1)} | "
                f"{_fmt(mrows, 'station_waiting', 1)} | {_fmt(mrows, 'customer_waiting', 1)} | "
                f"{_fmt(mrows, 'runtime_s', 2)} |"
            )
        care = [r for r in sub if r["method"] == "CARE-FI" and r.get("charging_time_s1") is not None]
        if care:
            lines += [
                "",
                "CARE-FI Stage 1 (full recharge, same charge policy as GI/ALNS/VNS/SA) vs Stage 2 (table above):",
                "",
                f"- mean charging time { _fmt(care, 'charging_time_s1', 1) } → { _fmt(care, 'charging_time', 1) }",
                f"- mean charging cost { _fmt(care, 'charging_expense_s1', 1) } → { _fmt(care, 'charging_expense', 1) }",
                f"- mean charged energy { _fmt(care, 'charged_energy_s1', 1) } → { _fmt(care, 'charged_energy', 1) }",
                f"- mean duration { _fmt(care, 'duration_s1', 1) } → { _fmt(care, 'duration', 1) }",
                f"- mean distance { _fmt(care, 'distance_s1', 1) } → { _fmt(care, 'distance', 1) } (K frozen at { _fmt(care, 'k', 2) })",
                "",
            ]
        else:
            lines.append("")
        lines += [
            "### Mean K by family",
            "",
            "| Method | " + " | ".join(FAMILIES) + " |",
            "|---|" + "|".join("---:" for _ in FAMILIES) + "|",
        ]
        for method in methods:
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
        for method in methods:
            cells = [method]
            for fam in FAMILIES:
                mrows = [r for r in sub if r["method"] == method and r["family"] == fam]
                cells.append(_fmt(mrows, "charging_expense", 1) if mrows else "")
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    lines += ["## Hyperparameters", ""]
    for key in ("GI", "ALNS", "VNS", "SA"):
        hp = HYPERPARAMETERS[key]
        lines.append(f"### {hp['name']} ({hp['abbreviation']})")
        lines.append("")
        for k, v in hp.items():
            if k in {"name", "abbreviation"}:
                continue
            lines.append(f"- `{k}`: `{v}`")
        lines.append("")
    lines += [
        "CARE-FI uses the two-stage method in `src/hanif/` (due-time insertion, most-constrained-first",
        "route elimination, then station/portion refinement). See `METHOD.md`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", choices=["original", "relaxed", "both"], default="both")
    ap.add_argument("--budget", type=float, default=TIME_LIMIT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    args = ap.parse_args()

    names = list(args.instances) if args.instances else list(list_instances() or REPRESENTATIVE)
    if args.representative:
        names = list(REPRESENTATIVE)
    tws = ["original", "relaxed"] if args.tw == "both" else [args.tw]
    jobs = [{"name": n, "tw": tw, "budget": args.budget} for tw in tws for n in names]

    print(f"=== Hanif baselines ({len(jobs)} instances, 4 methods, budget={args.budget}s) ===", flush=True)
    rows: list[dict] = []
    if args.workers <= 1:
        for job in jobs:
            rows.extend(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futs = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                rows.extend(fut.result())
    rows.sort(key=lambda r: (r["tw"], r["instance"], r["method"]))

    care = _carefi_rows(HANIF_OUT / "results.csv")
    all_rows = care + rows
    out_md = HANIF_OUT / "benchmarks.md"
    out_csv = HANIF_OUT / "benchmarks.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    out_md.write_text(_md(all_rows), encoding="utf-8")
    print(f"wrote {out_md}", flush=True)
    print(f"wrote {out_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
