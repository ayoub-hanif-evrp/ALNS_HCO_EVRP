"""Run ALNS-HCO on the Hanif 1000-customer set.

  python scripts/run_alns_hco.py --workers 8 --representative
  python scripts/run_alns_hco.py --workers 8 --tw both
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.hyperparameters import HYPERPARAMETERS  # noqa: E402
from src.alns_hco.paths import ALNS_HCO_OUT  # noqa: E402
from src.alns_hco.solver import solve_alns_hco  # noqa: E402
from src.hanif.loader import FAMILIES, REPRESENTATIVE, list_instances, load_hanif  # noqa: E402

CSV_FIELDS = [
    "instance", "family", "tw", "mode", "feasible", "served", "unserved", "hard_violations",
    "k", "distance", "duration", "charging_visits", "charged_energy",
    "charging_time", "charging_expense", "station_waiting", "customer_waiting",
    "k_s1", "distance_s1", "duration_s1", "charging_expense_s1", "charging_time_s1",
    "alns_k", "eliminations",
    "runtime_s", "phase1_s", "phase2_s", "iterations_p1", "iterations_p2",
    "k_ok", "distance_ok", "duration_ok",
]


def _family(name: str) -> str:
    return name.split("_")[0]


def _md(m) -> dict:
    return {
        "feasible": bool(m.feasible),
        "served": int(len(m.customers)),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
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


def _worker(job: dict) -> dict:
    inst = load_hanif(job["name"], job["tw"])
    result = solve_alns_hco(
        inst,
        mode=job["mode"],
        phase1_s=job["phase1"],
        phase2_s=job["phase2"],
        seed=job["seed"],
    )
    a = result.stages.get("phase1") or _md(result.metrics)
    b = _md(result.metrics)
    row = {
        "instance": job["name"],
        "family": _family(job["name"]),
        "tw": job["tw"],
        "mode": result.mode,
        **b,
        "k_s1": a.get("k"),
        "distance_s1": a.get("distance"),
        "duration_s1": a.get("duration"),
        "charging_expense_s1": a.get("charging_expense"),
        "charging_time_s1": a.get("charging_time"),
        "alns_k": a.get("alns_k"),
        "eliminations": a.get("eliminations"),
        "runtime_s": float(result.runtime_s),
        "phase1_s": float(a.get("runtime_s") or 0.0),
        "phase2_s": float((result.stages.get("phase2") or {}).get("runtime_s") or 0.0),
        "iterations_p1": int(a.get("iterations") or 0),
        "iterations_p2": int((result.stages.get("phase2") or {}).get("iterations") or 0),
        "k_ok": bool(result.validation.get("k_ok", True)),
        "distance_ok": bool(result.validation.get("distance_ok", True)),
        "duration_ok": bool(result.validation.get("duration_ok", True)),
    }
    print(
        f"  [{job['tw']}] {job['name']} {result.mode} feas={row['feasible']} "
        f"K={row['k']} D={row['distance']:.0f} C={row['charging_expense']:.0f} t={row['runtime_s']:.1f}s",
        flush=True,
    )
    return row


def _mean(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows if r.get(key) not in (None, "")]
    return sum(vals) / len(vals) if vals else None


def _summary(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), "feasible": sum(1 for r in rows if r["feasible"]), "hyperparameters": HYPERPARAMETERS}
    for tw in ("original", "relaxed"):
        sub = [r for r in rows if r["tw"] == tw]
        if not sub:
            continue
        out[tw] = {
            "n": len(sub),
            "feasible": sum(1 for r in sub if r["feasible"]),
            "mean_k": _mean(sub, "k"),
            "mean_distance": _mean(sub, "distance"),
            "mean_duration": _mean(sub, "duration"),
            "mean_charging_expense": _mean(sub, "charging_expense"),
            "mean_charging_time": _mean(sub, "charging_time"),
            "mean_station_waiting": _mean(sub, "station_waiting"),
            "mean_k_s1": _mean(sub, "k_s1"),
            "mean_alns_k": _mean(sub, "alns_k"),
            "mean_distance_s1": _mean(sub, "distance_s1"),
            "mean_charging_expense_s1": _mean(sub, "charging_expense_s1"),
            "mean_runtime_s": _mean(sub, "runtime_s"),
            "k_wins_vs_own_alns": sum(
                1 for r in sub
                if r.get("alns_k") not in (None, "") and int(r["k"]) < int(r["alns_k"])
            ),
            "by_family": {
                fam: {"n": sum(1 for r in sub if r["family"] == fam), "mean_k": _mean([r for r in sub if r["family"] == fam], "k")}
                for fam in FAMILIES
            },
        }
    return out


def _readme(summary: dict) -> str:
    lines = [
        "# ALNS-HCO results",
        "",
        "Adaptive Large Neighborhood Search with Heterogeneous Charging Oracle.",
        "Phase I is the same GI + routing ALNS as the 15 s ALNS baseline, then just-enough",
        "fleet compression. Phase II keeps searching with lex (K, distance, charging);",
        "the fleet may decrease and may not increase.",
        "",
        f"- Instances: {summary.get('n', 0)}",
        f"- Feasible: {summary.get('feasible', 0)}/{summary.get('n', 0)}",
        "",
    ]
    for tw, label in (("original", "Original TW"), ("relaxed", "Relaxed TW")):
        block = summary.get(tw)
        if not block:
            continue
        lines += [
            f"## {label}",
            "",
            f"- Mean K (own ALNS → HCO): {block.get('mean_alns_k'):.2f} → {block.get('mean_k'):.2f}" if block.get("mean_k") is not None else "",
            f"- Instances with K < own ALNS: {block.get('k_wins_vs_own_alns')}/{block.get('n')}" if block.get("n") else "",
            f"- Mean K after Phase I compress: {block.get('mean_k_s1'):.2f}" if block.get("mean_k_s1") is not None else "",
            f"- Mean distance: {block.get('mean_distance_s1'):.1f} → {block.get('mean_distance'):.1f}" if block.get("mean_distance") is not None else "",
            f"- Mean charging cost: {block.get('mean_charging_expense_s1'):.1f} → {block.get('mean_charging_expense'):.1f}" if block.get("mean_charging_expense") is not None else "",
            f"- Mean charging time: {block.get('mean_charging_time'):.1f}" if block.get("mean_charging_time") is not None else "",
            f"- Mean runtime (s): {block.get('mean_runtime_s'):.2f}" if block.get("mean_runtime_s") is not None else "",
            "",
        ]
    lines += [
        "Files: `results.csv`, `summary.json`. Hyperparameters are in `summary.json`.",
        "",
        "This is our implementation. It is not a reproduction of Keskin, Wang, or Kullman.",
        "",
    ]
    return "\n".join(x for x in lines if x is not None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", choices=["original", "relaxed", "both"], default="both")
    ap.add_argument("--mode", choices=["alns", "alns_hco_post", "alns_hco_fast", "alns_hco"], default="alns_hco")
    ap.add_argument("--phase1", type=float, default=HYPERPARAMETERS["phase1_time_limit_s"])
    ap.add_argument("--phase2", type=float, default=HYPERPARAMETERS["phase2_time_limit_s"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--out", default=str(ALNS_HCO_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    names = list(args.instances) if args.instances else list(list_instances() or REPRESENTATIVE)
    if args.representative:
        names = list(REPRESENTATIVE)
    tws = ["original", "relaxed"] if args.tw == "both" else [args.tw]
    jobs = [
        {"name": n, "tw": tw, "mode": args.mode, "phase1": args.phase1, "phase2": args.phase2, "seed": args.seed}
        for tw in tws for n in names
    ]
    print(f"=== ALNS-HCO ({len(jobs)} jobs, mode={args.mode}, p1={args.phase1}s p2={args.phase2}s) ===", flush=True)

    rows: list[dict] = []
    if args.workers <= 1:
        for job in jobs:
            rows.append(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futs = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())
    rows.sort(key=lambda r: (r["tw"], r["instance"]))

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = _summary(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "README.md").write_text(_readme(summary), encoding="utf-8")
    print(f"wrote {out / 'results.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
