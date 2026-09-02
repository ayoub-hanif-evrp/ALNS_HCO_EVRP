"""Run ALNS-HCO on the 56 large Schneider EVRPTW instances vs literature BKS.

Everything for this experiment lives in ``schneider_alns_hco/``.

  python schneider_alns_hco/run.py --quick
  python schneider_alns_hco/run.py --workers 8
  python schneider_alns_hco/run.py --compile-only
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from compare import compile_comparison
from config import (
    LARGE_56, PHASE1_S, PHASE2_S, QUICK_SIX, RESULT_FIELDS, RESULTS, ROUTES,
    SEEDS, TOTAL_BUDGET_S,
)
from job import solve_job
from load import list_instances


def _jobs(args) -> list[dict]:
    names = list(args.instances) if args.instances else list(list_instances() or LARGE_56)
    if args.quick:
        names = [n for n in QUICK_SIX if n in names] or list(QUICK_SIX)
    seeds = tuple(int(s) for s in args.seeds)
    p1 = float(args.phase1_s)
    p2 = float(args.phase2_s)
    jobs = []
    for name in names:
        for seed in seeds:
            jobs.append({
                "name": name,
                "seed": seed,
                "phase1_s": p1,
                "phase2_s": p2,
                "save_routes": bool(args.save_routes) and seed == seeds[0],
            })
    return jobs


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(RESULT_FIELDS), extrasaction="ignore")
        w.writeheader()
        for row in sorted(rows, key=lambda r: (r["instance"], int(r["seed"]))):
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser(description="ALNS-HCO on Schneider EVRPTW vs literature BKS")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="six representative instances only")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=list(SEEDS))
    ap.add_argument("--phase1-s", dest="phase1_s", type=float, default=PHASE1_S)
    ap.add_argument("--phase2-s", dest="phase2_s", type=float, default=PHASE2_S)
    ap.add_argument("--save-routes", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    ROUTES.mkdir(parents=True, exist_ok=True)
    cfg = {
        "experiment": "ALNS-HCO vs published Schneider EVRPTW-PR BKS",
        "method": "ALNS-HCO (src/alns-hco, unchanged)",
        "benchmark": "Schneider et al. (2014) 56 large instances, 100 customers, 21 stations",
        "recharge": "partial (just-enough); compared to EVRPTW-PR BKS",
        "ranking": "feasible, then min K, then min distance",
        "n_instances": 56 if not args.quick else 6,
        "seeds": [int(s) for s in args.seeds],
        "phase1_s": float(args.phase1_s),
        "phase2_s": float(args.phase2_s),
        "total_s": float(args.phase1_s) + float(args.phase2_s),
        "default_total_s": TOTAL_BUDGET_S,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_folder": str(RESULTS),
    }
    (RESULTS / "experiment_config.json").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    if not args.compile_only:
        jobs = _jobs(args)
        print(
            f"=== Schneider ALNS-HCO: {len(jobs)} jobs, workers={args.workers}, "
            f"budget={args.phase1_s:.0f}+{args.phase2_s:.0f}s ===",
            flush=True,
        )
        rows: list[dict] = []
        if args.workers <= 1:
            for job in jobs:
                row = solve_job(job)
                print(
                    f"  {row['instance']} seed={row['seed']} feas={row['feasible']} "
                    f"K={row['k']} D={row['distance']:.2f} t={row['runtime_s']:.1f}s",
                    flush=True,
                )
                rows.append(row)
        else:
            with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
                futs = {pool.submit(solve_job, job): job for job in jobs}
                for fut in as_completed(futs):
                    row = fut.result()
                    print(
                        f"  {row['instance']} seed={row['seed']} feas={row['feasible']} "
                        f"K={row['k']} D={row['distance']:.2f} t={row['runtime_s']:.1f}s",
                        flush=True,
                    )
                    rows.append(row)
        _write_csv(RESULTS / "results.csv", rows)
        print(f"wrote {RESULTS / 'results.csv'}", flush=True)

    summary = compile_comparison(RESULTS / "results.csv")
    print(
        f"vs BKS fleet W/T/L = {summary['fleet_wins']}/{summary['fleet_ties']}/{summary['fleet_losses']} "
        f"({summary['n_instances']} instances)",
        flush=True,
    )
    print(f"tables: {RESULTS / 'vs_literature.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
