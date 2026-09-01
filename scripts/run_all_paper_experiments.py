"""Run Hanif paper experiments and compile tables/figures.

  python scripts/run_all_paper_experiments.py --representative --seeds 0
  python scripts/run_all_paper_experiments.py --workers 8
  python scripts/run_all_paper_experiments.py --compile-only
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

from src.alns_hco.experiments import solve_job  # noqa: E402
from src.alns_hco.paper import compile_hanif_paper  # noqa: E402
from src.alns_hco.quality import check_master  # noqa: E402
from src.alns_hco.schneider_paper import compile_schneider_paper  # noqa: E402
from src.common.config import MASTER_FIELDS, METHOD_GI, PAPER_METHODS, SEEDS, experiment_config  # noqa: E402
from src.common.paths import hanif_dirs  # noqa: E402
from src.hanif.loader import REPRESENTATIVE, list_instances  # noqa: E402


def _jobs(args) -> list[dict]:
    names = list(args.instances) if args.instances else list(list_instances() or REPRESENTATIVE)
    if args.representative:
        names = list(REPRESENTATIVE)
    seeds = tuple(int(s) for s in args.seeds)
    tws = ["original", "relaxed"] if args.tw == "both" else [args.tw]
    methods = list(args.methods) if args.methods else list(PAPER_METHODS) + ["ALNS-PR+HCO"]
    jobs = []
    for tw in tws:
        for name in names:
            for method in methods:
                use_seeds = (0,) if method == METHOD_GI else seeds
                for seed in use_seeds:
                    jobs.append({
                        "name": name, "tw": tw, "method": method, "seed": seed,
                        "save_routes": bool(args.save_routes) and seed == seeds[0],
                    })
    return jobs


def _write_master(rows: list[dict], path: Path) -> None:
    fields = [f for f in MASTER_FIELDS]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["tw"], x["instance"], x["method"], x["seed"])):
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", choices=["original", "relaxed", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=list(SEEDS))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--save-routes", action="store_true")
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    dirs = hanif_dirs()
    master_path = dirs["raw"] / "master_results.csv"
    cfg_path = dirs["paper"] / "experiment_config.json"
    cfg_path.write_text(json.dumps(experiment_config(), indent=2) + "\n", encoding="utf-8")

    if not args.compile_only:
        jobs = _jobs(args)
        print(f"=== Hanif paper runs: {len(jobs)} jobs, workers={args.workers} ===", flush=True)
        rows: list[dict] = []
        if args.workers <= 1:
            for job in jobs:
                row = solve_job(job)
                print(
                    f"  [{row['tw']}] {row['instance']} {row['method']} seed={row['seed']} "
                    f"K={row['k']} D={row['distance']:.0f} C={row['charging_expense']:.0f}",
                    flush=True,
                )
                rows.append(row)
        else:
            with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
                futs = {pool.submit(solve_job, job): job for job in jobs}
                for fut in as_completed(futs):
                    row = fut.result()
                    print(
                        f"  [{row['tw']}] {row['instance']} {row['method']} seed={row['seed']} "
                        f"K={row['k']} D={row['distance']:.0f} C={row['charging_expense']:.0f}",
                        flush=True,
                    )
                    rows.append(row)
        _write_master(rows, master_path)
        (dirs["runs"] / "latest_rows.json").write_text(
            json.dumps([{k: v for k, v in r.items() if k != "routes"} for r in rows], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"wrote {master_path}", flush=True)

    errors = check_master(master_path)
    (dirs["diagnostics"] / "quality_checks.txt").write_text(
        "\n".join(errors) if errors else "all checks passed\n", encoding="utf-8",
    )
    if errors:
        print("quality checks:", *errors[:12], sep="\n  ", flush=True)
    compile_hanif_paper(master_path)
    compile_schneider_paper()
    print("paper package written under outputs/hanif/paper and outputs/schneider/paper", flush=True)
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
