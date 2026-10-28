"""Absolute ALNS-HCO results for reviewer revision.

Compares ALNS-PR (30 s), PhaseI-PR-15, PhaseII-NoHCO (15+15), PhaseII-Heterogeneous (15+15)
using identical stored Phase-I solutions for matched Phase-II branches.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import (  # noqa: E402
    aggregate_means,
    paired_instance_effects,
    reviewer_root,
    run_absolute_hco_job,
    select_instances,
    summarize_paired,
    write_csv,
    write_json,
)
from src.common.config import (  # noqa: E402
    METHOD_ALNS_PR,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_HET,
    METHOD_PHASE2_NOHCO,
    SEEDS,
    TW_VARIANTS,
)
from src.common.reporting_io import assert_reporting_fields_present, write_master_csv  # noqa: E402

METHODS = (METHOD_ALNS_PR, METHOD_PHASE1_PR15, METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tw", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    ap.add_argument("--skip-alns-pr", action="store_true", help="Skip expensive ALNS-PR 30 s jobs")
    args = ap.parse_args()

    pilot = bool(args.pilot) or not args.full
    instances = select_instances(pilot=pilot, names=args.instances)
    seeds = tuple(args.seeds) if args.seeds is not None else ((0,) if pilot else SEEDS)
    tws = tuple(args.tw) if args.tw is not None else (("original",) if pilot else TW_VARIANTS)
    methods = tuple(args.methods) if args.methods else METHODS
    if args.skip_alns_pr:
        methods = tuple(m for m in methods if m != METHOD_ALNS_PR)

    jobs = []
    for tw in tws:
        for name in instances:
            for method in methods:
                for seed in seeds:
                    jobs.append({
                        "instance_name": name, "tw": tw, "method": method, "seed": int(seed),
                    })
    if args.max_jobs is not None:
        jobs = jobs[: int(args.max_jobs)]

    out_dir = reviewer_root() / "alns_hco_absolute"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / ("pilot_runs.csv" if pilot else "full_runs.csv")
    status = {
        "mode": "pilot" if pilot else "full",
        "n_jobs_planned": len(jobs),
        "instances": instances,
        "seeds": list(seeds),
        "tw": list(tws),
        "methods": list(methods),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": [],
        "failed": [],
    }

    rows = []
    t_all = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        tag = f"{job['method']}|{job['instance_name']}|{job['tw']}|seed{job['seed']}"
        print(f"[{i}/{len(jobs)}] {tag}", flush=True)
        try:
            row = run_absolute_hco_job(**job)
            if not row.get("feasible"):
                raise RuntimeError(f"infeasible: unserved={row.get('unserved')} err={row.get('error')}")
            # Matched Phase-I checksum for Phase-II methods
            if job["method"] in (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET) and not row.get("phase1_checksum"):
                raise RuntimeError("missing phase1_checksum for Phase-II row")
            rows.append(row)
            status["completed"].append(tag)
            write_master_csv(raw_path, rows)
        except Exception as exc:  # noqa: BLE001
            err = {"job": job, "error": repr(exc), "traceback": traceback.format_exc()}
            status["failed"].append(err)
            write_json(out_dir / "failures.json", status["failed"])
            print(f"  FAILED: {exc}", flush=True)

    status["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    status["wall_s"] = time.perf_counter() - t_all
    status["n_completed"] = len(status["completed"])
    status["n_failed"] = len(status["failed"])
    status["incomplete"] = status["n_completed"] < status["n_jobs_planned"]
    write_json(out_dir / ("pilot_status.json" if pilot else "full_status.json"), status)

    if not rows:
        print("No successful runs; refusing to invent results.", flush=True)
        return 2

    write_master_csv(raw_path, rows)
    missing = assert_reporting_fields_present(raw_path)
    if missing:
        raise SystemExit(f"CSV missing reporting fields: {missing}")

    metrics = (
        "k", "distance", "duration",
        "enroute_purchased_energy", "propulsion_energy",
        "depot_replenishment_energy", "residual_battery_at_depot",
        "charging_visits", "station_waiting", "charging_time", "charging_expense",
        "alns_iterations", "alns_phase1_iterations", "alns_phase2_iterations",
        "solver_runtime_s", "end_to_end_runtime_s",
    )
    table = aggregate_means(rows, ("method", "tw"), metrics)
    write_csv(out_dir / "table_absolute_means.csv", table)

    paired_rows, summaries = [], []
    for tw in tws:
        for a, b, metric in (
            (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, "charging_expense"),
            (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, "distance"),
            (METHOD_PHASE1_PR15, METHOD_PHASE2_HET, "charging_expense"),
            (METHOD_ALNS_PR, METHOD_PHASE2_HET, "distance"),
        ):
            if a not in methods or b not in methods:
                continue
            effects = paired_instance_effects(rows, method_a=a, method_b=b, metric=metric, tw=tw)
            paired_rows.extend(effects)
            summaries.append({"tw": tw, "method_a": a, "method_b": b, "metric": metric, **summarize_paired(effects)})
    write_csv(out_dir / "paired_instance_effects.csv", paired_rows)
    write_csv(out_dir / "paired_summaries.csv", summaries)
    print(f"Wrote {raw_path} ({len(rows)} rows). incomplete={status['incomplete']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
