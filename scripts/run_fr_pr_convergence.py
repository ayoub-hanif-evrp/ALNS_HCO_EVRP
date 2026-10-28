"""FR vs PR convergence study for reviewer revision (independent of paper master CSV).

Does not modify ALNS algorithms. Writes only under outputs/reviewer_revision/.
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
    FR_PR_BUDGETS_S,
    aggregate_means,
    paired_instance_effects,
    plot_fr_pr_convergence,
    reviewer_root,
    run_fr_pr_job,
    select_instances,
    summarize_paired,
    write_csv,
    write_json,
)
from src.common.config import METHOD_ALNS_FR, METHOD_ALNS_PR, SEEDS, TW_VARIANTS  # noqa: E402
from src.common.reporting_io import assert_reporting_fields_present, write_master_csv  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true", help="Small subset + 30 s only + seed 0")
    ap.add_argument("--full", action="store_true", help="Balanced 12 instances, both TW, seeds 0-4, budgets 30/120/300")
    ap.add_argument("--budgets", type=float, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tw", nargs="*", default=None)
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--max-jobs", type=int, default=None, help="Hard cap for partial runs")
    args = ap.parse_args()

    pilot = bool(args.pilot) or not args.full
    instances = select_instances(pilot=pilot, names=args.instances)
    budgets = tuple(args.budgets) if args.budgets is not None else ((30.0,) if pilot else FR_PR_BUDGETS_S)
    seeds = tuple(args.seeds) if args.seeds is not None else ((0,) if pilot else SEEDS)
    tws = tuple(args.tw) if args.tw is not None else (("original",) if pilot else TW_VARIANTS)
    methods = (METHOD_ALNS_FR, METHOD_ALNS_PR)

    jobs = []
    for tw in tws:
        for name in instances:
            for method in methods:
                for budget in budgets:
                    for seed in seeds:
                        jobs.append({
                            "instance_name": name, "tw": tw, "method": method,
                            "seed": int(seed), "budget_s": float(budget),
                        })
    if args.max_jobs is not None:
        jobs = jobs[: int(args.max_jobs)]

    out_dir = reviewer_root() / "fr_pr_convergence"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / ("pilot_runs.csv" if pilot else "full_runs.csv")
    status = {
        "mode": "pilot" if pilot else "full",
        "n_jobs_planned": len(jobs),
        "instances": instances,
        "budgets_s": list(budgets),
        "seeds": list(seeds),
        "tw": list(tws),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": [],
        "failed": [],
    }

    rows = []
    t_all = time.perf_counter()
    for i, job in enumerate(jobs, 1):
        tag = f"{job['method']}|{job['instance_name']}|{job['tw']}|seed{job['seed']}|{job['budget_s']}s"
        print(f"[{i}/{len(jobs)}] {tag}", flush=True)
        try:
            row = run_fr_pr_job(**job)
            if not row.get("feasible"):
                raise RuntimeError(f"infeasible result: {row.get('error') or row.get('termination_reason')}")
            rows.append(row)
            status["completed"].append(tag)
            # Incremental save so partial runs are not lost
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
        "k", "distance", "charged_energy", "propulsion_energy",
        "enroute_purchased_energy", "depot_replenishment_energy",
        "residual_battery_at_depot", "charging_visits",
        "alns_iterations", "solver_runtime_s", "end_to_end_runtime_s",
    )
    agg = aggregate_means(rows, ("method", "budget_s", "tw"), metrics)
    # also overall ignoring tw
    agg_all = aggregate_means(rows, ("method", "budget_s"), metrics)
    write_csv(out_dir / "table_convergence_by_tw.csv", agg)
    write_csv(out_dir / "table_convergence.csv", agg_all)
    plot_fr_pr_convergence(agg_all, out_dir / "fig_fr_pr_convergence.png")

    paired_rows = []
    summaries = []
    for budget in budgets:
        for tw in tws:
            subset = [r for r in rows if float(r.get("budget_s") or -1) == float(budget) and r.get("tw") == tw]
            for metric in ("distance", "k", "propulsion_energy", "enroute_purchased_energy"):
                effects = paired_instance_effects(
                    subset, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR, metric=metric, tw=tw,
                )
                for e in effects:
                    e["budget_s"] = budget
                paired_rows.extend(effects)
                summaries.append({
                    "budget_s": budget, "tw": tw, "metric": metric,
                    **summarize_paired(effects),
                })
    write_csv(out_dir / "paired_instance_effects.csv", paired_rows)
    write_csv(out_dir / "paired_summaries.csv", summaries)
    print(f"Wrote {raw_path} ({len(rows)} rows). incomplete={status['incomplete']}", flush=True)
    return 0 if not status["incomplete"] else 0  # partial success still exit 0 after writing status


if __name__ == "__main__":
    raise SystemExit(main())
