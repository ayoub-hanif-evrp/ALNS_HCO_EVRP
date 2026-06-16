# Parallel CON+HC baseline sweep (Q grid); writes baseline_solver_results.csv next to this script.

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

from evrptw_solvers import load_ev_specs, run_baseline_job

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VARIANT_INSTANCE_DIRS = [
    ("A", os.path.join(_ROOT, "variant_A_original_TW")),
    ("B", os.path.join(_ROOT, "variant_B_relaxed_TW")),
]
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "baseline_solver_results.csv")

Q_VALUES = [500, 600, 700, 800]
METHODS = ["CON", "HC"]
METHOD_SORT = {m: i for i, m in enumerate(METHODS)}


# Build all (variant, instance, Q, method) tasks with h from ev_specs.json.
def discover_tasks():
    _, energy_rate = load_ev_specs()
    tasks = []
    for variant, vdir in VARIANT_INSTANCE_DIRS:
        if not os.path.isdir(vdir):
            continue
        for base in sorted(os.listdir(vdir)):
            if not base.lower().endswith(".txt"):
                continue
            txt_path = os.path.join(vdir, base)
            stem = base[:-4] if base.lower().endswith(".txt") else base
            csv_stations = os.path.join(vdir, stem + "_stations.csv")
            if not os.path.isfile(csv_stations):
                continue
            for Q in Q_VALUES:
                for method in METHODS:
                    tasks.append(
                        (
                            variant,
                            stem,
                            txt_path,
                            csv_stations,
                            Q,
                            method,
                            energy_rate,
                            METHOD_SORT[method],
                        )
                    )
    return tasks


# Run tasks (sequential or process pool) and write ordered rows to OUTPUT_CSV.
def main():
    ap = argparse.ArgumentParser(description="Baseline CON+HC sweep (parallel).")
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel processes (0 = max(1, CPU-1); 1 = sequential).",
    )
    args = ap.parse_args()
    ncpu = cpu_count() or 1
    if args.workers <= 0:
        workers = max(1, ncpu - 1)
    else:
        workers = args.workers

    tasks = discover_tasks()
    n_tasks = len(tasks)
    print(
        f"Baseline CON+HC: {n_tasks} jobs, {workers} worker(s). Output -> {OUTPUT_CSV}",
        flush=True,
    )
    if n_tasks == 0:
        print(
            "No tasks found. Check that variant_A_original_TW and variant_B_relaxed_TW "
            "exist next to the code/ folder (each .txt with matching _stations.csv).",
            flush=True,
        )
        return

    progress_every = 25 if n_tasks > 100 else max(1, n_tasks // 10)
    print(
        f"  Running… progress printed every {progress_every} job(s). "
        "Use --workers 1 to debug; default uses parallel workers.",
        flush=True,
    )

    header = [
        "variant",
        "instance",
        "Q",
        "energy_rate",
        "method",
        "fleet_in_file",
        "feasible",
        "served",
        "num_customers",
        "total_distance",
        "num_routes",
        "num_vehicles_used",
        "pct_fleet_used",
        "num_charges",
        "avg_charges_per_vehicle",
        "seconds",
    ]

    rows_by_key = {}

    if workers == 1 or len(tasks) == 1:
        for i, t in enumerate(tasks, 1):
            k, row = run_baseline_job(t)
            rows_by_key[k] = row
            if i % progress_every == 0 or i == n_tasks:
                print(f"  Progress: {i}/{n_tasks}", flush=True)
    else:
        print("  Starting process pool (first results may take a few seconds)…", flush=True)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_baseline_job, t): t for t in tasks}
            done = 0
            for fut in as_completed(futs):
                k, row = fut.result()
                rows_by_key[k] = row
                done += 1
                if done % progress_every == 0 or done == n_tasks:
                    print(f"  Progress: {done}/{n_tasks}", flush=True)

    ordered = sorted(rows_by_key.keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for k in ordered:
            w.writerow(rows_by_key[k])

    print("Done. Wrote", len(ordered), "rows to", OUTPUT_CSV, flush=True)


if __name__ == "__main__":
    main()
