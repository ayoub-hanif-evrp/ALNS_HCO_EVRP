"""Staged clean Hanif ALNS-HCO revision study.

  # Development subset for epsilon selection
  python scripts/select_phase2_epsilon.py

  # Full study (Hanif only)
  python scripts/run_clean_hanif_study.py --workers 8

  # Smoke test
  python scripts/run_clean_hanif_study.py --representative --seeds 0 --workers 1
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

from src.alns_hco.clean_experiments import solve_job  # noqa: E402
from src.alns_hco.clean_study import compile_clean_study  # noqa: E402
from src.alns_hco.hyperparameters import HYPERPARAMETERS  # noqa: E402
from src.common.config import (  # noqa: E402
    CLEAN_METHODS, METHOD_GI, METHOD_PHASE1_PR15, ROUTING_METHODS,
    SEEDS, SHARED_DEPENDENT, TW_VARIANTS, experiment_config,
)
from src.common.paths import hanif_dirs  # noqa: E402
from src.hanif.loader import REPRESENTATIVE, list_instances  # noqa: E402

SOLUTION_FIELDS = [
    "instance", "family", "tw", "method", "seed", "phase",
    "feasible", "served", "unserved", "hard_violations",
    "k", "distance", "duration", "customer_waiting",
    "charging_visits", "charged_energy", "station_waiting",
    "charging_time", "charging_expense",
    "runtime_s", "termination_reason",
    "phase1_s", "phase2_s", "phase1_k", "phase1_distance", "phase1_charging_expense",
    "k_star", "k_frozen_ok", "phase1_checksum", "hco_mode", "eps",
    "Cch_over_Ech", "Twait_over_Nch", "Tch_over_Ech", "error",
]


def _names(args) -> list[str]:
    if args.instances:
        return list(args.instances)
    if args.representative:
        return list(REPRESENTATIVE)
    return list(list_instances() or REPRESENTATIVE)


def _jobs(names, methods, seeds, tws, eps: float, phase2_s: float | None = None) -> list[dict]:
    jobs = []
    for tw in tws:
        for name in names:
            for method in methods:
                use_seeds = (0,) if method == METHOD_GI else seeds
                for seed in use_seeds:
                    jobs.append({
                        "name": name, "tw": tw, "method": method, "seed": int(seed),
                        "eps": eps,
                        "phase2_s": phase2_s,
                        "candidates_per_criterion": HYPERPARAMETERS["hco_candidates_per_criterion"],
                    })
    return jobs


def _append_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _run_pool(jobs: list[dict], workers: int, log_fh) -> list[dict]:
    rows = []
    if workers <= 1:
        for job in jobs:
            row = solve_job(job)
            msg = (
                f"[{row.get('tw')}] {row.get('instance')} {row.get('method')} "
                f"seed={row.get('seed')} feas={row.get('feasible')} "
                f"K={row.get('k')} err={row.get('error')}"
            )
            print(msg, flush=True)
            log_fh.write(msg + "\n")
            rows.append(row)
        return rows
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {pool.submit(solve_job, job): job for job in jobs}
        for fut in as_completed(futs):
            row = fut.result()
            msg = (
                f"[{row.get('tw')}] {row.get('instance')} {row.get('method')} "
                f"seed={row.get('seed')} feas={row.get('feasible')} "
                f"K={row.get('k')} err={row.get('error')}"
            )
            print(msg, flush=True)
            log_fh.write(msg + "\n")
            rows.append(row)
    return rows


def _flatten_logs(rows: list[dict], dirs) -> None:
    sol_rows, st_rows, rt_rows, act_rows = [], [], [], []
    for r in rows:
        sol = {k: r.get(k) for k in SOLUTION_FIELDS}
        sol_rows.append(sol)
        for sv in r.get("station_visits") or []:
            st_rows.append(sv)
        for rr in r.get("route_rows") or []:
            rt_rows.append(rr)
        act = r.get("hco_activity")
        if isinstance(act, dict) and act:
            act_rows.append({
                "instance": r.get("instance"),
                "family": r.get("family"),
                "tw": r.get("tw"),
                "method": r.get("method"),
                "seed": r.get("seed"),
                "hco_mode": r.get("hco_mode"),
                **act,
            })
    _append_csv(dirs["raw"] / "solution_results.csv", sol_rows, SOLUTION_FIELDS)
    if st_rows:
        fields = list(st_rows[0].keys())
        _append_csv(dirs["raw"] / "station_visits.csv", st_rows, fields)
    if rt_rows:
        fields = list(rt_rows[0].keys())
        _append_csv(dirs["raw"] / "route_results.csv", rt_rows, fields)
    if act_rows:
        fields = list(act_rows[0].keys())
        _append_csv(dirs["raw"] / "hco_activity.csv", act_rows, fields)
    master_fields = [f for f in SOLUTION_FIELDS if f != "phase"]
    _append_csv(dirs["raw"] / "master_results.csv", sol_rows, master_fields)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", choices=["original", "relaxed", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=list(SEEDS))
    ap.add_argument("--stage", choices=["all", "routing", "phase1", "dependent", "compile"], default="all")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="Delete prior raw CSVs before writing")
    args = ap.parse_args()

    dirs = hanif_dirs()
    cfg = experiment_config()
    frozen_path = dirs["config"] / "frozen_experiment_config.json"
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        eps = float(args.eps if args.eps is not None else frozen.get("phase2_distance_tolerance", cfg["phase2_distance_tolerance"]))
        # Sync budgets from frozen time-allocation selection.
        import src.common.config as cfgmod
        if "phase1_time_limit_s" in frozen:
            cfgmod.PHASE1_S = float(frozen["phase1_time_limit_s"])
        if "phase2_time_limit_s" in frozen:
            cfgmod.PHASE2_S = float(frozen["phase2_time_limit_s"])
        HYPERPARAMETERS["phase1_time_limit_s"] = float(frozen.get("phase1_time_limit_s", cfgmod.PHASE1_S))
        HYPERPARAMETERS["phase2_time_limit_s"] = float(frozen.get("phase2_time_limit_s", cfgmod.PHASE2_S))
        HYPERPARAMETERS["phase2_distance_tolerance"] = eps
    else:
        eps = float(args.eps if args.eps is not None else HYPERPARAMETERS["phase2_distance_tolerance"])
        (dirs["config"] / "frozen_experiment_config.json").write_text(
            json.dumps({**cfg, "phase2_distance_tolerance": eps, "provisional": True}, indent=2) + "\n",
            encoding="utf-8",
        )

    if args.fresh:
        for name in (
            "solution_results.csv", "station_visits.csv", "route_results.csv",
            "master_results.csv", "hco_activity.csv",
        ):
            p = dirs["raw"] / name
            if p.is_file():
                p.unlink()

    names = _names(args)
    seeds = tuple(int(s) for s in args.seeds)
    tws = list(TW_VARIANTS) if args.tw == "both" else [args.tw]
    import src.common.config as cfgmod
    phase2_s = float(cfgmod.PHASE2_S)
    log_path = dirs["logs"] / "clean_study_run.log"
    log_fh = log_path.open("a", encoding="utf-8")

    try:
        if args.stage in ("all", "routing"):
            jobs = _jobs(names, list(ROUTING_METHODS), seeds, tws, eps, phase2_s)
            print(f"=== Stage routing: {len(jobs)} jobs ===", flush=True)
            rows = _run_pool(jobs, args.workers, log_fh)
            _flatten_logs(rows, dirs)

        if args.stage in ("all", "phase1"):
            jobs = _jobs(names, [METHOD_PHASE1_PR15], seeds, tws, eps, phase2_s)
            print(f"=== Stage shared Phase I: {len(jobs)} jobs ===", flush=True)
            rows = _run_pool(jobs, args.workers, log_fh)
            _flatten_logs(rows, dirs)

        if args.stage in ("all", "dependent"):
            jobs = _jobs(names, list(SHARED_DEPENDENT), seeds, tws, eps, phase2_s)
            print(f"=== Stage fixed-route + Phase II: {len(jobs)} jobs ===", flush=True)
            rows = _run_pool(jobs, args.workers, log_fh)
            _flatten_logs(rows, dirs)

        if args.stage in ("all", "compile", "routing", "phase1", "dependent"):
            compile_clean_study()
    finally:
        log_fh.close()

    print("Clean Hanif study artifacts under outputs/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
