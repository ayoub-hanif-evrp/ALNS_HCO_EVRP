"""Rerun only corrected HCO / Phase-II branches from saved shared Phase-I solutions.

Does NOT rerun GI, VNS, SA, ALNS-FR, ALNS-PR, or regenerate shared Phase I.
Appends to existing trusted baseline rows in outputs/raw/.
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
    HCO_DEPENDENT_METHODS, MASTER_FIELDS, METHOD_GI, PHASE2_S, SEEDS, TW_VARIANTS,
    experiment_config,
)
from src.common.paths import hanif_dirs  # noqa: E402
from src.hanif.loader import REPRESENTATIVE, list_instances  # noqa: E402

# Include Round-1/2 reporting fields so post-hoc metrics survive CSV export.
SOLUTION_FIELDS = list(MASTER_FIELDS) + [
    "phase", "Cch_over_Ech", "Twait_over_Nch", "Tch_over_Ech",
]
# Preserve stable uniqueness / order
_seen = set()
SOLUTION_FIELDS = [f for f in SOLUTION_FIELDS if not (f in _seen or _seen.add(f))]


def _jobs(names, methods, seeds, tws, eps: float) -> list[dict]:
    jobs = []
    for tw in tws:
        for name in names:
            for method in methods:
                use_seeds = (0,) if method == METHOD_GI else seeds
                for seed in use_seeds:
                    jobs.append({
                        "name": name, "tw": tw, "method": method, "seed": int(seed),
                        "eps": eps,
                        "phase2_s": float(PHASE2_S),
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


def _flatten(rows: list[dict], dirs) -> None:
    sol_rows, st_rows, rt_rows, act_rows = [], [], [], []
    for r in rows:
        sol_rows.append({k: r.get(k) for k in SOLUTION_FIELDS})
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
        _append_csv(dirs["raw"] / "station_visits.csv", st_rows, list(st_rows[0].keys()))
    if rt_rows:
        _append_csv(dirs["raw"] / "route_results.csv", rt_rows, list(rt_rows[0].keys()))
    if act_rows:
        _append_csv(dirs["raw"] / "hco_activity.csv", act_rows, list(act_rows[0].keys()))
    master_fields = [f for f in SOLUTION_FIELDS if f != "phase"]
    _append_csv(dirs["raw"] / "master_results.csv", sol_rows, master_fields)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", default=list(SEEDS))
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()

    dirs = hanif_dirs()
    log_dir = dirs["logs"]
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "final_hco_rerun.log"

    frozen_path = dirs["config"] / "frozen_experiment_config.json"
    eps = float(HYPERPARAMETERS["phase2_distance_tolerance"])
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        eps = float(frozen.get("phase2_distance_tolerance", eps))
        import src.common.config as cfgmod
        if "phase1_time_limit_s" in frozen:
            cfgmod.PHASE1_S = float(frozen["phase1_time_limit_s"])
        if "phase2_time_limit_s" in frozen:
            cfgmod.PHASE2_S = float(frozen["phase2_time_limit_s"])
        HYPERPARAMETERS["phase2_distance_tolerance"] = eps
        HYPERPARAMETERS["phase2_repair_shortlist_size"] = int(
            frozen.get("phase2_repair_shortlist_size", HYPERPARAMETERS.get("phase2_repair_shortlist_size", 5))
        )

    if args.compile_only:
        compile_clean_study()
        print("Compiled artifacts under outputs/", flush=True)
        return 0

    names = list(args.instances) if args.instances else (
        list(REPRESENTATIVE) if args.representative else list(list_instances() or REPRESENTATIVE)
    )
    seeds = tuple(int(s) for s in args.seeds)
    methods = list(HCO_DEPENDENT_METHODS)
    jobs = _jobs(names, methods, seeds, TW_VARIANTS, eps)
    print(f"Final HCO rerun: {len(jobs)} jobs, eps={eps}, methods={methods}", flush=True)

    rows = []
    with log_path.open("w", encoding="utf-8") as log_fh:
        log_fh.write(f"eps={eps} jobs={len(jobs)} methods={methods}\n")
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
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
                if len(rows) % 20 == 0:
                    _flatten(rows[-20:], dirs)
        rem = len(rows) % 20
        if rem:
            _flatten(rows[-rem:], dirs)

    compile_clean_study()
    print("Final HCO rerun complete. Artifacts under outputs/", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
