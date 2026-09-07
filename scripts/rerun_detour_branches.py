"""Rerun only Detour branches after making Detour acceptance geometry-only.

Strips prior FixedRoute-HCO-Detour / PhaseII-Detour rows from raw CSVs,
reruns those methods from saved shared Phase I, appends, then recompiles.
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
    METHOD_FIXED_DETOUR, METHOD_PHASE2_DETOUR, PHASE2_S, SEEDS, TW_VARIANTS,
)
from src.common.paths import hanif_dirs  # noqa: E402
from src.hanif.loader import list_instances, REPRESENTATIVE  # noqa: E402

DETOUR_METHODS = (METHOD_FIXED_DETOUR, METHOD_PHASE2_DETOUR)

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


def _load(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _strip_detour(dirs) -> None:
    for name in ("solution_results.csv", "route_results.csv", "station_visits.csv", "hco_activity.csv"):
        path = dirs["raw"] / name
        rows = _load(path)
        kept = [r for r in rows if r.get("method") not in DETOUR_METHODS]
        if name == "solution_results.csv":
            _write(path, kept, SOLUTION_FIELDS)
        elif kept:
            _write(path, kept, list(kept[0].keys()))
        elif path.is_file():
            path.unlink()
        print(f"stripped detour from {name}: {len(rows)} -> {len(kept)}")
    # Drop obsolete master_results if present
    mp = dirs["raw"] / "master_results.csv"
    if mp.is_file():
        mp.unlink()
        print("deleted master_results.csv (canonical is solution_results.csv)")


def _append(path: Path, rows: list[dict], fields: list[str]) -> None:
    write_header = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _flatten(rows: list[dict], dirs) -> None:
    sol, st, rt, act = [], [], [], []
    for r in rows:
        sol.append({k: r.get(k) for k in SOLUTION_FIELDS})
        st.extend(r.get("station_visits") or [])
        rt.extend(r.get("route_rows") or [])
        a = r.get("hco_activity")
        if isinstance(a, dict) and a:
            act.append({
                "instance": r.get("instance"), "family": r.get("family"),
                "tw": r.get("tw"), "method": r.get("method"), "seed": r.get("seed"),
                "hco_mode": r.get("hco_mode"), **a,
            })
    _append(dirs["raw"] / "solution_results.csv", sol, SOLUTION_FIELDS)
    if st:
        _append(dirs["raw"] / "station_visits.csv", st, list(st[0].keys()))
    if rt:
        _append(dirs["raw"] / "route_results.csv", rt, list(rt[0].keys()))
    if act:
        _append(dirs["raw"] / "hco_activity.csv", act, list(act[0].keys()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--compile-only", action="store_true")
    args = ap.parse_args()
    dirs = hanif_dirs()

    frozen_path = dirs["config"] / "frozen_experiment_config.json"
    eps = float(HYPERPARAMETERS["phase2_distance_tolerance"])
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        eps = float(frozen.get("phase2_distance_tolerance", eps))
        HYPERPARAMETERS["phase2_distance_tolerance"] = eps

    if args.compile_only:
        compile_clean_study()
        return 0

    _strip_detour(dirs)
    names = list(list_instances() or REPRESENTATIVE)
    jobs = []
    for tw in TW_VARIANTS:
        for name in names:
            for method in DETOUR_METHODS:
                for seed in SEEDS:
                    jobs.append({
                        "name": name, "tw": tw, "method": method, "seed": int(seed),
                        "eps": eps, "phase2_s": float(PHASE2_S),
                        "candidates_per_criterion": HYPERPARAMETERS["hco_candidates_per_criterion"],
                    })
    print(f"Detour-only rerun: {len(jobs)} jobs, eps={eps}", flush=True)
    log_path = dirs["logs"] / "final_detour_rerun.log"
    dirs["logs"].mkdir(parents=True, exist_ok=True)
    rows = []
    with log_path.open("w", encoding="utf-8") as log_fh:
        with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futs = {pool.submit(solve_job, j): j for j in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                msg = (
                    f"[{row.get('tw')}] {row.get('instance')} {row.get('method')} "
                    f"seed={row.get('seed')} feas={row.get('feasible')} err={row.get('error')}"
                )
                print(msg, flush=True)
                log_fh.write(msg + "\n")
                rows.append(row)
                if len(rows) % 40 == 0:
                    _flatten(rows[-40:], dirs)
        rem = len(rows) % 40
        if rem:
            _flatten(rows[-rem:], dirs)

    compile_clean_study()
    print("Detour rerun + compile complete.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
