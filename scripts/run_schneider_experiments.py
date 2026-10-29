"""Schneider FR/PR (+ optional HCO-Detour) experiments — separate from Hanif outputs."""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.hco import solution_from_hco  # noqa: E402
from src.alns_hco.hco_activity import OracleCache  # noqa: E402
from src.alns_hco.job_store import JobStore, job_id_generic  # noqa: E402
from src.alns_hco.reviewer_experiments import (  # noqa: E402
    aggregate_means,
    reviewer_root,
    write_csv,
    write_json,
)
from src.alns_hco.schneider_adapter import (  # noqa: E402
    SCHNEIDER_REPRESENTATIVE,
    characteristics_table,
    ensure_representative_converted,
    load_schneider,
)
from src.alns_hco.solver import solve_alns_hco  # noqa: E402
from src.alns_hco.station_log import solution_row  # noqa: E402
from src.common.energy_accounting import attach_energy_reporting, attach_search_diagnostics  # noqa: E402
from src.common.paired_stats import paired_instance_effects, protocol_completion, summarize_paired  # noqa: E402
from src.hanif.benchmarks.alns import solve_alns  # noqa: E402
from src.hanif.benchmarks.greedy import solve_greedy  # noqa: E402

METHOD_FR = "ALNS-FR"
METHOD_PR = "ALNS-PR"
METHOD_DETOUR = "FixedRoute-HCO-Detour"


def _finalize(row, inst, routes, stages=None):
    t0 = time.perf_counter()
    if row.get("solver_runtime_s") is None and row.get("runtime_s") is not None:
        row["solver_runtime_s"] = float(row["runtime_s"])
    attach_energy_reporting(row, inst, routes)
    attach_search_diagnostics(row, stages=stages or {})
    oh = time.perf_counter() - t0
    row["reporting_overhead_s"] = oh
    row["end_to_end_runtime_s"] = float(row["solver_runtime_s"]) + oh
    row["execution_time_s"] = row["end_to_end_runtime_s"]
    row["benchmark"] = "Schneider"
    return row


def run_schneider_job(job: dict) -> dict:
    name, method, seed, budget = job["instance_name"], job["method"], int(job["seed"]), float(job["budget_s"])
    inst = load_schneider(name)
    from src.hanif.data.schneider_txt_convert import schneider_family
    meta = {
        "instance": name,
        "family": schneider_family(name),
        "tw": "schneider",
        "method": method,
        "seed": seed,
        "phase": "final",
        "budget_s": budget,
        "study": "schneider_fr_pr",
        "benchmark": "Schneider",
    }

    if method == METHOD_FR:
        gi = solve_greedy(inst, deadline=20.0)
        out = solve_alns(inst, gi["routes"], time_limit_s=budget, seed=seed)
        rt = float(out["runtime_s"]) + float(gi["runtime_s"])
        row = solution_row(out["metrics"], meta=meta, runtime_s=rt, instance=inst, routes=out["routes"])
        row.update({"termination_reason": "time_limit", "alns_iterations": out.get("iterations"), "error": None})
        return _finalize(row, inst, out["routes"])

    if method == METHOD_PR:
        result = solve_alns_hco(inst, mode="alns_pr", phase1_s=budget, phase2_s=0.0, seed=seed)
        p1 = result.stages.get("phase1") or {}
        row = solution_row(
            result.metrics, meta=meta, runtime_s=result.runtime_s,
            phase1_checksum=p1.get("customer_checksum"), instance=inst, routes=result.routes,
        )
        row.update({"termination_reason": "time_limit", "error": None, "k_star": result.validation.get("k_star")})
        return _finalize(row, inst, result.routes, stages=result.stages)

    if method == METHOD_DETOUR:
        # Build PR start then fixed-route detour HCO (compatible without hetero attrs)
        result = solve_alns_hco(inst, mode="alns_pr", phase1_s=budget, phase2_s=0.0, seed=seed)
        planned, metrics = solution_from_hco(
            inst, result.routes, cache=OracleCache(), mode="detour",
            candidates_per_criterion=5, non_worsening=True,
        )
        routes = planned if planned else result.routes
        metrics = metrics if planned else result.metrics
        row = solution_row(
            metrics, meta=meta, runtime_s=result.runtime_s, instance=inst, routes=routes,
        )
        row.update({
            "termination_reason": "alns_pr_then_hco_detour",
            "hco_mode": "detour",
            "error": None,
            "k_star": int(metrics.used_k),
        })
        return _finalize(row, inst, routes, stages=result.stages)

    raise ValueError(method)


def _worker(job: dict):
    jid = job_id_generic(job)
    try:
        row = run_schneider_job(job)
        # Always return the row (incl. infeasible) for feasibility reporting.
        return jid, row, None, None
    except Exception as exc:  # noqa: BLE001
        return jid, None, repr(exc), traceback.format_exc()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--all56", action="store_true", help="All 56 large Schneider instances, FR/PR only")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--budget", type=float, default=30.0)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()

    out = reviewer_root() / "schneider"
    out.mkdir(parents=True, exist_ok=True)

    if args.all56:
        from src.hanif.data.schneider_txt_convert import LARGE_56, ensure_schneider_converted
        ensure_schneider_converted(names=list(LARGE_56))
        chars = characteristics_table(list(LARGE_56))
        names = list(LARGE_56)
        seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2)
        methods = (METHOD_FR, METHOD_PR)
        label = "schneider_main_56"
    elif args.pilot and not args.full:
        ensure_representative_converted()
        chars = characteristics_table()
        names = list(SCHNEIDER_REPRESENTATIVE[:3])
        seeds = tuple(args.seeds) if args.seeds is not None else (0,)
        methods = (METHOD_FR, METHOD_PR)
        label = "pilot"
    else:
        ensure_representative_converted()
        chars = characteristics_table()
        names = list(SCHNEIDER_REPRESENTATIVE)
        seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2)
        methods = (METHOD_FR, METHOD_PR, METHOD_DETOUR)
        label = "full_selected"

    write_csv(out / "table_characteristics.csv", chars)
    write_json(out / "characteristics.json", chars)
    bad = [c for c in chars if not c["ok"]]
    if bad:
        raise SystemExit(f"Schneider validation failed: {bad[:3]}")

    budget = float(args.budget)
    planned = []
    for name in names:
        for method in methods:
            for seed in seeds:
                planned.append({
                    "instance_name": name, "method": method, "seed": int(seed), "budget_s": budget,
                })

    store = JobStore(out)
    if args.analyze_only:
        rows = store.rebuild_csv_all()
        print("rows", len(rows), protocol_completion(
            store.load_done_ids(require_feasible=False), planned, job_id_fn=job_id_generic,
        ))
        return 0

    # Count attempted (feasible or infeasible) so hard instances are not retried forever.
    done = store.load_done_ids(require_feasible=False)
    todo = [j for j in planned if job_id_generic(j) not in done]
    print(f"Schneider {label}: planned={len(planned)} todo={len(todo)}", flush=True)

    if todo:
        workers = max(1, int(args.workers))
        wave = max(workers * 4, 8)
        remaining = list(todo)
        done_i = 0
        while remaining:
            batch = remaining[:wave]
            remaining = remaining[wave:]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker, j): j for j in batch}
                for fut in as_completed(futs):
                    done_i += 1
                    job = futs[fut]
                    jid, row, err, tb = fut.result()
                    if err or row is None:
                        store.save_failure(jid, job, err or "unknown", tb)
                        print(f"[{done_i}/{len(todo)}] FAIL {jid}: {err}", flush=True)
                    else:
                        # Persist feasible and infeasible outcomes; JobStore.is_done
                        # still requires feasible for resume-skip.
                        store.save_success(jid, job, row)
                        tag = "OK" if row.get("feasible") else "INFEAS"
                        print(
                            f"[{done_i}/{len(todo)}] {tag} {jid} K={row.get('k')} "
                            f"D={float(row.get('distance') or 0):.2f} "
                            f"unserved={row.get('unserved')} feas={row.get('feasible')}",
                            flush=True,
                        )

    all_rows = store.rebuild_csv_all()
    feas_rows = [r for r in all_rows if str(r.get("feasible")).lower() in {"true", "1"}]
    status = {
        "label": label,
        "budget_s": budget,
        "neutral_charging_assumptions": (
            "Homogeneous rate 1/g from Schneider file; station waiting time = 0; "
            "uniform neutral expected price = 1.0 for schema only. "
            "Not used to claim heterogeneous price/waiting benefits."
        ),
        "n_attempted": len(all_rows),
        "n_feasible_rows": len(feas_rows),
        "n_infeasible_rows": len(all_rows) - len(feas_rows),
        "feasibility_rate": (len(feas_rows) / len(all_rows)) if all_rows else None,
        **protocol_completion(
            store.load_done_ids(require_feasible=False), planned, job_id_fn=job_id_generic,
        ),
    }
    write_json(out / "status.json", status)
    write_csv(out / "table_feasibility.csv", [
        {
            "instance": r.get("instance"), "family": r.get("family"), "method": r.get("method"),
            "seed": r.get("seed"), "budget_s": r.get("budget_s"),
            "feasible": r.get("feasible"), "unserved": r.get("unserved"),
            "k": r.get("k"), "distance": r.get("distance"),
        }
        for r in all_rows
    ])
    if feas_rows:
        metrics = (
            "k", "distance", "duration", "charging_visits", "charged_energy",
            "enroute_purchased_energy", "propulsion_energy",
            "alns_iterations", "solver_runtime_s",
        )
        write_csv(out / "table_absolute_means.csv", aggregate_means(feas_rows, ("method", "family"), metrics))
        write_csv(out / "table_absolute_by_instance.csv", aggregate_means(feas_rows, ("method", "instance"), metrics))
        paired, summaries = [], []
        for metric in ("distance", "k", "charged_energy", "solver_runtime_s"):
            eff = paired_instance_effects(
                feas_rows, method_a=METHOD_FR, method_b=METHOD_PR, metric=metric,
                tw="schneider", budget_s=budget, require_budget=True,
            )
            paired.extend(eff)
            summaries.append({"metric": metric, **summarize_paired(eff)})
        write_csv(out / "paired_instance_effects.csv", paired)
        write_csv(out / "paired_summaries.csv", summaries)
    print(status, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
