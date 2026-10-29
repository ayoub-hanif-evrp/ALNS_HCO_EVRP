"""Resumable FR vs PR multi-budget convergence (Reviewer Round 3).

Equal-budget ALNS-FR vs ALNS-PR. Does not modify algorithms.
Checkpoints under outputs/reviewer_revision/fr_pr_convergence/checkpoints/.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.job_store import JobStore, job_id_fr_pr  # noqa: E402
from src.alns_hco.reviewer_experiments import (  # noqa: E402
    FR_PR_BUDGETS_S,
    aggregate_means,
    plot_fr_pr_convergence,
    reviewer_root,
    run_fr_pr_job,
    select_instances,
    write_csv,
    write_json,
)
from src.common.config import METHOD_ALNS_FR, METHOD_ALNS_PR, SEEDS, TW_VARIANTS  # noqa: E402
from src.common.paired_stats import (  # noqa: E402
    paired_instance_effects,
    protocol_completion,
    summarize_paired,
)
from src.common.reporting_io import assert_reporting_fields_present  # noqa: E402


RUNTIME_NOTE = (
    "solver_runtime_s = wall-clock inside ALNS (+ GI for FR). "
    "reporting_overhead_s = post-hoc energy/diagnostics only. "
    "end_to_end_runtime_s = solver_runtime_s + reporting_overhead_s. "
    "Equal search budget is the ALNS time_limit_s / phase1_s argument (30/120/300)."
)


def _worker(job: dict) -> tuple[str, dict | None, str | None, str | None]:
    jid = job_id_fr_pr(job)
    try:
        row = run_fr_pr_job(**job)
        if not row.get("feasible"):
            return jid, None, f"infeasible:{row.get('error')}", None
        return jid, row, None, None
    except Exception as exc:  # noqa: BLE001
        return jid, None, repr(exc), traceback.format_exc()


def _ingest_legacy_csv(store: JobStore, path: Path) -> int:
    """Import Round-2 CSV rows as checkpoints when job_id matches."""
    import csv

    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("feasible")).lower() not in {"true", "1"}:
                continue
            job = {
                "method": r["method"],
                "instance_name": r["instance"],
                "tw": r["tw"],
                "seed": int(r["seed"]),
                "budget_s": float(r.get("budget_s") or 30.0),
            }
            jid = job_id_fr_pr(job)
            if store.is_done(jid):
                continue
            store.save_success(jid, job, r)
            n += 1
    return n


def build_jobs(instances, tws, budgets, seeds, methods, *, short_first: bool = True):
    """Build job list. Default: ascending budget so 30s cells finish before 300s."""
    jobs = []
    budget_order = sorted(budgets) if short_first else list(budgets)
    for budget in budget_order:
        for tw in tws:
            for name in instances:
                for method in methods:
                    for seed in seeds:
                        jobs.append({
                            "instance_name": name,
                            "tw": tw,
                            "method": method,
                            "seed": int(seed),
                            "budget_s": float(budget),
                        })
    return jobs


def analyze(store: JobStore, planned: list[dict], *, label: str) -> dict:
    rows = store.rebuild_csv()
    out_dir = store.root
    status = {
        "label": label,
        "runtime_definition": RUNTIME_NOTE,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **protocol_completion(store.load_done_ids(), planned, job_id_fn=job_id_fr_pr),
    }
    if not rows:
        store.write_status(status)
        return status

    missing_fields = assert_reporting_fields_present(store.runs_csv)
    status["missing_reporting_fields"] = missing_fields

    metrics = (
        "k", "distance", "charged_energy", "propulsion_energy",
        "enroute_purchased_energy", "depot_replenishment_energy",
        "residual_battery_at_depot", "charging_visits",
        "alns_iterations", "solver_runtime_s", "end_to_end_runtime_s",
    )
    write_csv(out_dir / "table_convergence_by_tw.csv", aggregate_means(rows, ("method", "budget_s", "tw"), metrics))
    write_csv(out_dir / "table_convergence.csv", aggregate_means(rows, ("method", "budget_s"), metrics))

    # Plots: overall and per TW when multi-budget present
    budgets_present = sorted({float(r["budget_s"]) for r in rows if r.get("budget_s") not in (None, "")})
    status["budgets_present_s"] = budgets_present
    plot_fr_pr_convergence(aggregate_means(rows, ("method", "budget_s"), metrics), out_dir / "fig_fr_pr_convergence.png")
    for tw in sorted({r.get("tw") for r in rows}):
        plot_fr_pr_convergence(
            aggregate_means([r for r in rows if r.get("tw") == tw], ("method", "budget_s", "tw"), metrics),
            out_dir / f"fig_fr_pr_convergence_{tw}.png",
            tw=tw,
        )

    paired_rows, summaries = [], []
    for budget in budgets_present:
        for tw in sorted({r.get("tw") for r in rows}):
            for metric in ("distance", "k", "propulsion_energy", "enroute_purchased_energy", "alns_iterations", "solver_runtime_s"):
                effects = paired_instance_effects(
                    rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR,
                    metric=metric, tw=tw, budget_s=budget, require_budget=True,
                )
                paired_rows.extend(effects)
                summaries.append({
                    "budget_s": budget, "tw": tw, "metric": metric,
                    **summarize_paired(effects),
                })
    write_csv(out_dir / "paired_instance_effects.csv", paired_rows)
    write_csv(out_dir / "paired_summaries.csv", summaries)
    # Effort table
    write_csv(
        out_dir / "table_effort.csv",
        aggregate_means(rows, ("method", "budget_s", "tw"), ("alns_iterations", "solver_runtime_s", "end_to_end_runtime_s")),
    )
    store.write_status(status)
    write_json(out_dir / "analysis_status.json", status)
    return status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--balanced", action="store_true",
                    help="Compute-limited fallback: 6 inst (1/family) × both TW × 3 budgets × seeds 0-2")
    ap.add_argument("--priority", action="store_true", help="12 inst × both TW × 3 budgets × seeds 0-2")
    ap.add_argument("--full", action="store_true", help="12 inst × both TW × 3 budgets × seeds 0-4")
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--budgets", type=float, nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--tw", nargs="*", default=None)
    ap.add_argument("--max-jobs", type=int, default=None)
    args = ap.parse_args()

    if args.full:
        seeds = tuple(args.seeds) if args.seeds is not None else SEEDS
        tws = tuple(args.tw) if args.tw is not None else TW_VARIANTS
        budgets = tuple(args.budgets) if args.budgets is not None else FR_PR_BUDGETS_S
        instances = select_instances(pilot=False)
        label = "full_protocol"
    elif args.balanced:
        seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2)
        tws = tuple(args.tw) if args.tw is not None else TW_VARIANTS
        budgets = tuple(args.budgets) if args.budgets is not None else FR_PR_BUDGETS_S
        instances = select_instances(balanced_core=True)
        label = "balanced_core_protocol"
    elif args.priority:
        seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2)
        tws = tuple(args.tw) if args.tw is not None else TW_VARIANTS
        budgets = tuple(args.budgets) if args.budgets is not None else FR_PR_BUDGETS_S
        instances = select_instances(pilot=False)
        label = "priority_protocol"
    elif args.pilot:
        seeds = tuple(args.seeds) if args.seeds is not None else (0,)
        tws = tuple(args.tw) if args.tw is not None else ("original",)
        budgets = tuple(args.budgets) if args.budgets is not None else (30.0,)
        instances = select_instances(pilot=True)
        label = "pilot"
    else:
        # Default under compute pressure: balanced core (not silent full priority).
        seeds = tuple(args.seeds) if args.seeds is not None else (0, 1, 2)
        tws = tuple(args.tw) if args.tw is not None else TW_VARIANTS
        budgets = tuple(args.budgets) if args.budgets is not None else FR_PR_BUDGETS_S
        instances = select_instances(balanced_core=True)
        label = "balanced_core_protocol"

    methods = (METHOD_ALNS_FR, METHOD_ALNS_PR)
    planned = build_jobs(instances, tws, budgets, seeds, methods, short_first=True)
    if args.max_jobs is not None:
        planned = planned[: int(args.max_jobs)]

    store = JobStore(reviewer_root() / "fr_pr_convergence")
    # Resume from Round-2 CSVs
    n_leg = 0
    n_leg += _ingest_legacy_csv(store, store.root / "full_runs.csv")
    n_leg += _ingest_legacy_csv(store, store.root / "pilot_runs.csv")
    print(f"Ingested {n_leg} legacy checkpoints", flush=True)

    if args.analyze_only:
        st = analyze(store, planned, label=label)
        print(st, flush=True)
        return 0 if st.get("complete_protocol") else 0

    done = store.load_done_ids()
    n_done_planned = sum(1 for j in planned if job_id_fr_pr(j) in done)
    todo = [j for j in planned if job_id_fr_pr(j) not in done]
    print(
        f"Protocol={label} planned={len(planned)} done={n_done_planned} "
        f"todo={len(todo)} workers={args.workers}",
        flush=True,
    )

    t0 = time.perf_counter()
    failures = []
    if todo:
        workers = max(1, int(args.workers))
        # Submit in waves so a blocked main thread cannot strand hundreds of
        # finished futures, and memory stays bounded.
        wave = max(workers * 4, 8)
        remaining = list(todo)
        done_i = 0
        while remaining:
            batch = remaining[:wave]
            remaining = remaining[wave:]
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(_worker, job): job for job in batch}
                for fut in as_completed(futs):
                    done_i += 1
                    job = futs[fut]
                    jid, row, err, tb = fut.result()
                    if err or row is None:
                        store.save_failure(jid, job, err or "unknown", tb)
                        failures.append({"job_id": jid, "error": err})
                        print(f"[{done_i}/{len(todo)}] FAIL {jid}: {err}", flush=True)
                    else:
                        store.save_success(jid, job, row)
                        print(
                            f"[{done_i}/{len(todo)}] OK {jid} K={row.get('k')} "
                            f"D={float(row.get('distance') or 0):.1f} "
                            f"iters={row.get('alns_iterations')} "
                            f"t={float(row.get('solver_runtime_s') or 0):.1f}s",
                            flush=True,
                        )
                    if done_i % 25 == 0:
                        store.rebuild_csv()
                        st = protocol_completion(
                            store.load_done_ids(), planned, job_id_fn=job_id_fr_pr,
                        )
                        store.write_status({
                            "label": label,
                            "runtime_definition": RUNTIME_NOTE,
                            "partial": True,
                            "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            **st,
                        })
                        print(
                            f"  checkpoint progress {st['n_completed']}/{st['n_planned']}",
                            flush=True,
                        )

    st = analyze(store, planned, label=label)
    st["wall_s"] = time.perf_counter() - t0
    st["n_failures_this_session"] = len(failures)
    st["reduction_note"] = (
        "balanced_core_protocol uses 1 instance/family (6) instead of 2 (12) due to "
        "machine time/RAM limits. Still covers all six families, both TW, budgets "
        "30/120/300 s, and seeds 0–2. The larger priority_protocol (432 jobs) remains "
        "incomplete; missing jobs are listed explicitly."
        if label == "balanced_core_protocol" else None
    )
    # Also record status of the larger priority design for transparency.
    if label == "balanced_core_protocol":
        pri = build_jobs(
            select_instances(pilot=False), TW_VARIANTS, FR_PR_BUDGETS_S, (0, 1, 2), methods,
        )
        st["priority_protocol_reference"] = protocol_completion(
            store.load_done_ids(), pri, job_id_fn=job_id_fr_pr,
        )
    store.write_status(st)
    write_json(store.root / "failures_session.json", failures)
    print(
        f"DONE session. completed_protocol={st['complete_protocol']} "
        f"completed={st['n_completed']}/{st['n_planned']} missing={st['n_missing']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
