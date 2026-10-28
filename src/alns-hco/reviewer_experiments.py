"""Independent reviewer-revision experiment helpers (no algorithm changes)."""
from __future__ import annotations

import csv
import json
import time
from collections import defaultdict
from pathlib import Path

from src.alns_hco.shared_phase1 import ensure_shared_phase1
from src.alns_hco.solver import solve_alns_hco
from src.alns_hco.station_log import solution_row
from src.common.config import (
    FAMILIES,
    METHOD_ALNS_FR,
    METHOD_ALNS_PR,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_HET,
    METHOD_PHASE2_NOHCO,
)
from src.common.energy_accounting import (
    attach_energy_reporting,
    attach_search_diagnostics,
)
from src.common.paths import hanif_dirs
from src.common.stats import bootstrap_mean_ci, mean, pct_change, wins_ties_losses
from src.hanif.benchmarks.alns import solve_alns
from src.hanif.benchmarks.greedy import solve_greedy
from src.hanif.loader import list_instances, load_hanif

# Balanced subset: 2 instances per family (12 total) covering both TW via jobs.
BALANCED_12 = (
    "C1_10_1", "C1_10_2",
    "C2_10_1", "C2_10_2",
    "R1_10_1", "R1_10_2",
    "R2_10_1", "R2_10_2",
    "RC1_10_1", "RC1_10_2",
    "RC2_10_1", "RC2_10_2",
)

PILOT_INSTANCES = ("C1_10_1", "R1_10_1", "RC2_10_1")
FR_PR_BUDGETS_S = (30.0, 120.0, 300.0)


def select_instances(*, pilot: bool = False, names: list[str] | None = None) -> list[str]:
    available = set(list_instances())
    if names:
        chosen = list(names)
    elif pilot:
        chosen = list(PILOT_INSTANCES)
    else:
        chosen = list(BALANCED_12)
    missing = [n for n in chosen if n not in available]
    if missing:
        raise FileNotFoundError(f"Hanif instances not converted / missing: {missing}")
    # Family coverage check
    fams = {n.split("_")[0] for n in chosen}
    if not pilot and not fams.issuperset(set(FAMILIES)):
        raise AssertionError(f"subset must cover all families; got {sorted(fams)}")
    return chosen


def reviewer_root() -> Path:
    root = Path("outputs") / "reviewer_revision"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _family(name: str) -> str:
    return name.split("_")[0]


def _finalize_row(row: dict, instance, routes, *, stages=None, validation=None, shared_iters=None) -> dict:
    t0 = time.perf_counter()
    if row.get("solver_runtime_s") is None and row.get("runtime_s") is not None:
        row["solver_runtime_s"] = float(row["runtime_s"])
    attach_energy_reporting(row, instance, routes)
    attach_search_diagnostics(
        row, stages=stages, validation=validation, shared_phase1_iterations=shared_iters,
    )
    overhead = time.perf_counter() - t0
    row["reporting_overhead_s"] = float(overhead)
    if row.get("solver_runtime_s") is not None:
        row["end_to_end_runtime_s"] = float(row["solver_runtime_s"]) + overhead
    row["execution_time_s"] = row.get("end_to_end_runtime_s")
    return row


def run_fr_pr_job(
    *,
    instance_name: str,
    tw: str,
    method: str,
    seed: int,
    budget_s: float,
) -> dict:
    """Single independent FR or PR run at a given wall-clock budget."""
    if method not in (METHOD_ALNS_FR, METHOD_ALNS_PR):
        raise ValueError(f"FR/PR runner only supports {METHOD_ALNS_FR}/{METHOD_ALNS_PR}")
    inst = load_hanif(instance_name, tw)
    meta = {
        "instance": instance_name,
        "family": _family(instance_name),
        "tw": tw,
        "method": method,
        "seed": int(seed),
        "phase": "final",
        "budget_s": float(budget_s),
        "study": "fr_pr_convergence",
    }
    if method == METHOD_ALNS_FR:
        gi = solve_greedy(inst, deadline=40.0)
        out = solve_alns(inst, gi["routes"], time_limit_s=float(budget_s), seed=int(seed))
        solver_rt = float(out["runtime_s"]) + float(gi["runtime_s"])
        row = solution_row(
            out["metrics"], meta=meta, runtime_s=solver_rt, instance=inst, routes=out["routes"],
        )
        row.update({
            "termination_reason": "time_limit",
            "alns_iterations": out.get("iterations"),
            "gi_runtime_s": gi["runtime_s"],
            "error": None,
        })
        return _finalize_row(row, inst, out["routes"])

    result = solve_alns_hco(
        inst, mode="alns_pr", phase1_s=float(budget_s), phase2_s=0.0, seed=int(seed),
    )
    p1 = result.stages.get("phase1") or {}
    row = solution_row(
        result.metrics, meta=meta, runtime_s=result.runtime_s,
        phase1_checksum=p1.get("customer_checksum"),
        instance=inst, routes=result.routes,
    )
    row.update({
        "termination_reason": "time_limit",
        "phase1_s": p1.get("runtime_s"),
        "k_star": result.validation.get("k_star"),
        "k_frozen_ok": True,
        "error": None,
    })
    return _finalize_row(row, inst, result.routes, stages=result.stages, validation=result.validation)


def run_absolute_hco_job(
    *,
    instance_name: str,
    tw: str,
    method: str,
    seed: int,
    phase1_s: float = 15.0,
    phase2_s: float = 15.0,
    alns_pr_s: float = 30.0,
) -> dict:
    """Absolute-results job using shared Phase I for matched Phase-II branches."""
    shared_root = hanif_dirs()["shared_phase1"]
    meta_base = {
        "instance": instance_name,
        "family": _family(instance_name),
        "tw": tw,
        "method": method,
        "seed": int(seed),
        "phase": "final",
        "study": "alns_hco_absolute",
    }

    if method == METHOD_ALNS_PR:
        inst = load_hanif(instance_name, tw)
        result = solve_alns_hco(
            inst, mode="alns_pr", phase1_s=float(alns_pr_s), phase2_s=0.0, seed=int(seed),
        )
        p1 = result.stages.get("phase1") or {}
        row = solution_row(
            result.metrics, meta=meta_base, runtime_s=result.runtime_s,
            phase1_checksum=p1.get("customer_checksum"),
            instance=inst, routes=result.routes,
        )
        row.update({
            "termination_reason": "time_limit",
            "phase1_s": p1.get("runtime_s"),
            "budget_s": float(alns_pr_s),
            "k_star": result.validation.get("k_star"),
            "k_frozen_ok": True,
            "error": None,
        })
        return _finalize_row(row, inst, result.routes, stages=result.stages, validation=result.validation)

    shared = ensure_shared_phase1(
        shared_root, instance_name=instance_name, tw=tw, seed=int(seed),
        time_limit_s=float(phase1_s),
    )
    inst = shared["instance"]
    rec = shared["record"]
    routes = shared["routes"]
    chk = rec["customer_checksum"]
    k_star = int(rec["k_star"])
    d_pr = float(rec["metrics"]["distance"])
    shared_iters = rec.get("iterations")

    if method == METHOD_PHASE1_PR15:
        row = solution_row(
            shared["metrics"], meta={**meta_base, "phase": "phase1"},
            runtime_s=float(rec["phase1_s"]), phase1_checksum=chk,
            instance=inst, routes=routes,
        )
        row.update({
            "termination_reason": "shared_phase1",
            "phase1_s": rec["phase1_s"],
            "phase2_s": 0.0,
            "budget_s": float(phase1_s),
            "phase1_k": rec["metrics"]["k"],
            "phase1_distance": d_pr,
            "phase1_charging_expense": rec["metrics"]["charging_expense"],
            "k_star": k_star,
            "k_frozen_ok": True,
            "alns_phase1_iterations": shared_iters,
            "alns_iterations": shared_iters,
            "error": None,
        })
        return _finalize_row(row, inst, routes, shared_iters=shared_iters)

    if method not in (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET):
        raise ValueError(f"unsupported absolute method {method}")

    hco_mode = "nohco" if method == METHOD_PHASE2_NOHCO else "heterogeneous"
    result = solve_alns_hco(
        inst, mode="phase2", phase2_s=float(phase2_s), seed=int(seed),
        start_routes=routes, hco_mode=hco_mode, k_star=k_star, d_pr=d_pr,
        phase1_checksum=chk,
    )
    if result.validation.get("phase1_checksum") != chk:
        raise AssertionError("Phase-II used different Phase-I checksum")
    if int(result.metrics.used_k) != k_star:
        raise AssertionError(f"Phase II changed K: {result.metrics.used_k} != {k_star}")
    p2 = result.stages.get("phase2") or {}
    row = solution_row(
        result.metrics, meta=meta_base,
        runtime_s=float(result.runtime_s) + float(rec["phase1_s"]),
        phase1_checksum=chk, instance=inst, routes=result.routes,
    )
    row.update({
        "termination_reason": "phase2_nohco" if method == METHOD_PHASE2_NOHCO else "phase2",
        "phase1_s": rec["phase1_s"],
        "phase2_s": p2.get("runtime_s"),
        "budget_s": float(phase1_s) + float(phase2_s),
        "phase1_k": k_star,
        "phase1_distance": d_pr,
        "phase1_charging_expense": rec["metrics"]["charging_expense"],
        "k_star": k_star,
        "k_frozen_ok": result.validation.get("k_frozen_ok"),
        "hco_mode": hco_mode,
        "alns_phase1_iterations": shared_iters,
        "hco_activity": p2.get("hco_activity") or result.validation.get("hco_activity"),
        "error": None,
    })
    return _finalize_row(
        row, inst, result.routes,
        stages=result.stages, validation=result.validation, shared_iters=shared_iters,
    )


def enrich_phase1_from_shared(*, instance_name: str, tw: str, seed: int) -> dict:
    """Post-hoc energy enrichment of stored Phase I (no re-solve)."""
    shared_root = hanif_dirs()["shared_phase1"]
    shared = ensure_shared_phase1(shared_root, instance_name=instance_name, tw=tw, seed=int(seed))
    return run_absolute_hco_job(
        instance_name=instance_name, tw=tw, method=METHOD_PHASE1_PR15, seed=int(seed),
    )


def paired_instance_effects(
    rows: list[dict],
    *,
    method_a: str,
    method_b: str,
    metric: str,
    tw: str | None = None,
) -> list[dict]:
    """Instance-level paired B−A effects with bootstrap CI over instances."""
    by_inst: dict[str, dict] = defaultdict(dict)
    for r in rows:
        if tw is not None and r.get("tw") != tw:
            continue
        if str(r.get("feasible")).lower() not in {"true", "1"}:
            continue
        if r.get("error"):
            continue
        key = (r["instance"], r.get("tw"), r.get("seed"))
        # average over seeds later at instance level
        by_inst.setdefault((r["instance"], r.get("tw")), {}).setdefault(r["method"], []).append(r)

    effects = []
    for (inst, twv), methods in sorted(by_inst.items()):
        a_rows = methods.get(method_a) or []
        b_rows = methods.get(method_b) or []
        if not a_rows or not b_rows:
            continue
        a_vals = [float(x[metric]) for x in a_rows if x.get(metric) not in (None, "")]
        b_vals = [float(x[metric]) for x in b_rows if x.get(metric) not in (None, "")]
        if not a_vals or not b_vals:
            continue
        a_m, b_m = mean(a_vals), mean(b_vals)
        effects.append({
            "instance": inst,
            "tw": twv,
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "a_mean": a_m,
            "b_mean": b_m,
            "diff_b_minus_a": (None if a_m is None or b_m is None else b_m - a_m),
            "pct_b_vs_a": pct_change(b_m, a_m) if a_m is not None and b_m is not None else None,
        })
    return effects


def summarize_paired(effects: list[dict], *, seed: int = 0) -> dict:
    diffs = [e["diff_b_minus_a"] for e in effects if e.get("diff_b_minus_a") is not None]
    pcts = [e["pct_b_vs_a"] for e in effects if e.get("pct_b_vs_a") is not None]
    a = [e["a_mean"] for e in effects if e.get("a_mean") is not None]
    b = [e["b_mean"] for e in effects if e.get("b_mean") is not None]
    lo, hi = bootstrap_mean_ci(diffs, seed=seed) if diffs else (None, None)
    plo, phi = bootstrap_mean_ci(pcts, seed=seed + 1) if pcts else (None, None)
    w, t, l = wins_ties_losses(b, a) if a and b and len(a) == len(b) else (0, 0, 0)
    return {
        "n_instances": len(effects),
        "mean_diff": mean(diffs),
        "ci95_diff_lo": lo,
        "ci95_diff_hi": hi,
        "mean_pct": mean(pcts),
        "ci95_pct_lo": plo,
        "ci95_pct_hi": phi,
        "wins_b": w,
        "ties": t,
        "losses_b": l,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def aggregate_means(rows: list[dict], group_keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if str(r.get("feasible")).lower() not in {"true", "1"}:
            continue
        if r.get("error"):
            continue
        key = tuple(r.get(k) for k in group_keys)
        groups[key].append(r)
    out = []
    for key, items in sorted(groups.items()):
        rec = {k: v for k, v in zip(group_keys, key)}
        rec["n"] = len(items)
        for m in metrics:
            vals = []
            for it in items:
                v = it.get(m)
                if v in (None, ""):
                    continue
                try:
                    vals.append(float(v))
                except (TypeError, ValueError):
                    continue
            rec[f"mean_{m}"] = mean(vals)
        out.append(rec)
    return out


def plot_fr_pr_convergence(agg_rows: list[dict], out_path: Path) -> None:
    """Simple matplotlib convergence plot from measured aggregates."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        write_json(out_path.with_suffix(".json"), {"error": "matplotlib_missing", "rows": agg_rows})
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in zip(
        axes,
        ("mean_distance", "mean_k"),
        ("Mean distance", "Mean fleet size K"),
    ):
        for method, marker in ((METHOD_ALNS_FR, "o"), (METHOD_ALNS_PR, "s")):
            pts = [r for r in agg_rows if r.get("method") == method]
            pts = sorted(pts, key=lambda r: float(r.get("budget_s") or 0))
            if not pts:
                continue
            xs = [float(r["budget_s"]) for r in pts]
            ys = [r.get(metric) for r in pts]
            ax.plot(xs, ys, marker=marker, label=method)
        ax.set_xlabel("Budget (s)")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
