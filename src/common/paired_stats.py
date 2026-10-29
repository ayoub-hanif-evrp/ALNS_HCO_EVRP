"""Instance-level paired statistics with matched seeds and budgets.

Protocol:
1. Keep only feasible, error-free rows.
2. Match methods on (instance, tw, seed, budget_s).
3. For each (instance, tw, budget_s), average the metric over seeds that
   exist for *both* methods (matched seeds only).
4. Bootstrap CIs over the resulting instance-level effects.
"""
from __future__ import annotations

from collections import defaultdict

from src.common.stats import bootstrap_mean_ci, mean, pct_change, wins_ties_losses


def _f(row: dict, key: str):
    v = row.get(key)
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _budget_key(row: dict) -> float | None:
    for k in ("budget_s", "budget"):
        v = _f(row, k)
        if v is not None:
            return float(v)
    return None


def _is_ok(row: dict) -> bool:
    if row.get("error"):
        return False
    return str(row.get("feasible")).lower() in {"true", "1"}


def paired_instance_effects(
    rows: list[dict],
    *,
    method_a: str,
    method_b: str,
    metric: str,
    tw: str | None = None,
    budget_s: float | None = None,
    require_budget: bool = True,
) -> list[dict]:
    """Paired B−A effects after matching instance/TW/seed/budget and averaging seeds."""
    # Index: (instance, tw, budget, seed, method) -> metric value
    cells: dict[tuple, float] = {}
    for r in rows:
        if not _is_ok(r):
            continue
        if tw is not None and r.get("tw") != tw:
            continue
        b = _budget_key(r)
        if require_budget and b is None:
            continue
        if budget_s is not None:
            if b is None or abs(float(b) - float(budget_s)) > 1e-9:
                continue
        if r.get("method") not in (method_a, method_b):
            continue
        val = _f(r, metric)
        if val is None:
            continue
        seed = r.get("seed")
        try:
            seed_i = int(seed)
        except (TypeError, ValueError):
            continue
        key = (str(r["instance"]), str(r.get("tw")), None if b is None else float(b), seed_i, str(r["method"]))
        cells[key] = float(val)

    # Collect matched seeds per (instance, tw, budget)
    groups: dict[tuple, dict[str, dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for (inst, twv, b, seed_i, method), val in cells.items():
        groups[(inst, twv, b)][method][seed_i] = val

    effects = []
    for (inst, twv, b), methods in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] or -1)):
        a_map = methods.get(method_a) or {}
        b_map = methods.get(method_b) or {}
        matched = sorted(set(a_map) & set(b_map))
        if not matched:
            continue
        a_vals = [a_map[s] for s in matched]
        b_vals = [b_map[s] for s in matched]
        a_m, b_m = mean(a_vals), mean(b_vals)
        effects.append({
            "instance": inst,
            "tw": twv,
            "budget_s": b,
            "method_a": method_a,
            "method_b": method_b,
            "metric": metric,
            "n_matched_seeds": len(matched),
            "matched_seeds": matched,
            "a_mean": a_m,
            "b_mean": b_m,
            "diff_b_minus_a": None if a_m is None or b_m is None else b_m - a_m,
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
    # Lower-is-better win count for B vs A
    w = t = l = 0
    if a and b and len(a) == len(b):
        w, t, l = wins_ties_losses(b, a)
    return {
        "n_instances": len(effects),
        "mean_n_matched_seeds": mean([float(e["n_matched_seeds"]) for e in effects]),
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


def protocol_completion(
    completed_job_ids: set[str],
    planned_jobs: list[dict],
    *,
    job_id_fn,
) -> dict:
    """Distinguish completed subset vs complete planned protocol."""
    planned_ids = [job_id_fn(j) for j in planned_jobs]
    planned_set = set(planned_ids)
    done = sorted(planned_set & completed_job_ids)
    missing = sorted(planned_set - completed_job_ids)
    return {
        "n_planned": len(planned_set),
        "n_completed": len(done),
        "n_missing": len(missing),
        "complete_protocol": len(missing) == 0,
        "missing_job_ids": missing,
        "completion_rate": (len(done) / len(planned_set)) if planned_set else 0.0,
    }
