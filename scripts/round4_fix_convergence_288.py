"""Regenerate primary 288-run convergence analysis (seeds 0–1 only).

Seed-2 rows are written to a separately labeled supplementary folder.
Does not re-solve ALNS.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import (  # noqa: E402
    BALANCED_12,
    aggregate_means,
    plot_fr_pr_convergence,
    reviewer_root,
    write_csv,
    write_json,
)
from src.common.config import METHOD_ALNS_FR, METHOD_ALNS_PR  # noqa: E402
from src.common.paired_stats import paired_instance_effects, summarize_paired  # noqa: E402

PRIMARY_SEEDS = (0, 1)
BUDGETS = (30.0, 120.0, 300.0)
METRICS = (
    "k", "distance", "charged_energy", "propulsion_energy",
    "enroute_purchased_energy", "depot_replenishment_energy",
    "residual_battery_at_depot", "charging_visits",
    "alns_iterations", "solver_runtime_s", "end_to_end_runtime_s",
)


def _load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _ok(r: dict) -> bool:
    if r.get("error"):
        return False
    return str(r.get("feasible")).lower() in {"true", "1"}


def _is_primary(r: dict) -> bool:
    try:
        seed = int(float(r["seed"]))
    except (TypeError, ValueError, KeyError):
        return False
    return (
        r.get("instance") in set(BALANCED_12)
        and seed in PRIMARY_SEEDS
        and r.get("method") in (METHOD_ALNS_FR, METHOD_ALNS_PR)
        and _ok(r)
    )


def _is_seed2_supp(r: dict) -> bool:
    try:
        seed = int(float(r["seed"]))
    except (TypeError, ValueError, KeyError):
        return False
    return (
        r.get("instance") in set(BALANCED_12)
        and seed == 2
        and r.get("method") in (METHOD_ALNS_FR, METHOD_ALNS_PR)
        and _ok(r)
    )


def assert_primary_cells(rows: list[dict]) -> dict:
    cells: dict[tuple, int] = defaultdict(int)
    for r in rows:
        cells[(r["method"], r["tw"], float(r["budget_s"]))] += 1
    bad = {f"{k}": v for k, v in cells.items() if v != 24}
    assert len(rows) == 288, f"expected 288 primary rows, got {len(rows)}"
    assert len(cells) == 12, f"expected 12 cells, got {len(cells)}"
    assert not bad, f"cells not size 24: {bad}"
    return {"n_rows": len(rows), "n_cells": len(cells), "cell_size": 24, "ok": True}


def lex_compare_gaps(rows: list[dict]) -> list[dict]:
    """FR–PR gaps by budget with fleet-size-first interpretation notes."""
    out = []
    for budget in BUDGETS:
        for tw in ("original", "relaxed"):
            eff_k = paired_instance_effects(
                rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR,
                metric="k", tw=tw, budget_s=budget, require_budget=True,
            )
            eff_d = paired_instance_effects(
                rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR,
                metric="distance", tw=tw, budget_s=budget, require_budget=True,
            )
            sk = summarize_paired(eff_k)
            sd = summarize_paired(eff_d)
            # Instance-level: only interpret distance when K does not worsen for PR
            n_k_equal = n_pr_better_k = n_fr_better_k = 0
            n_dist_when_k_tie_pr_wins = n_dist_when_k_tie_fr_wins = 0
            by_inst = {e["instance"]: e for e in eff_k}
            for ed in eff_d:
                ek = by_inst.get(ed["instance"])
                if not ek:
                    continue
                dk = ek["diff_b_minus_a"]
                dd = ed["diff_b_minus_a"]
                if dk is None or dd is None:
                    continue
                if abs(dk) < 1e-9:
                    n_k_equal += 1
                    if dd < 0:
                        n_dist_when_k_tie_pr_wins += 1
                    elif dd > 0:
                        n_dist_when_k_tie_fr_wins += 1
                elif dk < 0:
                    n_pr_better_k += 1
                else:
                    n_fr_better_k += 1
            out.append({
                "budget_s": budget,
                "tw": tw,
                "n_instances": sk["n_instances"],
                "mean_n_matched_seeds": sk["mean_n_matched_seeds"],
                "mean_diff_k_PR_minus_FR": sk["mean_diff"],
                "ci95_k_lo": sk["ci95_diff_lo"],
                "ci95_k_hi": sk["ci95_diff_hi"],
                "mean_diff_distance_PR_minus_FR": sd["mean_diff"],
                "ci95_distance_lo": sd["ci95_diff_lo"],
                "ci95_distance_hi": sd["ci95_diff_hi"],
                "n_instances_K_tie": n_k_equal,
                "n_PR_better_K": n_pr_better_k,
                "n_FR_better_K": n_fr_better_k,
                "n_PR_better_distance_when_K_tie": n_dist_when_k_tie_pr_wins,
                "n_FR_better_distance_when_K_tie": n_dist_when_k_tie_fr_wins,
                "lex_note": (
                    "Fleet size is primary. Distance comparisons are unconditional means "
                    "plus conditional counts when K is tied; lower distance alone is not "
                    "unconditional superiority when fleet sizes differ."
                ),
            })
    return out


def analyze_protocol(rows: list[dict], out_dir: Path, *, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "runs.csv", rows)
    status = {
        "label": label,
        "protocol": "288_primary_seeds_0_1" if "primary" in label else label,
    }
    if "primary" in label:
        status.update(assert_primary_cells(rows))
    else:
        status["n_rows"] = len(rows)
    write_csv(out_dir / "table_convergence_by_tw.csv", aggregate_means(rows, ("method", "budget_s", "tw"), METRICS))
    write_csv(out_dir / "table_convergence.csv", aggregate_means(rows, ("method", "budget_s"), METRICS))
    write_csv(out_dir / "table_effort.csv", aggregate_means(
        rows, ("method", "budget_s", "tw"), ("alns_iterations", "solver_runtime_s", "end_to_end_runtime_s"),
    ))
    plot_fr_pr_convergence(
        aggregate_means(rows, ("method", "budget_s"), METRICS),
        out_dir / "fig_fr_pr_convergence.png",
    )
    for tw in sorted({r["tw"] for r in rows}):
        plot_fr_pr_convergence(
            aggregate_means([r for r in rows if r["tw"] == tw], ("method", "budget_s", "tw"), METRICS),
            out_dir / f"fig_fr_pr_convergence_{tw}.png",
            tw=tw,
        )

    paired, summaries = [], []
    for budget in BUDGETS:
        for tw in sorted({r["tw"] for r in rows}):
            for metric in ("distance", "k", "propulsion_energy", "enroute_purchased_energy", "alns_iterations", "solver_runtime_s"):
                effects = paired_instance_effects(
                    rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR,
                    metric=metric, tw=tw, budget_s=budget, require_budget=True,
                )
                # Assert matched seeds == 2 for primary
                if "primary" in label:
                    for e in effects:
                        assert int(e["n_matched_seeds"]) == 2, e
                paired.extend(effects)
                summaries.append({
                    "budget_s": budget, "tw": tw, "metric": metric, **summarize_paired(effects),
                })
    write_csv(out_dir / "paired_instance_effects.csv", paired)
    write_csv(out_dir / "paired_summaries.csv", summaries)
    gaps = lex_compare_gaps(rows)
    write_csv(out_dir / "table_fr_pr_gaps_by_budget_lex.csv", gaps)
    status["n_paired_summaries"] = len(summaries)
    status["mean_n_matched_seeds_max"] = max(
        (float(s["mean_n_matched_seeds"] or 0) for s in summaries), default=None,
    )
    write_json(out_dir / "status.json", status)
    return status


def main() -> int:
    src = reviewer_root() / "fr_pr_convergence" / "runs.csv"
    # Prefer rebuilding from checkpoints if available
    ckpt = reviewer_root() / "fr_pr_convergence" / "checkpoints"
    rows = []
    if ckpt.is_dir() and any(ckpt.glob("*.json")):
        for p in sorted(ckpt.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            r = rec.get("row")
            if r:
                rows.append(r)
    if not rows and src.is_file():
        rows = _load_rows(src)

    primary = [r for r in rows if _is_primary(r)]
    # Deduplicate by job key keeping last
    uniq = {}
    for r in primary:
        key = (r["method"], r["instance"], r["tw"], int(float(r["seed"])), float(r["budget_s"]))
        uniq[key] = r
    primary = list(uniq.values())
    assert len(primary) == 288, f"primary after dedupe={len(primary)}"

    seed2 = []
    uniq2 = {}
    for r in rows:
        if _is_seed2_supp(r):
            key = (r["method"], r["instance"], r["tw"], 2, float(r["budget_s"]))
            uniq2[key] = r
    seed2 = list(uniq2.values())

    primary_dir = reviewer_root() / "fr_pr_convergence_primary288"
    supp_dir = reviewer_root() / "fr_pr_convergence_seed2_supplementary"
    st = analyze_protocol(primary, primary_dir, label="primary_288_seeds01")
    # Also overwrite main fr_pr_convergence analysis artifacts with primary-only
    st_main = analyze_protocol(primary, reviewer_root() / "fr_pr_convergence", label="primary_288_seeds01")
    write_json(primary_dir / "ASSERTIONS.json", {
        "n_primary": 288,
        "cell_size": 24,
        "matched_seeds_required": 2,
        "mean_n_matched_seeds_max": st["mean_n_matched_seeds_max"],
        "passed": st["mean_n_matched_seeds_max"] == 2.0,
    })
    if seed2:
        write_csv(supp_dir / "runs.csv", seed2)
        write_csv(supp_dir / "table_means.csv", aggregate_means(seed2, ("method", "budget_s", "tw"), METRICS))
        write_json(supp_dir / "README.json", {
            "label": "seed2_supplementary_NOT_primary_288",
            "n_rows": len(seed2),
            "note": "Extra seed-2 runs retained for transparency; excluded from primary 288 protocol.",
        })
    print(json.dumps({"primary": st_main, "seed2_n": len(seed2)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
