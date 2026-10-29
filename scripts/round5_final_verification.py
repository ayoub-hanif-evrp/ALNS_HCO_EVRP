"""Round 5: final gap-change stats, claim verification, archive obsolete files."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import reviewer_root, write_csv, write_json  # noqa: E402
from src.common.config import METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15, METHOD_PHASE2_HET, METHOD_PHASE2_NOHCO  # noqa: E402
from src.common.paired_stats import paired_instance_effects  # noqa: E402
from src.common.stats import bootstrap_mean_ci, mean  # noqa: E402


def _f(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def gap_change_30_to_300() -> dict:
    """Matched instance-level change in (PR−FR) gap from 30s to 300s."""
    rows = load_csv(reviewer_root() / "fr_pr_convergence_primary288" / "runs.csv")
    assert len(rows) == 288
    out_rows = []
    summary = []
    for tw in ("original", "relaxed"):
        for metric in ("k", "distance"):
            g30 = {
                e["instance"]: e
                for e in paired_instance_effects(
                    rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR,
                    metric=metric, tw=tw, budget_s=30.0, require_budget=True,
                )
            }
            g300 = {
                e["instance"]: e
                for e in paired_instance_effects(
                    rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR,
                    metric=metric, tw=tw, budget_s=300.0, require_budget=True,
                )
            }
            common = sorted(set(g30) & set(g300))
            assert len(common) == 12
            deltas = []
            for inst in common:
                d30 = g30[inst]["diff_b_minus_a"]
                d300 = g300[inst]["diff_b_minus_a"]
                # change in gap: (PR-FR)_300 - (PR-FR)_30
                # negative means PR's relative disadvantage shrinks (or advantage grows)
                delta = d300 - d30
                deltas.append(delta)
                out_rows.append({
                    "tw": tw, "metric": metric, "instance": inst,
                    "gap_PR_minus_FR_30s": d30,
                    "gap_PR_minus_FR_300s": d300,
                    "gap_change_300_minus_30": delta,
                    "n_matched_seeds_30": g30[inst]["n_matched_seeds"],
                    "n_matched_seeds_300": g300[inst]["n_matched_seeds"],
                    "note": "Seeds averaged within each budget before instance-level delta; TW analyzed separately.",
                })
            lo, hi = bootstrap_mean_ci(deltas, seed=42)
            summary.append({
                "tw": tw, "metric": metric,
                "n_instances": len(deltas),
                "mean_gap_change_300_minus_30": mean(deltas),
                "ci95_lo": lo, "ci95_hi": hi,
                "interpretation": (
                    "Positive mean gap change: PR−FR increases (PR relatively worse on lower-is-better metric). "
                    "Negative: PR−FR decreases (PR relatively better / less worse)."
                ),
            })
    # Lex conditional: distance gap change only on instances with K-tie at both budgets or report separately
    lex_rows = []
    for tw in ("original", "relaxed"):
        k30 = {e["instance"]: e for e in paired_instance_effects(
            rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR, metric="k", tw=tw, budget_s=30.0)}
        k300 = {e["instance"]: e for e in paired_instance_effects(
            rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR, metric="k", tw=tw, budget_s=300.0)}
        d30 = {e["instance"]: e for e in paired_instance_effects(
            rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR, metric="distance", tw=tw, budget_s=30.0)}
        d300 = {e["instance"]: e for e in paired_instance_effects(
            rows, method_a=METHOD_ALNS_FR, method_b=METHOD_ALNS_PR, metric="distance", tw=tw, budget_s=300.0)}
        for inst in sorted(set(k30) & set(k300) & set(d30) & set(d300)):
            tie30 = abs(k30[inst]["diff_b_minus_a"]) < 1e-9
            tie300 = abs(k300[inst]["diff_b_minus_a"]) < 1e-9
            lex_rows.append({
                "tw": tw, "instance": inst,
                "K_tie_at_30": tie30, "K_tie_at_300": tie300,
                "K_tie_both": tie30 and tie300,
                "distance_gap_30": d30[inst]["diff_b_minus_a"],
                "distance_gap_300": d300[inst]["diff_b_minus_a"],
                "distance_gap_change": d300[inst]["diff_b_minus_a"] - d30[inst]["diff_b_minus_a"],
                "k_gap_30": k30[inst]["diff_b_minus_a"],
                "k_gap_300": k300[inst]["diff_b_minus_a"],
            })
    out = reviewer_root() / "final_stats"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "table_gap_change_30_to_300_by_instance.csv", out_rows)
    write_csv(out / "table_gap_change_30_to_300_summary.csv", summary)
    write_csv(out / "table_gap_change_lex_conditional.csv", lex_rows)
    write_json(out / "gap_change_summary.json", summary)
    return {"summary": summary, "n_lex_rows": len(lex_rows)}


def verify_claims() -> dict:
    hanif = load_csv(reviewer_root() / "hanif_main_seeds012" / "runs.csv")
    assert len(hanif) == 2520
    n_inst = len({r["instance"] for r in hanif})
    n_tw = len({r["tw"] for r in hanif})
    assert n_inst == 60 and n_tw == 2

    # PhaseII-Het vs PhaseI aggregate identity
    idx = {(r["instance"], r["tw"], int(float(r["seed"])), r["method"]): r for r in hanif}
    n = n_same = 0
    for inst, tw, seed in sorted({(r["instance"], r["tw"], int(float(r["seed"]))) for r in hanif}):
        p1 = idx.get((inst, tw, seed, METHOD_PHASE1_PR15))
        het = idx.get((inst, tw, seed, METHOD_PHASE2_HET))
        if not p1 or not het:
            continue
        n += 1
        same = (
            abs(_f(p1, "distance") - _f(het, "distance")) <= 1e-6
            and abs(_f(p1, "charged_energy") - _f(het, "charged_energy")) <= 1e-6
            and abs(_f(p1, "k") - _f(het, "k")) <= 1e-9
        )
        if same:
            n_same += 1
    assert n == 360
    assert n_same == 350

    # Energy coverage
    for r in hanif:
        assert _f(r, "residual_battery_at_depot") is not None
        assert _f(r, "depot_replenishment_energy") is not None

    # Iteration throughput FR vs PR on primary 288
    conv = load_csv(reviewer_root() / "fr_pr_convergence_primary288" / "runs.csv")
    iters = defaultdict(list)
    for r in conv:
        iters[(r["method"], float(r["budget_s"]))].append(_f(r, "alns_iterations"))
    iter_means = {
        f"{m}|{b:g}s": mean(vs) for (m, b), vs in sorted(iters.items())
    }

    sch = load_csv(reviewer_root() / "schneider" / "table_main30_feasibility.csv")
    assert len(sch) == 336
    n_feas = sum(1 for r in sch if str(r["feasible"]).lower() in {"true", "1"})
    assert n_feas == 264

    paired = load_csv(reviewer_root() / "schneider" / "paired_main30_summaries.csv")
    dist_row = next(r for r in paired if r["metric"] == "distance")
    assert int(float(dist_row["n_instances"])) == 44

    hco = json.loads((reviewer_root() / "hco_contribution_audit" / "summary.json").read_text(encoding="utf-8"))
    assert hco["n_identical_aggregate_metrics_vs_phase1"] == 350

    claims = {
        "hanif_rows": 2520,
        "hanif_instances": 60,
        "hanif_tw": 2,
        "phase2_het_identical_aggregate_vs_phase1": n_same,
        "phase2_het_compared": n,
        "fraction_identical": n_same / n,
        "schneider_attempts": 336,
        "schneider_feasible": n_feas,
        "schneider_infeasible": 336 - n_feas,
        "schneider_paired_instances": 44,
        "convergence_primary_rows": 288,
        "mean_alns_iterations_by_method_budget": iter_means,
        "verified": True,
    }
    write_json(reviewer_root() / "final_stats" / "claim_verification.json", claims)
    return claims


def archive_obsolete() -> list[str]:
    """Move conflicting preliminary status artifacts into an archive folder (keep copies readable)."""
    root = reviewer_root()
    arch = root / "archive_obsolete_preliminary"
    arch.mkdir(parents=True, exist_ok=True)
    moved = []
    # Mark obsolete analysis_status that may conflict with primary288 status
    candidates = [
        root / "fr_pr_convergence" / "analysis_status.json",
        root / "fr_pr_convergence" / "pilot_status.json",
        root / "fr_pr_convergence" / "full_status.json",
        root / "alns_hco_absolute" / "pilot_status.json",
        root / "alns_hco_absolute" / "full_status.json",
    ]
    readme = {
        "purpose": "Obsolete/preliminary status files superseded by Round 4–5 canonical outputs",
        "canonical_convergence": "fr_pr_convergence_primary288/",
        "canonical_hanif_main": "hanif_main_seeds012/",
        "canonical_schneider": "schneider/main30_status.json",
        "seed2_supplementary": "fr_pr_convergence_seed2_supplementary/",
    }
    for src in candidates:
        if not src.is_file():
            continue
        dest = arch / f"{src.parent.name}__{src.name}"
        if not dest.exists():
            shutil.copy2(src, dest)
            moved.append(str(dest.relative_to(root)))
        # Write OBSOLETE marker beside original
        marker = src.with_suffix(src.suffix + ".OBSOLETE.txt")
        marker.write_text(
            f"OBSOLETE preliminary file. Canonical sources:\n{json.dumps(readme, indent=2)}\n",
            encoding="utf-8",
        )
        moved.append(str(marker.relative_to(root)))
    write_json(arch / "README.json", readme)
    return moved


def build_manifest() -> list[dict]:
    root = reviewer_root()
    entries = [
        ("Table", "Convergence means by TW/budget", "fr_pr_convergence_primary288/table_convergence_by_tw.csv"),
        ("Table", "FR–PR paired summaries (seeds 0–1)", "fr_pr_convergence_primary288/paired_summaries.csv"),
        ("Table", "FR–PR lex gaps by budget", "fr_pr_convergence_primary288/table_fr_pr_gaps_by_budget_lex.csv"),
        ("Table", "Gap change 30→300s summary", "final_stats/table_gap_change_30_to_300_summary.csv"),
        ("Table", "Gap change 30→300s by instance", "final_stats/table_gap_change_30_to_300_by_instance.csv"),
        ("Figure", "FR–PR convergence (all TW)", "fr_pr_convergence_primary288/fig_fr_pr_convergence.png"),
        ("Figure", "FR–PR convergence original TW", "fr_pr_convergence_primary288/fig_fr_pr_convergence_original.png"),
        ("Figure", "FR–PR convergence relaxed TW", "fr_pr_convergence_primary288/fig_fr_pr_convergence_relaxed.png"),
        ("Table", "Hanif main means by method", "hanif_main_seeds012/table_means_by_method.csv"),
        ("Table", "Hanif energy coverage", "hanif_main_seeds012/table_energy_coverage.csv"),
        ("Table", "HCO contribution counts", "hco_contribution_audit/summary_counts.csv"),
        ("Table", "Schneider 30s feasibility", "schneider/table_main30_feasibility.csv"),
        ("Table", "Schneider 30s paired FR/PR", "schneider/paired_main30_summaries.csv"),
        ("Table", "Economic sensitivity scenarios", "economic_sensitivity/table_means_selected_scenarios.csv"),
        ("Table", "Hanif benchmark characteristics", "benchmark_characteristics/table_hanif_instances.csv"),
        ("Table", "Schneider benchmark characteristics", "benchmark_characteristics/table_schneider_instances.csv"),
        ("Figure", "Hanif main original TW", "hanif_main_seeds012/fig_hanif_main_original.png"),
        ("Figure", "Hanif main relaxed TW", "hanif_main_seeds012/fig_hanif_main_relaxed.png"),
        ("Note", "HCO-Detour verification", "HCO_DETOUR_VERIFICATION.md"),
        ("Note", "Schneider neutral charging", "schneider/NEUTRAL_CHARGING.md"),
        ("Paper fig attribution", "Fig4/Fig5 = NoHCO→HCO (existing)", "outputs/figures/final_figure_manifest.md"),
    ]
    rows = []
    for kind, title, rel in entries:
        p = root / rel if not rel.startswith("outputs/") else ROOT / rel
        rows.append({
            "kind": kind, "title": title, "path": rel,
            "exists": p.is_file(),
            "canonical": True,
        })
    write_csv(root / "FINAL_TABLE_FIGURE_MANIFEST.csv", rows)
    return rows


def main() -> int:
    (reviewer_root() / "final_stats").mkdir(parents=True, exist_ok=True)
    gaps = gap_change_30_to_300()
    claims = verify_claims()
    moved = archive_obsolete()
    manifest = build_manifest()
    payload = {
        "gap_change": gaps["summary"],
        "claims": claims,
        "archived_markers": moved,
        "manifest_n": len(manifest),
        "manifest_missing": [m["path"] for m in manifest if not m["exists"]],
    }
    write_json(reviewer_root() / "final_stats" / "round5_verification.json", payload)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
