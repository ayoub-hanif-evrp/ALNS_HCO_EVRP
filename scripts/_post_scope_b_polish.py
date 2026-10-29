"""Polish Scope B tables/docs after the main run."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "reviewer_revision" / "PUBLICATION_RESULTS_FINAL"
SCOPE = ROOT / "outputs" / "reviewer_revision" / "schneider_scope_b"
TABLES = OUT / "FINAL_TABLES"
LOGS = OUT / "QC_LOGS"


def main() -> int:
    summary = json.loads((SCOPE / "summary.json").read_text(encoding="utf-8"))
    keys = [
        "tw", "metric", "baseline", "treatment", "n_instances", "mean_n_matched_seeds",
        "mean_diff", "ci95_diff_lo", "ci95_diff_hi", "mean_pct", "ci95_pct_lo", "ci95_pct_hi",
        "wins_b", "ties", "losses_b", "n_equal_k", "n_plain_lower_distance",
        "n_het_lower_distance", "n_distance_ties",
    ]
    rows = [{k: s.get(k, "") for k in keys} for s in summary["plain_vs_het"]["summaries"]]
    with (TABLES / "T_PhaseII_Plain_vs_Heterogeneous_paired.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    sb = list(csv.DictReader((TABLES / "Schneider_ScopeB_56_FR_seed0.csv").open(encoding="utf-8")))
    disc = [r for r in sb if r["repro_status"] == "DISCREPANCY"]
    with (TABLES / "Schneider_ScopeB_repro_discrepancies.csv").open("w", encoding="utf-8", newline="") as f:
        cols = [
            "instance", "feasible", "primary_feasible", "k", "primary_k", "distance",
            "primary_distance", "distance_abs_diff", "unserved", "primary_unserved", "repro_status",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in disc:
            w.writerow({c: r.get(c) for c in cols})

    ties = [r for r in sb if r.get("lex_verdict") == "FLEET_TIE"]
    with (TABLES / "Schneider_ScopeB_BKS_fleet_ties.csv").open("w", encoding="utf-8", newline="") as f:
        cols = [
            "instance", "family", "k", "bks_k", "report_sol_cost_3dp", "bks_d",
            "distance_gap_pct", "bks_source_name", "external_valid_raw", "external_valid_collapsed",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(ties, key=lambda x: float(x["distance_gap_pct"])):
            w.writerow({c: r.get(c) for c in cols})

    disc_lines = "\n".join(
        f"- {r['instance']}: ScopeB K={r['k']} vs primary {r['primary_k']}; "
        f"delta_d={float(r['distance_abs_diff']):+.4f}"
        for r in disc
    )
    extra = f"""
## Reproducibility discrepancies (5/56)
Same feasibility label for all five; stochastic ALNS search under identical seed/budget did not reproduce exact distance/K on:
{disc_lines}

These Scope-B results are retained and reported; primary archive is not overwritten.

## Fleet-tie BKS distance gaps (externally valid)
See `FINAL_TABLES/Schneider_ScopeB_BKS_fleet_ties.csv` ({len(ties)} instances). All reported gaps are positive (worse than inventory BKS distance) under 30 s seed-0 FR.

## Collapse study
Removing consecutive identical station IDs never converted an infeasible raw solution into a feasible one (n_conceal=0). Distance change under collapse is zero when consecutive stations share coordinates (same node). Raw and collapsed external validity counts match (44/56).

## Rounding evidence
Java verifier source uses unrounded Euclidean arc lengths and rounds only the **sum** to 3 decimals — compatible with our Euclidean reporting after 3-dp formatting. This does **not** prove identity with every published table that may use 2-decimal arc rounding; inventory BKS distances are compared after external validity under the verifier convention.
"""
    ana_path = OUT / "SCHNEIDER_SCOPE_B_ANALYSIS.md"
    ana = ana_path.read_text(encoding="utf-8")
    if "Reproducibility discrepancies" not in ana:
        ana = ana.rstrip() + "\n" + extra + "\n"
        ana_path.write_text(ana, encoding="utf-8")
        (SCOPE / "ANALYSIS.md").write_text(ana, encoding="utf-8")

    auth = OUT / "FINAL_VERIFIED_RESULTS_FOR_AUTHOR.md"
    t = auth.read_text(encoding="utf-8")
    old_lim = """## 7. Limitations
1. Full primary Schneider BKS gap table still PARTIAL (no route archives for all 264).
2. Declarations / manuscript / response letters: author-owned (R2.6 PARTIAL for declarations).
3. FR/PR ≠ pure recharge ablation; policy experiment is the controlled ablation.
4. Depot energy reconstructed; economics post-hoc.
5. Diagnostic 30 s Schneider tours are not competitive BKS claims."""
    new_lim = """## 7. Limitations
1. Schneider BKS remains **PARTIAL**: Scope B verifies 44/56 FR seed-0 routes externally; primary 336 still lacks full multi-seed route archives. Inventory BKS gaps (esp. fleet ties) show 30 s FR is not competitive.
2. Scope B had 5/56 non-exact reproductions vs primary seed-0 metrics (stochastic search); primary archive unchanged.
3. Declarations / manuscript / response letters: author-owned (R2.6 PARTIAL).
4. FR/PR ≠ pure recharge ablation; policy experiment is the controlled ablation.
5. Depot energy reconstructed; economics post-hoc.
6. Recommend optional **168 FR (56×3 seeds)** before 336; do not rerun PR for full-recharge verifier."""
    if old_lim in t:
        t = t.replace(old_lim, new_lim)

    old_sb = """## Schneider Scope B (new)
See `SCHNEIDER_SCOPE_B_ANALYSIS.md` and `FINAL_TABLES/Schneider_ScopeB_56_FR_seed0.csv`.
Primary 336 archive unchanged. External verifier applied to all 56 Scope-B FR exports."""
    new_sb = """## Schneider Scope B (new)
- 56 ALNS-FR × seed 0 × 30 s with full route archives; primary 336 **unchanged**.
- Reproduced exact primary metrics: **51/56**; discrepancies: **5/56** (`Schneider_ScopeB_repro_discrepancies.csv`).
- Feasible **44/56**; external Java verifier valid **44/56** (invalid = incomplete customer cover, incl. hard C65/C64/C75 cases).
- Lex BKS (valid only): fleet better/tie/worse = **0/10/34**; all 10 fleet ties have positive distance gaps vs inventory BKS.
- Analysis: `SCHNEIDER_SCOPE_B_ANALYSIS.md`. **Recommendation:** Scope B supports optional extension to **168 FR runs** (not 336 yet; no PR rerun for this verifier)."""
    if old_sb in t:
        t = t.replace(old_sb, new_sb)

    # checklist upsert for R2.BKS (before QC vars)
    checklist = OUT / "FINAL_REVIEWER_EVIDENCE_CHECKLIST.csv"
    rows_c = list(csv.DictReader(checklist.open(encoding="utf-8")))
    for row_c in rows_c:
        if row_c.get("request_id") == "R2.BKS":
            row_c["status"] = "PARTIAL"
            row_c["evidence"] = (
                "Scope B 44/56 externally valid FR seed0 with routes; "
                "Schneider_ScopeB_56_FR_seed0.csv; SCHNEIDER_SCOPE_B_ANALYSIS.md; primary 336 unchanged"
            )
    keys_c = list(rows_c[0].keys()) if rows_c else ["request_id", "request", "status", "evidence"]
    with checklist.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys_c)
        w.writeheader()
        w.writerows(rows_c)

    pytest_proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    algo_proc = subprocess.run(
        [sys.executable, "scripts/verify_no_algo_diff.py"],
        cwd=str(ROOT), capture_output=True, text=True,
    )
    (LOGS / "pytest_scope_b.txt").write_text(pytest_proc.stdout + "\n" + pytest_proc.stderr, encoding="utf-8")
    (LOGS / "verify_no_algo_diff.txt").write_text(algo_proc.stdout + "\n" + algo_proc.stderr, encoding="utf-8")
    tail = (pytest_proc.stdout.strip().splitlines()[-1:] or [""])[0]
    if "## 8. QC" in t:
        t = t.split("## 8. QC")[0] + (
            f"## 8. QC\n"
            f"- pytest: {tail} (ok={pytest_proc.returncode == 0})\n"
            f"- algo-diff vs main: ok={algo_proc.returncode == 0}\n"
            f"- Scope B summary: `QC_LOGS/schneider_scope_b_summary.json`\n"
        )
    auth.write_text(t, encoding="utf-8")

    (OUT / "SCHNEIDER_BENCHMARK_COMPARABILITY.md").write_text(
        ana_path.read_text(encoding="utf-8")
        + "\n## Relation to primary 336\n"
        "Primary protocol **336/264/72** remains the official coverage table. Scope B is an **additive** "
        "route-archive + external-verifier study on FR seed 0 only.\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "pytest_ok": pytest_proc.returncode == 0,
        "algo_ok": algo_proc.returncode == 0,
        "n_disc": len(disc),
        "n_ties": len(ties),
        "recommend_168": summary.get("recommend_168_fr"),
    }))
    return 0 if pytest_proc.returncode == 0 and algo_proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
