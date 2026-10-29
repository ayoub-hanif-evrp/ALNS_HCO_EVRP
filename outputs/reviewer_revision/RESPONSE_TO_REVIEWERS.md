# Response to Reviewers

**Manuscript:** ALNS-HCO for EVRPTW (heterogeneous charging / partial recharging)  
**Revision branch / evidence root:** `reviewer-revision-2026` → `outputs/reviewer_revision/`  
**Policy:** Every claim below cites verified tables, figures, or code. Where evidence is incomplete, the limitation is stated explicitly — we do **not** mark a request as fully satisfied without support.

**Algorithm constraint:** No changes were made to ALNS/HCO architecture, objectives, operators, acceptance, charging, or feasibility logic relative to `main` (verified by `scripts/verify_no_algo_diff.py` on protected decision-logic files).

---

# Reviewer 1

## R1.1 Equal-budget FR vs PR and longer-budget / convergence behaviour

**Reviewer concern.** Full vs partial recharging should be compared under equal computational budgets, with evidence on whether conclusions change at longer runtimes.

**Correction / new evidence.**  
- Main Hanif FR/PR comparison: equal 30 s wall-clock budgets (`hanif_main_seeds012/`).  
- Multi-budget protocol: 12 instances × 2 TW × FR/PR × {30,120,300} s × seeds {0,1} = **288** primary runs (`fr_pr_convergence_primary288/`).  
- Seed-2 extras excluded from primary inference (`fr_pr_convergence_seed2_supplementary/`).  
- Matched instance-level change in the FR–PR gap from 30→300 s with 95% bootstrap CIs, **separately** for Original and Relaxed TW (`final_stats/table_gap_change_30_to_300_summary.csv`).  
- On Original TW, mean \(\Delta(\mathrm{PR}-\mathrm{FR})\) for fleet size is +1.17 [0.08, 2.33]: longer budgets do not close PR’s relative \(K\) disadvantage on that panel. Distance gap-change CIs include 0 on both TW panels.

**Locations.** `fr_pr_convergence_primary288/paired_summaries.csv`, `table_fr_pr_gaps_by_budget_lex.csv`, `fig_fr_pr_convergence*.png`, `final_stats/`, `ROUND5_FINAL_REPORT.md` §2; manuscript text: `MANUSCRIPT_REVISION_GUIDE.md` §C.

**Remaining limitation.** Convergence panel is a 12-instance subset (supplementary to the 60-instance 30 s main study), seeds {0,1} only. Lex-conditional distance CIs for \(K\)-ties at both budgets have \(n\in\{1,2\}\) — descriptive only.

---

## R1.2 Lexicographic objective (fleet size before distance)

**Reviewer concern.** Distance improvements must not be claimed when fleet sizes differ under a fleet-first lexicographic objective.

**Correction / new evidence.**  
Primary reporting now emphasises \(\Delta K\) first. Distance is reported with explicit conditional notes and win/tie/loss counts for \(K\) (`table_fr_pr_gaps_by_budget_lex.csv`; Round 4 report). Gap-change tables include a lex-conditional companion file (`table_gap_change_lex_conditional.csv`).

**Locations.** `ROUND4_REPORT.md` §1; `MANUSCRIPT_REVISION_GUIDE.md` §A, §C.

**Remaining limitation.** Formal hypothesis tests restricted to the \(K\)-tie subset at long budgets remain underpowered on the 12-instance panel.

---

## R1.3 Related TW variants vs independent replications

**Reviewer concern.** Original and Relaxed TW must not be treated as independent stochastic replicates of the same base instance.

**Correction / new evidence.**  
All paired analyses condition on TW; Original and Relaxed are never pooled as independent base instances. Manuscript wording states they are related benchmark variants (`MANUSCRIPT_REVISION_GUIDE.md` §B). Bootstrap units are instances within a TW panel; seeds are averaged within instance.

**Locations.** `src/common/paired_stats.py`; primary288 / Hanif paired pipelines; `ROUND5_FINAL_REPORT.md` §1–2.

**Remaining limitation.** None for the statistical protocol; authors must ensure the submitted manuscript text matches this wording.

---

## R1.4 Energy accounting / total electricity interpretation

**Reviewer concern.** Clarify propulsion, en-route purchased energy, depot replenishment, and residual return energy; avoid conflating observed charging expense with total electricity cost.

**Correction / new evidence.**  
Energy identity and labelled reconstruction applied to all 2520 Hanif main rows; coverage tables confirm residual and depot fields present (`hanif_main_seeds012/table_energy_coverage.csv`, `energy_reconstruction_note.json`). Manuscript text separates \(E_{\mathrm{enroute}}\), reconstructed \(E_{\mathrm{depot}}\), and expense (`MANUSCRIPT_REVISION_GUIDE.md` §D).

**Locations.** `ROUND4_REPORT.md` §3; Round 5 claim verification.

**Remaining limitation.** Depot energy is reconstructed at aggregate level when route SOC traces are missing — labelled, not claimed as directly observed SOC telemetry.

---

## R1.5 HCO contribution vs Phase-I / NoHCO controls

**Reviewer concern.** Distinguish Phase-II search improvements, NoHCO improvements, and genuine HCO effects; avoid overstating HCO impact.

**Correction / new evidence.**  
Matched audit (n=360): PhaseII-Heterogeneous identical to Phase I on \(K\), distance, and charged energy in **350/360** cases; PhaseI→NoHCO shows larger search-driven improvement counts; NoHCO→Het mostly unchanged with some worsenings (`hco_contribution_audit/summary_counts.csv`, `summary.json`). ε=0 distance gate documented without algorithm change (`phase2_het_diagnosis/DIAGNOSIS.md`).

**Locations.** `ROUND4_REPORT.md` §4; `MANUSCRIPT_REVISION_GUIDE.md` §E; paper Figures 4–5 = NoHCO→HCO (`outputs/figures/final_figure_manifest.md`).

**Remaining limitation.** Route-sequence identity PhaseII-Het ≡ Phase I is **unproven** (no route checksums in publication CSV). Aggregate identity only.

---

## R1.6 HCO-Detour description accuracy

**Reviewer concern.** Manuscript description of Detour mode must match implementation.

**Correction / new evidence.**  
Verification note: shortlist uses detour; replacement tournament is **distance-led geometric**, not pure detour throughout (`HCO_DETOUR_VERIFICATION.md`; tests `tests/test_hco_detour_science.py`). Ready replacement wording in `MANUSCRIPT_REVISION_GUIDE.md` §F.

**Locations.** As above.

**Remaining limitation.** Authors must edit the manuscript prose; code was intentionally **not** changed to match older wording.

---

## R1.7 FR vs PR implementation / iteration throughput

**Reviewer concern.** Acknowledge that FR and PR searches are not identical engines under equal wall-clock time.

**Correction / new evidence.**  
Primary288 mean iterations: FR ≫ PR at every budget (e.g. ~44 398 vs ~12 156 at 300 s) — `claim_verification.json` → `mean_alns_iterations_by_method_budget`. Manuscript note in §C of the guide.

**Remaining limitation.** We do not re-tune operators to equalise iterations; equal wall-clock remains the experimental contract.

---

## R1.8 Partial charging vs nonlinear charging speed

**Reviewer concern.** Do not attribute benefits of nonlinear charging-speed curves to linear partial recharging.

**Correction / new evidence.**  
Clarifying sentence added for manuscript (`MANUSCRIPT_REVISION_GUIDE.md` §D): model uses constant rate; PR effect is just-enough quantity.

**Remaining limitation.** Nonlinear \(r(\mathrm{SoC})\) is outside the study — no new experiment claimed.

---

## R1.9 Experimental setup transparency (hardware, seeds, budgets, benchmarks)

**Reviewer concern.** Report hardware, seeds, budgets, and benchmark characteristics clearly.

**Correction / new evidence.**  
Setup text (`MANUSCRIPT_REVISION_GUIDE.md` §A); benchmark tables (`benchmark_characteristics/table_hanif_instances.csv`, `table_schneider_instances.csv`); frozen config in `outputs/final_experiment_report.md` / `outputs/config/frozen_experiment_config.json`.

**Remaining limitation.** Exact wall-clock of every historical publication job is not re-logged beyond `solver_runtime_s` columns in CSVs.

---

# Reviewer 2

## R2.1 Schneider et al. validation coverage and honesty about failures

**Reviewer concern.** External Schneider validation should cover the intended instance set and report failures transparently.

**Correction / new evidence.**  
All **56** large instances retained; **336** attempts; **264** feasible; **72** infeasible; performance comparisons conditional on **44** both-feasible instances (`schneider/table_main30_feasibility.csv`, `paired_main30_summaries.csv`). We explicitly **do not** claim all 56 were solved.

**Locations.** `ROUND4_REPORT.md` §2; `ROUND5_FINAL_REPORT.md` §3; `MANUSCRIPT_REVISION_GUIDE.md` §G.

**Remaining limitation.** 30 s budget only for Schneider primary protocol; longer Schneider budgets were not re-run in Round 5 (by design).

---

## R2.2 Feasibility failures: search vs mathematical infeasibility

**Reviewer concern.** Distinguish solver inability from proven instance infeasibility.

**Correction / new evidence.**  
Round-3 solo `replay_solution` bug corrected; Round-4 uses `replay_route`. Station-insertion probes succeed for customers failing direct trips on all 12 hard instances → classified as **construction/search failure**, not proven mathematical infeasibility (`feasibility_investigation_round4.json`).

**Locations.** `ROUND4_REPORT.md` §2; disclaimer in investigation JSON.

**Remaining limitation.** Full MIP infeasibility certificates / alternative distance-rounding studies were **not** performed — we do not claim a proof of feasibility of published optima under every convention.

---

## R2.3 Heterogeneous price/waiting claims on Schneider

**Reviewer concern.** Schneider does not provide heterogeneous prices/waits; do not over-claim.

**Correction / new evidence.**  
`schneider/NEUTRAL_CHARGING.md`: homogeneous rate; wait disabled; neutral price placeholder. Explicit prohibition on using Schneider to validate heterogeneous price/waiting benefits. Manuscript §G mirrors this.

**Remaining limitation.** None if manuscript text is updated; Schneider remains FR/PR routing validation only.

---

## R2.4 Economic interpretation, depot tariffs, and wages

**Reviewer concern.** Clarify economic metrics; address sensitivity to wages / depot energy prices without silently changing the optimiser.

**Correction / new evidence.**  
Post-hoc scenarios only (`economic_sensitivity/ASSUMPTIONS.json`, `table_means_selected_scenarios.csv`): depot tariffs and wages applied to frozen routes; labelled `post_hoc_sensitivity_NOT_optimized`. Observed public charging expense kept separate from hypothetical depot×tariff and wage×duration totals.

**Locations.** `ROUND4_REPORT.md` §5; `MANUSCRIPT_REVISION_GUIDE.md` §H.

**Remaining limitation.** **Satisfied as sensitivity analysis only.** We do **not** claim the solver optimises wage-inclusive or depot-tariff objectives — that would require algorithm/objective changes, which were forbidden.

---

## R2.5 Benchmark description tables

**Reviewer concern.** Provide clearer benchmark characteristics (families, capacities, charging assumptions, TW regimes).

**Correction / new evidence.**  
`benchmark_characteristics/table_hanif_instances.csv`, `table_schneider_instances.csv` (and family summaries where present).

**Remaining limitation.** Authors should embed condensed versions in the paper or supplement.

---

## R2.6 Data availability and declarations

**Reviewer concern.** Provide data availability and required declaration headings.

**Correction / new evidence.**  
Ready text: `MANUSCRIPT_REVISION_GUIDE.md` §J–K (Data Availability; Competing interest; Funding placeholder; CRediT placeholder; Acknowledgements). Canonical path manifest: `FINAL_TABLE_FIGURE_MANIFEST.csv`.

**Remaining limitation.** Funding / CRediT must be filled by authors — not invented here. Manuscript file not in repo; paste required.

---

## R2.7 Figure / caption / attribution corrections (esp. Figure 5)

**Reviewer concern.** Ensure figures, units, and captions match the scientific comparison.

**Correction / new evidence.**  
Verified manifesto: Figure 4 and Figure 5 are **PhaseII-NoHCO → PhaseII-Heterogeneous** (`outputs/figures/final_figure_manifest.md`). Captions and source CSVs listed there. Convergence figures regenerated from primary288 only.

**Remaining limitation.** Authors must replace any in-manuscript mis-attribution of Figure 5 to FR/PR or PhaseI→PhaseII.

---

# Cross-cutting validation (both reviewers)

| Check | Evidence | Status |
|---|---|---|
| Protected algorithm files ≡ `main` | `scripts/verify_no_algo_diff.py` | Pass |
| Full pytest | `final_stats/pytest_full_suite.txt` (52 passed, 10 skipped) | Pass* |
| Primary conclusions use 288 only | `fr_pr_convergence_primary288/ASSERTIONS.json` + Round 5 gap script | Pass |
| Obsolete status labelled | `archive_obsolete_preliminary/`, `*.OBSOLETE.txt` | Pass |
| No manufactured missing results | `ROUND5_FINAL_REPORT.md` §7 | Explicit gaps listed |

\*Infrastructure fix in `src/__init__.py` (execute hyphenated package `__init__.py`) restored collection of `tests/test_hanif_carefi.py`; not an algorithm change.

---

# Summary for the editor

We addressed the statistical, feasibility, energy, HCO-attribution, Schneider-coverage, economic-sensitivity, and reporting issues with verified artefacts and manuscript-ready text. We **do not** claim: (i) all Schneider instances solved; (ii) Schneider validates heterogeneous pricing/waiting; (iii) Phase-II Heterogeneous broadly improves on Phase I beyond the 10/360 aggregate cases; (iv) wage/depot-tariff optimality; (v) nonlinear charging-speed benefits from linear PR; (vi) pure-detour replacement ranking throughout HCO-Detour. Remaining gaps are listed under each response and in `ROUND5_FINAL_REPORT.md` §7.
