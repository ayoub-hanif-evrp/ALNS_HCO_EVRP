# HCO contribution and parameter sensitivity

## Five-seed contribution (raw matched records)

Phase I → PhaseII-Heterogeneous aggregate identity on \(K\), distance, charged energy:
**584/600** (97.33%).

NoHCO → Heterogeneous charging expense (run-level, n=600):
`{'improved': 16, 'unchanged': 501, 'worsened': 83}`.

**Route identity is not claimed** from matching aggregates.

### Instance-level paired inference
See `FINAL_TABLES/HCO_instance_level_paired_all_contrasts.csv` (seeds averaged within instance; TW separate).

### Family / leave-one-out
`FINAL_TABLES/HCO_family_Cch_leave_one_out.csv` — C1 Original can dominate mean \(C_{ch}\) reductions; medians and leave-out scopes are reported.

### Why Phase II often retains Phase I
Configured `phase2_distance_tolerance` \(\varepsilon=0\) admits only \(D\le D^{PR}\) with \(K\le K^*\). HCO evaluations may be feasible while destroy/repair neighbours fail the distance gate. Existing development ε and budget-split panels produced ties (ΔCch=0) on the tested subset (`outputs/config/frozen_experiment_config.json`, `outputs/analysis/time_allocation_sensitivity.csv`).

### Candidate-count sensitivity (supplementary)
Development subset (6 families × 2 TW × seeds 0–1) with shared Phase I; Phase II Heterogeneous for `candidates_per_criterion` ∈ {2,5,8}.
Tables: `HCO_sensitivity_candidates.csv`, `HCO_sensitivity_candidates_paired_vs_k5.csv`.
**Interpretation:** treat differences as supplementary; do not retune the frozen primary study. If paired CIs vs k=5 include 0, selected k=5 remains a reasonable fixed experimental choice, not a proven optimum.

### Worsenings
Listed in `HCO_NoHCO_vs_Het_worsened_charging_expense_runs.csv` (83 runs).

## Addendum � candidate-count experiment outcome
On the predeclared development subset, k?{2,5,8} produced **identical** fleet size, distance, and charging metrics (Phase II retained Phase I). Runtime increased slightly with k. This supports reporting k=5 as a fixed protocol setting rather than an optimized hyperparameter.
