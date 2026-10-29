# FINAL VERIFIED RESULTS FOR AUTHOR

**Authoritative scientific evidence package** (results only; manuscripts/letters are author-owned).
Audit: raw=6720, seven-method panel=3720, PhaseI≡Het=584/600, NoHCO→Het Cch=16/501/83.

## Reviewer map → artifacts

| ID | Concern | Status | Evidence |
|---|---|---|---|
| R1.1 | Equal FR/PR budgets + 30/120/300 convergence | SATISFIED | `T8_*`; `FigS_FR_PR_convergence_*`; primary288 |
| R1.2 | Fleet size before distance | SATISFIED | `T8_table_fr_pr_gaps_by_budget_lex.csv` |
| R1.3 | Original vs Relaxed TW not pooled as i.i.d. | SATISFIED | `paired_comparisons_all.csv` (TW-conditioned); audit `tw_separated_frpr` |
| R1.4 | Energy balance / distinct energy terms | SATISFIED | `T6_energy_accounting.csv` (depot reconstructed) |
| R1.5 | HCO attribution PhaseI/NoHCO/Detour/Het + Plain | SATISFIED | `T4_*`; `T5_*`; PhaseII-Plain in archive; five-seed **584/600**, **16/501/83** |
| R1.6 | Detour description vs code | SATISFIED | `HCO_DETOUR_VERIFICATION.md`; `tests/test_hco_detour_science.py` |
| R1.7 | FR/PR different engines / iteration throughput | SATISFIED | `FR_PR_FAIRNESS_CORRECTED.md`; primary288 iteration means |
| R1.8 | Linear PR ≠ nonlinear charging-speed model | SATISFIED | constant-rate model; no nonlinear experiment claimed |
| R1.9 | Hardware/seeds/budgets/benchmarks | SATISFIED | `benchmark_characteristics/*`; frozen config; seeds 0–4 documented |
| R2.1 | Schneider coverage + failures | SATISFIED | 336/264/72; T7_* |
| R2.2 | Search failure vs proven infeasibility | SATISFIED | hard-customer diagnosis; external verifier missing C65/C75 |
| R2.3 | No heterogeneous claims from Schneider | SATISFIED | `schneider/NEUTRAL_CHARGING.md` |
| R2.4 | Post-hoc tariff/wage | SATISFIED | `T9_economic_sensitivity_posthoc.csv` |
| R2.5 | Benchmark characteristic tables | SATISFIED | `benchmark_characteristics/table_*_instances.csv` |
| R2.6 | Data availability paths (declarations author-owned) | PARTIAL | paths/manifest ready; declarations not authored here |
| R2.7 | Figures 4–5 = NoHCO→Het | SATISFIED | `Figure4_*`, `Figure5_*`, `figure5_family_values.json` ↔ T5 |
| R2.BKS | Verified primary numerical BKS gap table | PARTIAL | External Java verifier OK on 3/3 feasible diagnostics; primary routes not archived |

## 1. Denominators
- Hanif raw **6720**; seven-method panel **3720** (=120 GI + 6×600).
- PhaseI→Het aggregate identity **584/600** (routes not claimed identical).
- NoHCO→Het charging expense **16 / 501 / 83**.
- Convergence primary288: **288**.
- Fixed-route policy: {'scope': 'all_seed01_archives', 'n': 240, 'partial_feasible': 240, 'full_feasible': 42, 'both_feasible': 42, 'partial_ok_full_infeasible': 198}.
- Schneider primary: **336 / 264 / 72**; hard C65/C64/C75 preserved.
- Historical three-seed **350/360**: superseded (see `SUPERSEDED_REPORTS_INDEX.md`).

## 2. FR → PR paired % (five-seed)

| TW | Metric | n | Mean % | 95% CI |
|---|---|---:|---:|---|
| original | k | 60 | +0.40% | [-0.08, 0.91] |
| original | distance | 60 | +3.34% | [2.11, 4.68] |
| original | charged_energy | 60 | -26.01% | [-33.42, -18.52] |
| original | charging_expense | 60 | -26.82% | [-34.57, -19.10] |
| original | charging_time | 60 | -25.43% | [-32.73, -17.92] |
| original | station_waiting | 60 | -22.39% | [-27.15, -17.89] |
| relaxed | k | 60 | +0.57% | [0.05, 1.16] |
| relaxed | distance | 60 | +3.37% | [2.55, 4.24] |
| relaxed | charged_energy | 60 | -11.52% | [-15.75, -7.41] |
| relaxed | charging_expense | 60 | -12.62% | [-17.26, -8.14] |
| relaxed | charging_time | 60 | -10.74% | [-14.83, -6.80] |
| relaxed | station_waiting | 60 | -15.41% | [-19.59, -11.54] |

PR reduces public charging expense/energy/time; distance may increase. FR/PR are different search configurations.

## 3. HCO
- Benefits limited; concentrated in C1 Original TW (~−21% NoHCO→Het Cch); Relaxed can worsen.
- HCO not universally superior to NoHCO (83 worsenings on Cch).
- Candidate count k=5 is a fixed design choice (`HCO_sensitivity_candidates*.csv`).
- PhaseII-Plain negative control present in raw archive methods.

## 4. Energy & economics
- T6 identity OK: 24/24 (`E_depot = E_drive − E_enroute`, reconstructed).
- T9 depot tariff / wage: post-hoc only.

## 5. Fixed-route full vs partial
- 240 / 42 / 198: the 198 are **counterfactual full-recharge infeasibility on fixed partial routes**,
  not evidence that a free FR solver fails those instances.

## 6. Schneider external verifier
- Tool: https://github.com/ghiermann/evrptw-verifier (built jar + OpenJDK 17 under `tools/`).
- Collapsed consecutive-station mapping: **3** externally valid feasible diagnostics;
  **2** hard incomplete (missing C65/C75) correctly INVALID.
- Diagnostic BKS gaps (fleet-first; inventory BKS; **not** primary-336 claims):
  - c101_21: K=12 vs BKS 12; cost=1197.084; valid=True; gap%=13.593653625347555; distance_gap_vs_inventory_BKS_after_external_valid
  - c201_21: K=5 vs BKS 4; cost=739.248; valid=True; gap%=14.583669167338348; fleet_diff our_k=5 bks_k=4; distance_gap secondary
  - r101_21: K=25 vs BKS 18; cost=2262.128; valid=False; gap%=—; not_claimed_solution_invalid
  - r201_21: K=5 vs BKS 3; cost=1325.483; valid=True; gap%=4.837620222728424; fleet_diff our_k=5 bks_k=3; distance_gap secondary
  - rc101_21: K=24 vs BKS 16; cost=2702.059; valid=False; gap%=—; not_claimed_solution_invalid

## PhaseII-Plain vs PhaseII-Heterogeneous (five-seed; unfavorable control evidence)

PhaseII-Plain uses Phase-I destroy operators (no Phase-II charging-aware destroy / HCO).
PhaseII-NoHCO uses Phase-II destroy/search without HCO station selection.
PhaseII-Heterogeneous adds HCO (heterogeneous scoring).

Paired seed-averaged results (treatment − baseline = Het − Plain):

| TW | Metric | n | Mean diff (Het−Plain) | Mean paired % | 95% CI (diff) | Notes |
|---|---|---:|---:|---:|---|---|
| original | k | 60 | 0 | +0.000% | [0, 0] | equal K on 60/60 |
| original | distance | 60 | 682.987 | +1.567% | [585, 780.4] | Plain lower on 60/60 |
| original | charging_expense | 60 | 508.52 | +4.804% | [382.4, 634.4] |  |
| original | charged_energy | 60 | 928.598 | +5.413% | [728.8, 1134] |  |
| original | station_waiting | 60 | -1.08264 | -1.485% | [-10.28, 7.336] |  |
| relaxed | k | 60 | 0 | +0.000% | [0, 0] | equal K on 60/60 |
| relaxed | distance | 60 | 1156.42 | +2.692% | [996.3, 1323] | Plain lower on 60/60 |
| relaxed | charging_expense | 60 | 940.825 | +10.895% | [862, 1020] |  |
| relaxed | charged_energy | 60 | 1700.93 | +10.151% | [1573, 1834] |  |
| relaxed | station_waiting | 60 | 19.943 | +3.151% | [12.71, 27.42] |  |

**Interpretation:** Plain achieves lower distance than Heterogeneous on all 60 base instances in each TW with equal mean fleet size — HCO Heterogeneous is not a universal routing improvement over the Plain negative control. Source: `FINAL_TABLES/T_PhaseII_Plain_vs_Heterogeneous_*.csv`.

## Schneider Scope B (new)
- 56 ALNS-FR × seed 0 × 30 s with full route archives; primary 336 **unchanged**.
- Reproduced exact primary metrics: **51/56**; discrepancies: **5/56** (`Schneider_ScopeB_repro_discrepancies.csv`).
- Feasible **44/56**; external Java verifier valid **44/56** (invalid = incomplete customer cover, incl. hard C65/C64/C75 cases).
- Lex BKS (valid only): fleet better/tie/worse = **0/10/34**; all 10 fleet ties have positive distance gaps vs inventory BKS.
- Analysis: `SCHNEIDER_SCOPE_B_ANALYSIS.md`. **Recommendation:** Scope B supports optional extension to **168 FR runs** (not 336 yet; no PR rerun for this verifier).

## 7. Limitations
1. Schneider BKS remains **PARTIAL**: Scope B verifies 44/56 FR seed-0 routes externally; primary 336 still lacks full multi-seed route archives. Inventory BKS gaps (esp. fleet ties) show 30 s FR is not competitive.
2. Scope B had 5/56 non-exact reproductions vs primary seed-0 metrics (stochastic search); primary archive unchanged.
3. Declarations / manuscript / response letters: author-owned (R2.6 PARTIAL).
4. FR/PR ≠ pure recharge ablation; policy experiment is the controlled ablation.
5. Depot energy reconstructed; economics post-hoc.
6. Recommend optional **168 FR (56×3 seeds)** before 336; do not rerun PR for full-recharge verifier.

## 8. QC
- pytest: 52 passed, 10 skipped in 7.21s (ok=True)
- algo-diff vs main: ok=True
- Scope B summary: `QC_LOGS/schneider_scope_b_summary.json`
