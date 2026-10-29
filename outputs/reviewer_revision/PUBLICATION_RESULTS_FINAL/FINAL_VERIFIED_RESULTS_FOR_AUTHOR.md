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

## 7. Limitations
1. Full primary Schneider BKS gap table still PARTIAL (no route archives for all 264).
2. Declarations / manuscript / response letters: author-owned (R2.6 PARTIAL for declarations).
3. FR/PR ≠ pure recharge ablation; policy experiment is the controlled ablation.
4. Depot energy reconstructed; economics post-hoc.
5. Diagnostic 30 s Schneider tours are not competitive BKS claims.

## 8. QC
- pytest: 52 passed, 10 skipped in 7.39s (ok=True)
- algo-diff vs main: ok=True
- Figure5↔T5: True

