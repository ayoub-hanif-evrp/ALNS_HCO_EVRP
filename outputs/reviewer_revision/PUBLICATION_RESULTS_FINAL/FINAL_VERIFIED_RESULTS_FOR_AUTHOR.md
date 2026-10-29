# FINAL VERIFIED RESULTS FOR AUTHOR

**Authoritative scientific evidence package** for reviewer-required experiments.
Do not use older three-seed (350/360) summaries as final claims.

Generated against raw archive `6720` rows; seven-method panel `3720` rows.

## 0. Package map (reviewer concern → artifact)

| Reviewer concern | Primary table/figure | Path |
|---|---|---|
| Hanif five-seed routing baseline | Table T3 / overall | `FINAL_TABLES/T3_overall_routing_performance.csv` |
| Partial recharging FR vs PR | Figure 2 + paired table | `FINAL_FIGURES/Figure2_*`, `paired_comparisons_all.csv` (ALNS-FR→PR) |
| Fixed-route HCO Detour vs Het | Figure 3 | `FINAL_FIGURES/Figure3_*` |
| PhaseII NoHCO vs Het | Figure 4 + T4 counts | `FINAL_FIGURES/Figure4_*`, `T4_HCO_contribution_counts_seeds0to4.csv` |
| Family HCO effects / C1 concentration | Figure 5 + T5 | `FINAL_FIGURES/Figure5_*`, `T5_family_HCO_effects.csv` |
| Energy purchased vs propulsion | T6 | `FINAL_TABLES/T6_energy_accounting.csv` |
| Economic sensitivity (post-hoc) | T9 | `FINAL_TABLES/T9_economic_sensitivity_posthoc.csv` |
| Schneider external set | T7* | `FINAL_TABLES/T7_Schneider_*.csv` |
| Convergence 30/120/300 | T8* + FigS | `FINAL_TABLES/T8_*`, `FINAL_FIGURES/FigS_FR_PR_convergence_*` |
| Fixed-route full-versus-partial counterfactual | Policy CSV | `FR_PR_TRUE_FIXED_ROUTE_POLICY_EXPERIMENT.csv` |
| HCO candidate sensitivity 2/5/8 | Sensitivity | `FINAL_TABLES/HCO_sensitivity_candidates*.csv` |
| FR/PR fairness / GI | Fairness MD | `FR_PR_FAIRNESS_CORRECTED.md` |

## 1. Protocol denominators (verified)
- Hanif raw: **6720**
- Seven-method panel: **3720** = 120 GI + 6×600 (VNS, SA, ALNS-FR, ALNS-PR, PhaseII-NoHCO, PhaseII-Heterogeneous)
- Seeds Hanif: 0–4 (GI deterministic seed 0 only)
- Schneider primary: 56×2×3 = **336**; feasible **264**; unsuccessful **72**
- Convergence: **288** unique (seeds 0–1)
- Fixed-route policy: **240** evaluations; both-feasible **42**; partial-OK/full-infeasible **198**
- PhaseI → PhaseII-Heterogeneous identity (K, distance, charged energy): **584/600**
- NoHCO → Het charging expense (run-level, both TW): **16 improved / 501 unchanged / 83 worsened**

### Historical (not final)
- Three-seed PhaseI≡Het **350/360** — superseded by 584/600; see `SUPERSEDED_REPORTS_INDEX.md`.

## 2. ALNS-FR → ALNS-PR (five-seed, instance-level paired %)

| TW | Metric | n | Mean paired % | 95% CI |
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

**Interpretation for author:** PR reduces public charging expense / charged energy / charging time; distance (and thus propulsion energy under constant rate) can increase. FR and PR use different search implementations — not pure policy isolation (see controlled fixed-route experiment).

## 3. HCO contribution
- Run-level NoHCO→Het charging expense: **16 / 501 / 83** (`T4_HCO_contribution_counts_seeds0to4.csv`).
- Aggregate PhaseI≡Het on K/distance/charged energy: **584/600** (route identity not claimed).
- Family means: C1 Original TW dominates favorable charging-expense change (~−21% phase2_nohco_hco); Relaxed TW can worsen (e.g. C1 ~+1.4%). Source: `T5_family_HCO_effects.csv`, Figure 5.
- Candidate counts {2,5,8}: identical metrics on development sensitivity — **k=5 is a fixed design choice**.

## 4. Energy & economics
- Propulsion vs en-route purchased vs reconstructed depot/residual: `T6_energy_accounting.csv` (both TW).
- Identity used for reconstruction: `E_depot = E_drive − E_enroute` (label reconstructed, not observed).
- Depot tariff / wage: `T9_economic_sensitivity_posthoc.csv` — **post-hoc**, not optimized objectives.

## 5. Convergence (canonical 288)
- Tables: `T8_table_convergence_by_tw.csv`, `T8_table_fr_pr_gaps_by_budget_lex.csv`.
- Fleet size primary; distance secondary when K tied.

## 6. Fixed-route full vs partial counterfactual
- 240 / 42 / 198 outcomes; both-feasible paired CIs in `FR_PR_TRUE_POLICY_paired_summaries.csv`.
- Fixes customer+station visits; a free FR solver could choose different routes.

## 7. Schneider
- Feasibility T7; hard customers C65/C64/C75; fidelity 56/56.
- Targeted route export diagnostic: 3/3 internal replay OK.
- **Numerical BKS gaps: NOT SATISFIED** (rounding + incomplete primary route archives). See `SCHNEIDER_BENCHMARK_COMPARABILITY.md`.
- Homogeneous Schneider charging does not validate heterogeneous HCO price/wait claims.

## 8. Figure ↔ CSV check
- Overall OK: **True** — detail `QC_LOGS/figure_csv_consistency.json`.

## 9. Limitations (must remain visible)
1. No verified Schneider BKS numerical gap table.
2. Full-solver FR/PR ≠ pure recharge ablation.
3. HCO k=5 not experimentally proven optimal.
4. Aggregate PhaseII≡PhaseI does not prove identical routes.
5. Depot/residual energy often reconstructed.

