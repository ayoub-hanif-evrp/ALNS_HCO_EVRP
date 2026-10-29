# FINAL VERIFIED RESULTS FOR AUTHOR

**Authoritative numerical package (release audit).** Reviewer coverage checked against the repository's documented reviewer-concern summary; exact original reviewer-letter wording not independently verified.

Source-locked counts: raw=6720, panel=3720, conv=288, Schneider336=336/264/72, FR168=168/132/132, PhaseI≡Het=584/600, NoHCO→Het=(16, 501, 83), policy={'scope': 'all_seed01_archives', 'n': 240, 'partial_feasible': 240, 'full_feasible': 42, 'both_feasible': 42, 'partial_ok_full_infeasible': 198}.

## Reviewer evidence map

| ID | Concern | Status | Primary evidence |
|---|---|---|---|
| R1.1 | Equal FR/PR budgets; 30/120/300 | SATISFIED | T8_*; FigS_*; primary288 |
| R1.2 | Fleet before distance | SATISFIED | T8 lex; FR168 BKS lex |
| R1.3 | TW not pooled | SATISFIED | paired_comparisons_all.csv |
| R1.4 | Energy accounting | SATISFIED | T6_* |
| R1.5 | HCO attribution + Plain | SATISFIED | T4_*; T5_*; Plain vs Het |
| R1.6 | Detour vs code | SATISFIED | HCO_DETOUR_VERIFICATION.md |
| R1.7 | FR/PR engines | SATISFIED | FR_PR_FAIRNESS_CORRECTED.md |
| R1.8 | Linear PR ≠ nonlinear curve | SATISFIED | constant-rate model |
| R1.9 | Setup transparency | SATISFIED | benchmark_characteristics; frozen config |
| R2.1 | Schneider coverage | SATISFIED | Primary 336 + FR168 routes |
| R2.2 | Unsuccessful explained | SATISFIED | hard customers; verifier missing-customer logs |
| R2.3 | No hetero from Schneider | SATISFIED | NEUTRAL_CHARGING.md |
| R2.4 | Post-hoc economics | SATISFIED | T9_* |
| R2.5 | Benchmark tables | SATISFIED | benchmark_characteristics/* |
| R2.6 | Data paths; declarations author-owned | PARTIAL | manifests; declarations author-owned |
| R2.7 | Figs 4–5 NoHCO→Het | SATISFIED | Figure4_*; Figure5_* |
| R2.BKS | FR external verification + BKS gaps | PARTIAL | FR168 complete for 132 valid FR routes; primary 336 FR/PR not route-verified; not competitive vs inventory BKS |
| R2.PR_routes | Independent PR route verification | PARTIAL | Not explicitly requested in available reviewer-concern summary; independent PR route-level external verification remains unavailable |

## Schneider FR168
- Per-run: **168** attempts; **132** feasible; **132** externally valid; **36** unsuccessful kept visible.
- Lex BKS (valid runs only): {'FLEET_TIE': 34, 'FLEET_WORSE_THAN_BKS': 98}
- Repro vs primary: exact 144; discrepancies 24 (all audited).
- Instance-level table: **best-of-three valid seeds (K then distance)**, not averaging.
- Provenance: `Schneider_BKS_FR_provenance_56.csv` (Erdelić & Carić 2022 consolidation).
- Java verifier = **full recharge only**.

## Plain vs Heterogeneous (unfavorable; retain)
| TW | Metric | n | Mean Het−Plain % | Note |
|---|---|---:|---:|---|
| original | k | 60 | +0.000 | equal K on 60/60 |
| original | distance | 60 | +1.567 | Plain lower on 60/60 |
| original | charging_expense | 60 | +4.804 |  |
| relaxed | k | 60 | +0.000 | equal K on 60/60 |
| relaxed | distance | 60 | +2.692 | Plain lower on 60/60 |
| relaxed | charging_expense | 60 | +10.895 |  |

## FR→PR paired (Hanif five-seed)
| TW | Metric | n | Mean % | 95% CI |
|---|---|---:|---:|---|
| original | k | 60 | +0.40% | [-0.08, 0.91] |
| original | distance | 60 | +3.34% | [2.11, 4.68] |
| original | charging_expense | 60 | -26.82% | [-34.57, -19.10] |
| relaxed | k | 60 | +0.57% | [0.05, 1.16] |
| relaxed | distance | 60 | +3.37% | [2.55, 4.24] |
| relaxed | charging_expense | 60 | -12.62% | [-17.26, -8.14] |

## Figure verification
- Programmatic source checks: **True** — detail `QC_LOGS/figure_source_verification.json`.
- FigS: source T8 present; PNG pixels not re-extracted.

## Limitations
1. FR168 ≠ route verification of primary 336 FR/PR.
2. 30 s FR generally worse than inventory BKS; not competitive-performance claims.
3. 24 wall-clock metric discrepancies vs primary (feasibility matched); 5% distance split is descriptive.
4. Independent PR route-level external verification unavailable; not asserted as required without original-letter verification.
5. Manuscript/letters/declarations: author-owned.

## Author action warning
Do not copy **350/360**, **4200** panel, or “seed-0-only Schneider verification” into the manuscript. See `SUPERSEDED_REPORTS_INDEX.md`.

