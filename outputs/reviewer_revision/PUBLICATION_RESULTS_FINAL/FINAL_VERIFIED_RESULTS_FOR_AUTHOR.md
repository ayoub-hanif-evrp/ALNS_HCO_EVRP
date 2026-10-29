# FINAL VERIFIED RESULTS FOR AUTHOR

**Authoritative scientific evidence package** (results only). Manuscripts and reviewer letters are author-owned.

Freeze check: raw=6720, panel=3720, PhaseI≡Het=584/600, NoHCO→Het Cch=(16, 501, 83), policy={'scope': 'all_seed01_archives', 'n': 240, 'partial_feasible': 240, 'full_feasible': 42, 'both_feasible': 42, 'partial_ok_full_infeasible': 198}, fig5_ok=True.

## Reviewer evidence map

| ID | Concern | Status | Evidence |
|---|---|---|---|
| R1.1 | Equal FR/PR budgets + 30/120/300 | SATISFIED | T8_*; FigS_*; primary288 |
| R1.2 | Fleet before distance | SATISFIED | T8 lex tables; Schneider FR168 lex BKS |
| R1.3 | TW not pooled as i.i.d. | SATISFIED | paired_comparisons_all.csv |
| R1.4 | Energy terms + balances | SATISFIED | T6_energy_accounting.csv |
| R1.5 | HCO attribution + Plain control | SATISFIED | T4_*; T5_*; Plain vs Het tables |
| R1.6 | Detour vs code | SATISFIED | HCO_DETOUR_VERIFICATION.md |
| R1.7 | FR/PR engine / throughput | SATISFIED | FR_PR_FAIRNESS_CORRECTED.md |
| R1.8 | Linear PR ≠ nonlinear speed curve | SATISFIED | constant-rate model |
| R1.9 | Setup transparency | SATISFIED | benchmark_characteristics; frozen config |
| R2.1 | Schneider coverage | SATISFIED | Primary 336/264/72 + FR168 routes |
| R2.2 | Unsuccessful explained | SATISFIED | hard C65/C64/C75; verifier missing customers |
| R2.3 | No hetero claims from Schneider | SATISFIED | NEUTRAL_CHARGING.md |
| R2.4 | Post-hoc economics | SATISFIED | T9_* |
| R2.5 | Benchmark tables | SATISFIED | benchmark_characteristics/* |
| R2.6 | Data paths; declarations author-owned | PARTIAL | manifests; declarations not authored here |
| R2.7 | Figs 4–5 NoHCO→Het | SATISFIED | Figure4_*; Figure5_* ↔ T5 |
| R2.BKS | FR BKS with external verification | PARTIAL | FR168: 132/168 externally valid; not full 336 FR/PR; not competitive BKS quality |

## Schneider FR168 (final experimental round)
- Attempts: **168** (56×3). Feasible **132**. Externally valid **132**.
- Exact match to primary FR metrics: **144/168** (wall-clock throughput differences documented).
- Serial diagnosis of 5 seed-0 mismatches: 2/5 exact under serial load.
- Tables: `Schneider_FR168_external_verification.csv`, `Schneider_FR168_BKS_fleet_ties.csv`.
- Analysis: `SCHNEIDER_FR168_ANALYSIS.md`, `SCHNEIDER_REPRODUCIBILITY_DIAGNOSIS.md`.
- **Not** independent validation of ALNS-PR (full-recharge verifier).

## PhaseII-Plain vs PhaseII-Heterogeneous (unfavorable)
Plain = Phase-I destroy operators (no Phase-II charging-aware destroy/HCO). NoHCO = Phase-II search without HCO. Heterogeneous = Phase-II + HCO.

| TW | Metric | n | Mean (Het−Plain) | Mean % | Notes |
|---|---|---:|---:|---:|---|
| original | k | 60 | 0.0 | 0.0 | equal K on 60/60 |
| original | distance | 60 | 682.9873725692428 | 1.5669679674860144 | Plain lower on 60/60 |
| original | charging_expense | 60 | 508.51983638654553 | 4.804318796014366 |  |
| original | charged_energy | 60 | 928.5977640023469 | 5.412661278087101 |  |
| original | station_waiting | 60 | -1.0826429009437462 | -1.484558081135401 |  |
| relaxed | k | 60 | 0.0 | 0.0 | equal K on 60/60 |
| relaxed | distance | 60 | 1156.4154836897044 | 2.6917668743939434 | Plain lower on 60/60 |
| relaxed | charging_expense | 60 | 940.8249134173235 | 10.894653098405035 |  |
| relaxed | charged_energy | 60 | 1700.9291871623425 | 10.151018685651419 |  |
| relaxed | station_waiting | 60 | 19.942953910827647 | 3.1505946803822034 |  |

## Other frozen denominators (not rerun)
- Hanif raw **6720**; seven-method **3720**
- PhaseI→Het **584/600**; NoHCO→Het Cch **(16, 501, 83)**
- Convergence **288**; policy **{'scope': 'all_seed01_archives', 'n': 240, 'partial_feasible': 240, 'full_feasible': 42, 'both_feasible': 42, 'partial_ok_full_infeasible': 198}**

## Limitations
1. FR168 BKS is PARTIAL: externally verified FR routes exist, but 30 s solutions are generally worse than inventory BKS; not a 336 FR/PR verification.
2. Wall-clock ALNS ⇒ primary vs re-archive metrics can differ under CPU load; both preserved.
3. No independent partial-recharge verifier was required/available for PR route-level claims; PR not externally route-validated.
4. Manuscripts/declarations/response letters: author-owned.
5. HCO is not universally superior (Plain distance wins; NoHCO→Het 83 worsenings).

