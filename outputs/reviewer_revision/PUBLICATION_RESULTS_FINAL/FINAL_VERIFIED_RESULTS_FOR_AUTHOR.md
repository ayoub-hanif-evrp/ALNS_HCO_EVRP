# FINAL VERIFIED RESULTS FOR AUTHOR

**Single authoritative scientific-results package.** Manuscripts, declarations, and reviewer-response letters are **author-owned** (not failed experiments).

Freeze: raw=6720, panel=3720, PhaseI≡Het=584/600, NoHCO→Het Cch=(16, 501, 83), policy={'scope': 'all_seed01_archives', 'n': 240, 'partial_feasible': 240, 'full_feasible': 42, 'both_feasible': 42, 'partial_ok_full_infeasible': 198}, T6=24/24, fig5=True.

## Reviewer evidence map (documented concerns from repository RESPONSE summary)

| ID | Concern | Status | Evidence |
|---|---|---|---|
| R1.1 | Equal FR/PR budgets; 30/120/300 convergence | SATISFIED | `T8_*`; `FigS_*`; primary288 |
| R1.2 | Fleet before distance | SATISFIED | T8 lex; FR168 lex BKS |
| R1.3 | TW not pooled as i.i.d. | SATISFIED | `paired_comparisons_all.csv` |
| R1.4 | Energy terms + balances | SATISFIED | `T6_energy_accounting.csv` |
| R1.5 | HCO attribution + Plain control | SATISFIED | T4_*; T5_*; Plain vs Het |
| R1.6 | Detour vs code | SATISFIED | `HCO_DETOUR_VERIFICATION.md` |
| R1.7 | FR/PR engines / throughput | SATISFIED | `FR_PR_FAIRNESS_CORRECTED.md` |
| R1.8 | Linear PR ≠ nonlinear charging curve | SATISFIED | constant-rate model |
| R1.9 | Setup transparency | SATISFIED | benchmark tables; frozen config |
| R2.1 | Schneider coverage + failures | SATISFIED | Primary 336/264/72 + FR168 routes |
| R2.2 | Unsuccessful explained | SATISFIED | hard C65/C64/C75; verifier missing-customer logs |
| R2.3 | No hetero claims from Schneider | SATISFIED | `NEUTRAL_CHARGING.md` |
| R2.4 | Post-hoc economics | SATISFIED | T9_* |
| R2.5 | Benchmark tables | SATISFIED | `benchmark_characteristics/*` |
| R2.6 | Data paths; declarations author-owned | PARTIAL | manifests ready; declarations not authored here |
| R2.7 | Figs 4–5 = NoHCO→Het | SATISFIED | Figure4_*; Figure5_* ↔ T5 |
| R2.BKS | FR BKS + external verification | PARTIAL | FR168: 132/168 externally valid; inventory BKS gaps; not 336 FR/PR |
| R2.PR_routes | Independent PR route-level verifier | NOT REQUIRED | Documented reviewer summary (R2.1–R2.3) requires Schneider coverage/honesty, not a PR-specific external route checker; retained as transparent limitation |

## Schneider FR168 (complete three-seed FR verification)

- Attempts: **168** = 56 × 3 seeds (ALNS-FR, GI 20 s, ALNS 30 s).
- Feasible / unsuccessful: **132 / 36**.
- Externally valid (Java full-recharge verifier): **132/168** (equals feasible count).
- Exact reproduce vs primary FR metrics: **144/168**; discrepancies: **24/168**.
- Lex BKS among externally valid runs (per-run): fleet better/tie/worse = **0 / 34 / 98**.
- Primary FR/PR **336** archive **unchanged**.
- **Java verifier = full recharging only.** It does **not** validate ALNS-PR.
- Seeds are **not** independent benchmark instances; see instance-level summary table.

Tables: `Schneider_FR168_external_verification.csv`, `Schneider_FR168_BKS_by_run.csv`, `Schneider_FR168_BKS_instance_summary.csv`, `Schneider_FR168_all24_discrepancies.csv`.

### All 24 reproducibility discrepancies
- Feasibility consistent with primary in **all 24**.
- Impact classes: {'DISTANCE_MINOR': 16, 'DISTANCE_MATERIAL': 3, 'FLEET_CHANGED': 5}
- Cause classes: {'SUPPORTED_LOWER_ITERATION_THROUGHPUT': 24}
- Fleet/feasibility conclusion changes vs primary: **5/24** (same feasibility always; fleet changes where `FLEET_CHANGED`).
- Not every case is attributed solely to CPU contention; see per-row `likely_cause_class`.

## PhaseII-Plain vs Heterogeneous (unfavorable; retained)
Plain uses Phase-I destroy (no Phase-II charging-aware destroy/HCO). NoHCO uses Phase-II search without HCO. Heterogeneous adds HCO.

| TW | Metric | n | Mean Het−Plain | Mean % | Note |
|---|---|---:|---:|---:|---|
| original | k | 60 | 0.0 | 0.0 | equal K 60/60 |
| original | distance | 60 | 682.9873725692428 | 1.5669679674860144 | Plain lower 60/60 |
| original | charging_expense | 60 | 508.51983638654553 | 4.804318796014366 |  |
| original | charged_energy | 60 | 928.5977640023469 | 5.412661278087101 |  |
| original | station_waiting | 60 | -1.0826429009437462 | -1.484558081135401 |  |
| relaxed | k | 60 | 0.0 | 0.0 | equal K 60/60 |
| relaxed | distance | 60 | 1156.4154836897044 | 2.6917668743939434 | Plain lower 60/60 |
| relaxed | charging_expense | 60 | 940.8249134173235 | 10.894653098405035 |  |
| relaxed | charged_energy | 60 | 1700.9291871623425 | 10.151018685651419 |  |
| relaxed | station_waiting | 60 | 19.942953910827647 | 3.1505946803822034 |  |

## ALNS-FR → ALNS-PR (five-seed Hanif; not pure policy isolation)

| TW | Metric | n | Mean paired % | 95% CI |
|---|---|---:|---:|---|
| original | k | 60 | +0.40% | [-0.08, 0.91] |
| original | distance | 60 | +3.34% | [2.11, 4.68] |
| original | charging_expense | 60 | -26.82% | [-34.57, -19.10] |
| original | charged_energy | 60 | -26.01% | [-33.42, -18.52] |
| relaxed | k | 60 | +0.57% | [0.05, 1.16] |
| relaxed | distance | 60 | +3.37% | [2.55, 4.24] |
| relaxed | charging_expense | 60 | -12.62% | [-17.26, -8.14] |
| relaxed | charged_energy | 60 | -11.52% | [-15.75, -7.41] |

PR reduces public charging expense; distance may increase. FR/PR are different search configurations.

## Other frozen denominators
- Hanif **6720**; seven-method panel **3720**
- PhaseI→Het **584/600**; NoHCO→Het Cch **(16, 501, 83)**
- Convergence **288**; fixed-route policy **{'scope': 'all_seed01_archives', 'n': 240, 'partial_feasible': 240, 'full_feasible': 42, 'both_feasible': 42, 'partial_ok_full_infeasible': 198}**
- Energy T6 identity **24/24**; figures present **{'Figure2': True, 'Figure3': True, 'Figure4': True, 'Figure5': True}**

## What is scientifically complete vs remaining

**Completed:** Hanif five-seed panel; FR/PR convergence; HCO ablations incl. Plain; policy counterfactual; energy/econ/sensitivity; Schneider primary 336 coverage; FR168 route archives + external FR verification + BKS tables; full 24-discrepancy audit.

**Limitations (not author writing tasks):**
1. FR168 BKS PARTIAL: externally valid FR exist, but 30 s solutions generally do not beat inventory BKS; not a 336 FR/PR verification.
2. 24/168 metric discrepancies vs primary under wall-clock ALNS; both archives preserved; feasibility always matched.
3. Independent PR route-level external verification was **not** explicitly required by the documented reviewer summary; no PR-compatible external verifier campaign was run.
4. HCO is not universally superior (Plain distance wins; NoHCO→Het worsenings).

**Author-owned (outside this package):** manuscript prose, figure placement, declarations, response-to-reviewers letters.

