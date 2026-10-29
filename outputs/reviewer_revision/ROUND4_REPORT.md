# Round 4 report — scientific reporting corrections & Reviewer 2 analysis

**Branch:** `reviewer-revision-2026`  
**Constraint:** no ALNS/HCO/Phase I–II algorithm changes; `main` untouched.  
**Prior commit baseline:** Round 3 experimental workload complete.

---

## 1. Convergence analysis — primary 288 protocol (FIXED)

### Problem
`fr_pr_convergence/runs.csv` mixed **352** rows (288 primary + **64 seed-2** extras). Paired summaries showed `mean_n_matched_seeds` > 2.

### Correction
- Regenerated **all** primary tables, bootstrap CIs, and figures from exactly **288** records (12 instances × 2 TW × 2 methods × 3 budgets × seeds **{0,1}**).
- Asserted **24** observations per method × TW × budget cell.
- Asserted **exactly 2** matched seeds per instance in primary paired comparisons.
- Seed-2 archived separately: `fr_pr_convergence_seed2_supplementary/` (labeled **not** primary).
- Canonical primary outputs: `fr_pr_convergence_primary288/` and refreshed `fr_pr_convergence/` analysis files.

### Lexicographic FR vs PR gaps (PR − FR), seeds 0–1 only

| Budget | TW | Mean ΔK | 95% CI K | Mean Δdistance | 95% CI distance | K ties | PR better K | FR better K |
|---|---|---|---|---|---|---|---|---|
| 30 | original | −0.21 | [−1.17, 0.71] | +2072 | [196, 4351] | 4 | 5 | 3 |
| 30 | relaxed | +1.67 | [0.75, 2.67] | +1596 | [−34, 3394] | 3 | 0 | 9 |
| 120 | original | +0.46 | [−0.67, 1.50] | +1433 | [−256, 3691] | 2 | 4 | 6 |
| 120 | relaxed | +1.08 | [0.50, 1.63] | **−1240** | [−2212, −210] | 1 | 1 | 10 |
| 300 | original | +0.96 | [−0.38, 2.21] | +1548 | [252, 3013] | 2 | 3 | 7 |
| 300 | relaxed | +1.08 | [0.58, 1.63] | +999 | [−1117, 3029] | 3 | 0 | 9 |

**Interpretation:** Fleet size is primary. On relaxed TW, FR often has smaller fleets; distance advantages for PR (e.g. 120 s relaxed) are **not** unconditional superiority when K differs. En-route purchased energy remains lower for PR across budgets (see `paired_summaries.csv`). Matched-seed mean is **2.0** everywhere.

---

## 2. Schneider feasibility — corrected investigation

### Bug fixed
Round-3 `try_solo_route()` called `replay_solution()` on a one-route solution, marking the other **99** customers unserved. That is invalid for route-level feasibility.

### Method now
`src.hanif.replay.replay_route` for direct and station-insertion walks.

### Findings (12 ALNS-infeasible instances @30 s)
For every customer that fails depot–customer–depot, a **feasible station-insertion pattern exists** in a small probe (`fail_all_probes = 0` on all 12 instances).  
**Classification:** construction/search failure under the current implementation — **not** a proof of mathematical infeasibility of official Schneider instances (esp. under alternative distance rounding).

### Primary Schneider stats (30 s only)
| Quantity | Value |
|---|---|
| Planned / completed | **336 / 336** |
| Feasible | **264** |
| Infeasible | **72** |
| Instances with both methods feasible (paired) | **44** |

Conditional FR vs PR (44 instances, 3 seeds): charged-energy mean diff favours PR (CI excludes 0); distance CI overlaps 0.  
File: `schneider/feasibility_investigation_round4.json`, `table_main30_*.csv`, `paired_main30_*.csv`.

No adapter/data error requiring re-solves was established; all 336 primary attempts retained.

---

## 3. Hanif energy accounting — completed

### Method
Under verified full-departure batteries and no charging losses:

- `E_drive = K·B_max + E_enroute − E_return`
- `E_depot = E_drive − E_enroute`
- `E_return = K·B_max − E_depot`

Phase-I metrics taken from **shared Phase-I routes** where available. Other methods: fill missing residual/depot via the identity; label  
`energy_source=…+reconstructed_aggregate_energy_balance`  
(not directly observed route-level SOC).

### Coverage (seeds 0–2, 2520 rows)
All seven methods: **360/360** residual and depot fields present after reconstruction; **0** flagged negative/out-of-range rows.

Provenance: `hanif_main_seeds012/energy_reconstruction_note.json`, `table_energy_coverage.csv`.

---

## 4. HCO contribution audit

Compared matched (instance, TW, seed) triples on the full Hanif 3-seed set (n=360).

### PhaseI → PhaseII-Heterogeneous (lower-is-better)
| Metric | Improved | Unchanged | Worsened |
|---|---|---|---|
| distance | 10 | **350** | 0 |
| charged_energy | 10 | **350** | 0 |
| charging_expense | 10 | 350 | 0 |
| station_waiting | 10 | 350 | 0 |
| k | 0 | 360 | 0 |
| duration | 6 | 350 | 4 |

### PhaseI → PhaseII-NoHCO (search without HCO)
| Metric | Improved | Unchanged | Worsened |
|---|---|---|---|
| distance | 69 | 291 | 0 |
| charged_energy | 58 | 301 | 1 |
| charging_expense | 59 | 301 | 0 |

### NoHCO → Heterogeneous
Mostly unchanged on aggregates; some worsened distance/expense counts show Het does not dominate NoHCO on this freeze.

**Aggregate identity:** 350/360 (97.2%) match Phase I on K, distance, and charged energy.  
**Route identity:** **not asserted** — publication CSV lacks final-route checksums/node sequences. Status recorded as `aggregate_metrics_match_phase1_route_identity_unproven`.

**ε = 0.0:** `phase2_distance_tolerance` freezes `D ≤ D_PR` with `K ≤ K*` and lex improvement. Documented only; algorithm unchanged.  
Outputs: `hco_contribution_audit/`.

---

## 5. Reviewer 2 — economic & benchmark analysis (post-hoc)

### Economic sensitivity (optimizer unchanged)
Hypothetical depot tariffs `{0, 0.10, 0.20, 0.50}` and wages per duration `{0, 0.25, 0.50, 1.00}` applied to Hanif main rows.

- Observed public charging expense kept separate from hypothetical depot expense.
- Totals labeled `post_hoc_sensitivity_NOT_optimized`.
- **No claim** that ALNS/HCO optimizes wage-inclusive or depot-tariff cost.

Outputs: `economic_sensitivity/` (`ASSUMPTIONS.json`, scenario means, FR/PR notes).

### Benchmark characteristics
`benchmark_characteristics/table_hanif_instances.csv`, `table_schneider_instances.csv` — families, capacities, charging assumptions, TW regimes.

---

## 6. Validation

- Primary convergence assertions: 288 rows, cell size 24, matched seeds = 2 (`fr_pr_convergence_primary288/ASSERTIONS.json`).
- Regression tests including `tests/test_round4_convergence_filter.py`.
- `scripts/verify_no_algo_diff.py`: protected algorithmic files unchanged vs `origin/main`.

---

## 7. Reviewer checklist

| Request | Round-4 status |
|---|---|
| Clean multi-budget FR/PR convergence | **Fixed** (288-only) |
| Lexicographic interpretation | **Documented** with conditional distance counts |
| Schneider feasibility honesty | **Corrected** (replay_route; search failure vs proof) |
| Full Hanif energy fields | **Filled** with labeled reconstruction |
| HCO actual contribution | **Audited** (improved/unchanged/worsened) |
| Economic / wage / depot sensitivity | **Post-hoc tables** |
| Benchmark description tables | **Done** |

---

## 8. Outstanding for Round 5

- Optional: recover final-route archives to prove/refute PhaseII route identity beyond aggregates.
- Optional: Schneider distance-rounding sensitivity study (still without algorithm changes).
- Manuscript text integration of Round-4 tables/figures.
- Any remaining editorial Reviewer comments not covered by experiments.
