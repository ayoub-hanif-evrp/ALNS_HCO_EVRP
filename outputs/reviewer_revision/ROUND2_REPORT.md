# Round 2 report — Reviewer 1 experimental requirements

**Branch:** [`reviewer-revision-2026`](https://github.com/ayoub-hanif-evrp/ALNS_HCO_EVRP/tree/reviewer-revision-2026)  
**Base:** `origin/main` @ `e8e7ecf` (untouched)  
**Constraint:** no ALNS/HCO algorithm, operator, objective, or decision-criterion changes.

---

## 1. Round 1 reporting corrections

### Bugs fixed

| Issue | Fix |
|---|---|
| Shared Phase-I iterations overwritten by empty `stages["phase1"]` | `attach_search_diagnostics` now preserves row / `shared_phase1_iterations`; never coerces missing → 0 |
| Total iterations lost Phase-I when only Phase-II present in stages | `sum_optional_ints` sums known phases; `None` only if both missing |
| Runtime conflated solver + reporting | Added `solver_runtime_s`, `reporting_overhead_s`, `end_to_end_runtime_s`; `execution_time_s` = end-to-end |
| New metrics stripped on CSV export | `src/common/reporting_io.py` + `MASTER_FIELDS`; `scripts/run_final_hco_branches.py` uses `MASTER_FIELDS` |
| Double energy accounting | `attach_energy_reporting(..., force=False)` skips when already filled by `solution_row` |

### Validation

- Unit: `tests/test_reporting_diagnostics.py`
- Real Hanif routes: `tests/test_energy_hanif_routes.py` (shared Phase I + GI) — charged energy / visits match `replay_solution`; energy balance holds
- Empirically on PhaseII-Heterogeneous rows: `alns_phase1_iterations` retained from shared Phase I (e.g. 691 + 29 → 720)

---

## 2. FR versus PR convergence

**Runner:** `scripts/run_fr_pr_convergence.py` (uses existing `solve_alns` / `solve_alns_hco`; independent budgets)

### Executed (real measurements)

| Scope | Instances | TW | Seeds | Budgets (s) | Rows | Status |
|---|---|---|---|---|---|---|
| Pilot | C1_10_1, R1_10_1, RC2_10_1 | original | 0 | 30 | 6 | **complete** |
| Extended | balanced 12 (2/family) | original | 0 | 30 | 24 | **complete** |

### Not executed (resource / time)

- Budgets **120 s** and **300 s**
- TW **relaxed**
- Seeds **1–4**
- Full Cartesian plan ≈ 12 × 2 × 5 × 3 × 2 = **720** jobs — **incomplete**; do not treat Extended as the full Reviewer protocol

### Outputs

- `outputs/reviewer_revision/fr_pr_convergence/pilot_runs.csv`
- `outputs/reviewer_revision/fr_pr_convergence/full_runs.csv` (extended 30 s only)
- `table_convergence.csv`, `paired_instance_effects.csv`, `paired_summaries.csv`
- `fig_fr_pr_convergence.png`

### Headline means (extended, original TW, seed 0, 30 s, n=12)

| Method | Mean K | Mean distance | Mean propulsion | Mean en-route purchased | Mean iterations | Mean solver runtime (s) |
|---|---|---|---|---|---|---|
| ALNS-FR | 79.00 | 49667.2 | 49667.2 | 23393.2 | 3683.0 | 30.67 |
| ALNS-PR | 78.58 | 51595.1 | 51595.1 | 17484.6 | 1047.2 | 30.74 |

Algorithms were **not** altered to favour PR. On this 30 s seed-0 slice, FR has lower mean distance; PR purchases less en-route energy (partial recharge). Full multi-budget curves remain outstanding.

---

## 3. Complete ALNS-HCO absolute results

**Runner:** `scripts/run_alns_hco_absolute_results.py`  
Phase-II jobs reuse **identical stored Phase-I** via `ensure_shared_phase1`.

### Executed

| Scope | Instances | TW | Seed | Methods | Rows |
|---|---|---|---|---|---|
| Pilot | 3 | original | 0 | PR / PhaseI-15 / NoHCO / Het | 12 |
| Extended | 12 | original | 0 | same | 48 |

### Not executed

- Relaxed TW; seeds 1–4; remaining balanced instances beyond the 12-name list

### Outputs

- `outputs/reviewer_revision/alns_hco_absolute/pilot_runs.csv`
- `outputs/reviewer_revision/alns_hco_absolute/full_runs.csv`
- `table_absolute_means.csv`, paired CSVs

### Headline means (extended, original, seed 0, n=12)

| Method | K | Distance | En-route E | Propulsion | Depot replenish | Residual | Visits | Expense | Iters (total) | Solver s |
|---|---|---|---|---|---|---|---|---|---|---|
| ALNS-PR (30) | 78.58 | 51650.0 | 17701.2 | 51650.0 | 33948.8 | 13201.2 | 55.08 | 9862.7 | 966.7 | 30.78 |
| PhaseI-PR-15 | 79.67 | 51991.1 | 18193.7 | 51991.1 | 33797.3 | 14002.7 | 56.50 | 9980.0 | 774.2 | 15.91 |
| PhaseII-NoHCO | 79.67 | 51968.0 | 18144.1 | 51968.0 | 33823.9 | 13976.1 | 56.50 | 9951.1 | 787.9 | 31.45 |
| PhaseII-Heterogeneous | 79.67 | 51991.1 | 18193.7 | 51991.1 | 33797.3 | 14002.7 | 56.50 | 9980.0 | 787.6 | 31.37 |

On this seed-0 / 12-instance slice, PhaseII-Heterogeneous means coincide with PhaseI-PR-15 for several charging metrics (few accepted HCO improvements within 15 s). Paired CSVs give instance-level diffs and 95% bootstrap CIs where effects exist. **Do not over-generalise** beyond the executed design.

All rows: `energy_balance_ok=True`; en-route purchased matches replay `charged_energy`.

---

## 4. HCO-Detour verification

See `outputs/reviewer_revision/HCO_DETOUR_VERIFICATION.md`.

- Detour shortlist / insertion preference: **invariant** to price/wait (tests pass).
- Feasibility duration: **uses** actual wait and rate (tests pass).
- Replacement tournament: **distance-led geometric keys** — manuscript “detour-only ranking throughout” is **not fully accurate**; logic unchanged in Round 2.

---

## 5. Files changed (Round 2)

| Path | Role |
|---|---|
| `src/common/energy_accounting.py` | Diagnostics / optional energy attach |
| `src/common/reporting_io.py` | **New** CSV survival helpers |
| `src/common/config.py` | Runtime reporting fields |
| `src/alns-hco/clean_experiments.py` | Finalize with shared iters + overhead |
| `src/alns-hco/station_log.py` | `solver_runtime_s` |
| `src/alns-hco/reviewer_experiments.py` | **New** experiment helpers |
| `scripts/run_fr_pr_convergence.py` | **New** |
| `scripts/run_alns_hco_absolute_results.py` | **New** |
| `scripts/run_final_hco_branches.py` | Export `MASTER_FIELDS` |
| `tests/test_reporting_diagnostics.py` | **New** |
| `tests/test_energy_hanif_routes.py` | **New** |
| `tests/test_hco_detour_science.py` | **New** |
| `outputs/reviewer_revision/**` | Results, figures, this report |

Original `outputs/raw/solution_results.csv` and paper tables were **not** overwritten.

---

## 6. Tests

```text
pytest tests/test_reporting_diagnostics.py tests/test_energy_accounting.py \
  tests/test_energy_hanif_routes.py tests/test_hco_detour_science.py \
  tests/test_hco_final_fix.py tests/test_alns_hco.py -q
→ 34 passed, 2 skipped
```

---

## 7. Remaining issues

1. Full Reviewer protocol (120/300 s, both TW, seeds 0–4) **not finished** — runners support `--full` for overnight completion.
2. HCO-Detour manuscript wording vs replacement implementation — deferred algorithmic fix.
3. Extended absolute slice shows limited Phase-II movement vs Phase I on seed 0; multi-seed paper freezes remain the authority for published HCO claims until full re-run.
4. `route_results.csv` in git is an LFS pointer locally; reviewer runs write self-contained CSVs under `outputs/reviewer_revision/`.
