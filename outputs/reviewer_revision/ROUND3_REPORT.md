# Round 3 report — full-benchmark revision (updated design)

**Branch:** `reviewer-revision-2026`  
**Constraint:** no changes to ALNS-FR / ALNS-PR / HCO / Phase I–II / objectives / acceptance / charging logic. `main` untouched.

**Runtime definitions:** `solver_runtime_s` = solver wall-clock (ALNS ± GI); `reporting_overhead_s` = post-hoc metrics only; `end_to_end_runtime_s` = sum. Equal FR/PR search budget = ALNS `time_limit_s` / Phase-I budget argument.

---

## 0. Final experimental design

| Study | Design | Role | Status |
|---|---|---|---|
| **A. Hanif main** | 60×{original,relaxed}=120 TW cells; FR/PR/PhaseI/PhaseII-NoHCO/PhaseII-Het (+ fixed-route HCO); seeds **0–2** primary; **0–4** legacy | Primary | **Complete** |
| **B. Schneider main** | All **56** `*_21`; FR & PR; seeds **0–2**; **30 s** | Primary | **Complete** |
| **C. Convergence 288** | 12 Hanif (2/family)×2 TW×FR/PR×{30,120,300}s×seeds **0,1** | Supplementary | **Complete** |

Manifests: `outputs/reviewer_revision/manifests/`.

---

## 1. Corrections from Rounds 1–2

| Issue | Status |
|---|---|
| Pairing (instance × TW × seed × budget) | `src/common/paired_stats.py` + tests |
| Completion vs partial protocol | `protocol_completion` + status JSON |
| Runtime scope | Fields above |
| Energy / iterations | Phase-I from `outputs/shared_phase1` routes; other methods: charged energy from CSV + propulsion = distance×rate when node sequences not archived |
| PhaseII-Het ≡ Phase I | **350/360** (seeds 0–2) and **584/600** (0–4) identical on distance & charged energy vs PhaseI; `eps=0`; **algorithm unchanged** |
| Schneider R1/RC1 | Investigated; not silent exclusion (§4) |
| Aggregate plot crash (str/float budget) | Fixed in `aggregate_means` |

---

## 2. Exact experiments / reuse counts

### A. Hanif main

| Item | Count |
|---|---|
| Planned (7 methods × 60 × 2 TW × seeds 0–2) | **2520** |
| Reused `outputs/raw/solution_results.csv` | **2520** |
| New ALNS resolves | **0** |
| Shared Phase-I caches | **600** (seeds 0–4) |
| Feasible (3-seed) | **2520/2520** |

Outputs: `hanif_main_seeds012/`, `hanif_main_seeds0to4/`, figures `fig_hanif_main_*.png`.

### B. Schneider main

| Item | Count |
|---|---|
| Planned | **336** |
| Completed protocol | **336/336** |
| Feasible @30 s | **264** |
| Infeasible @30 s | **72** (12 instances × 3 seeds × 2 methods) |

Infeasible instances: `r101,r102,r103,r105,r106,r108,rc101,rc103–rc107` (`_21`).  
Paired FR/PR on 44 feasible instances: charged-energy mean diff −111 (CI excludes 0); distance CI overlaps 0.  
Tables: `schneider/table_main30_*.csv`, `paired_main30_*.csv`. Neutral attrs: `schneider/NEUTRAL_CHARGING.md`.

### C. Convergence 288

| Item | Count |
|---|---|
| Planned | **288** |
| Completed | **288/288** (`complete_protocol=true`) |
| Budgets present | 30, 120, 300 s |

Outputs: `fr_pr_convergence/` tables, paired CSVs, `fig_fr_pr_convergence*.png`.

---

## 3. Convergence conclusions (supplementary)

Equal-budget FR vs PR multi-budget curves are measured (not single-budget placeholders). Use `table_convergence_by_tw.csv` and `paired_summaries.csv` for publication numbers. Treat as **supplementary** to the full Hanif 30 s main evaluation—not a replacement.

---

## 4. Schneider infeasibility (not silent exclusion)

`schneider/infeasibility_investigation.json`: for `r101_21` cust 65, solo depot–customer–depot is **infeasible** (`solo_feasible=False`); round-trip energy ≈99.9 vs Q=62.14 under **unrounded** Euclidean conversion. All 56 instances were still run; infeasible outcomes recorded with `unserved`. No claim that official Schneider instances are infeasible under every distance-rounding convention; no BKS comparison claimed.

---

## 5. Phase-II HCO diagnostic

On the **full** Hanif reused set, PhaseII-Heterogeneous almost always returns the Phase-I solution on distance/charged energy while still recording HCO evaluations in prior absolute pilots. Configured `phase2_distance_tolerance=0.0`. Documented only—**no algorithm change**.

---

## 6. Tests and algorithm integrity

```
pytest tests/test_paired_stats_round3.py tests/test_job_store_round3.py \
       tests/test_schneider_round3.py tests/test_reporting_diagnostics.py
```

`scripts/verify_no_algo_diff.py`: protected solver/HCO files unmodified vs `origin/main`.

---

## 7. Reviewer checklist

| Request | Status |
|---|---|
| Full Hanif 120 TW cells | **Done** (verified reuse + enrichment) |
| FR/PR equal budget | **Done** (30 s main; 30/120/300 supplementary) |
| Matched Phase-I for Phase-II | **Done** (600 shared caches) |
| Schneider full FR/PR | **Done** (336/336) |
| Longer budgets / convergence | **Done** (288/288) |
| PhaseII-Het behaviour | Documented on full set |
| Transparent completion | Manifests + status JSON |

---

## 8. Outstanding limitations

- Full residual/depot energy for non–Phase-I methods depends on node-sequence archives (not in recoverable `route_results` aggregates); Phase-I fully enriched from shared routes; propulsion derived as distance×rate elsewhere.
- Schneider infeasibilities under unrounded distances should be discussed as converter convention, not solver “timeout.”
- Convergence 288 uses seeds 0–1 only (per updated design); seed-2 cells exist as extras from earlier runs and are not part of the 288 protocol.
