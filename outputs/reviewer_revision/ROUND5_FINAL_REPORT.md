> **SUPERSEDED for five-seed publication claims.** Use outputs/reviewer_revision/PUBLICATION_RESULTS_FINAL/FINAL_VERIFIED_RESULTS_FOR_AUTHOR.md. Historical three-seed **350/360** is replaced by **584/600**; seven-method panel is **3720**, not 4200.

# Round 5 — Final scientific verification & manuscript-delivery report

**Branch:** `reviewer-revision-2026`  
**Constraint:** No ALNS/HCO architecture, objective, operator, acceptance, charging, or feasibility-logic modifications. `main` and original publication outputs untouched.  
**Prior tip:** `3647cd6` (Round 4).  
**Mode:** Verification and documentation only — no unnecessary experiment re-runs.

---

## 1. Canonical data sources (do not mix)

| Study | Canonical path | Rows / coverage | Notes |
|---|---|---|---|
| FR/PR multi-budget convergence | `fr_pr_convergence_primary288/` | **288** | Seeds **{0,1}** only; 12 instances × 2 TW × 2 methods × 3 budgets |
| Seed-2 extras | `fr_pr_convergence_seed2_supplementary/` | 64 | **Not** primary |
| Hanif main | `hanif_main_seeds012/` | **2520** | 60×2 TW×7 methods×seeds {0,1,2} |
| Schneider 30 s | `schneider/` | **336** attempts | 56 instances × FR/PR × seeds {0,1,2} |
| HCO audit | `hco_contribution_audit/` | 360 matched triples | PhaseI / NoHCO / Het |
| Economics | `economic_sensitivity/` | post-hoc | Optimizer unchanged |
| Gap change 30→300 | `final_stats/` | Round 5 | Bootstrap CIs |

Obsolete preliminary status files are copied under `archive_obsolete_preliminary/` and marked `*.OBSOLETE.txt` beside originals.

---

## 2. Final statistical verification — FR/PR gap change 30→300 s

**Protocol:** For each TW separately, average seeds within (instance, method, budget), form instance-level gap \(G_b = \overline{\mathrm{PR}}_b - \overline{\mathrm{FR}}_b\), then \(\Delta G = G_{300} - G_{30}\). Bootstrap mean 95% CI over the **12 instances** (not treating TW or seeds as independent base instances). Lexicographic objective: **fleet size \(K\) first**, distance second.

### Unconditional instance-mean gap change (PR − FR)

| TW | Metric | Mean \(\Delta G\) (300−30) | 95% bootstrap CI |
|---|---|---|---|
| Original | \(K\) | **+1.17** | **[+0.08, +2.33]** |
| Original | Distance | −523 | [−2069, +849] |
| Relaxed | \(K\) | −0.58 | [−1.67, +0.33] |
| Relaxed | Distance | −597 | [−3966, +2542] |

**Conclusions (defensible):**

1. On **Original TW**, the FR–PR **fleet-size gap widens against PR** from 30 s to 300 s (CI excludes 0): longer budgets do **not** close the \(K\) disadvantage of PR relative to FR on this panel.
2. On **Relaxed TW**, the mean \(K\) gap change CI **includes 0**; no claim of systematic \(K\)-gap closure.
3. **Distance** gap changes have wide CIs that **include 0** on both TW panels — do **not** claim unconditional distance convergence of PR toward FR.
4. Distance must be interpreted **conditionally on \(K\)**. Instances with \(K\)-tie at **both** 30 s and 300 s: Original **2/12** (R2_10_1, R2_10_2; mean distance \(\Delta G\) ≈ −4370); Relaxed **1/12** (C1_10_2; distance \(\Delta G\) ≈ +6851). Too few for a separate CI claim — report as descriptive only (`table_gap_change_lex_conditional.csv`).

**Primary level comparisons** (absolute gaps at each budget) remain in `fr_pr_convergence_primary288/paired_summaries.csv` and `table_fr_pr_gaps_by_budget_lex.csv` (Round 4). En-route purchased energy remains lower for PR across budgets (paired CIs exclude 0).

**Iteration throughput (means over primary 288 cells):** FR ≈ 3897 / 15955 / 44398 iterations at 30/120/300 s; PR ≈ 1147 / 4801 / 12156. PR completes ~3–4× fewer ALNS iterations under equal wall-clock budgets — implementations are **not** interchangeable search engines.

Files: `final_stats/table_gap_change_30_to_300_*.csv`, `gap_change_summary.json`, `claim_verification.json`.

---

## 3. Final Schneider interpretation

| Quantity | Value | Location |
|---|---|---|
| Instances retained | **56** | `schneider/` |
| Attempts @30 s | **336** | `table_main30_feasibility.csv` |
| Feasible runs | **264** | same |
| Infeasible runs | **72** | 12 instances × 3 seeds × 2 methods |
| Paired both-feasible instances | **44** | `paired_main30_summaries.csv` |

**Conditional FR vs PR (44 instances, seeds 0–2):** charged-energy mean difference favours PR (CI excludes 0); distance CI overlaps 0; \(K\) slightly higher for PR (CI excludes 0). See `paired_main30_summaries.csv`.

**Feasibility investigation (Round 4, retained):** `replay_route` (not broken solo `replay_solution`) shows station-insertion probes succeed for customers that fail direct depot–customer–depot walks. Classification: **construction/search failure under the current solver**, **not** a mathematical proof that official Schneider instances are infeasible (esp. under alternative distance rounding). File: `schneider/feasibility_investigation_round4.json`.

**Forbidden claims:** Do **not** claim the solver solved all 56 instances. Do **not** claim Schneider validates heterogeneous price/waiting benefits (`schneider/NEUTRAL_CHARGING.md`).

---

## 4. Verified Hanif scientific claims

| Claim | Status | Evidence |
|---|---|---|
| 60 instances × 2 TW = 120 cells; seeds 0–2; 7 methods → 2520 rows | **Verified** | `hanif_main_seeds012/runs.csv` |
| PhaseII-Heterogeneous ≡ Phase I on \(K\), distance, charged energy in **350/360** matched cases | **Verified** | `hco_contribution_audit/`; Round 5 `claim_verification.json` |
| PhaseI → NoHCO: many distance/energy improvements (search without HCO) | **Verified** | `summary_counts.csv` (e.g. distance improved 69 / unchanged 291) |
| NoHCO → Het: mostly unchanged; Het does not dominate NoHCO | **Verified** | same |
| HCO-Detour replacement is **distance-led geometric**, not pure detour ranking throughout | **Documented** | `HCO_DETOUR_VERIFICATION.md` |
| Energy: observed en-route charge vs reconstructed depot replenishment labeled | **Verified** | `table_energy_coverage.csv`; reconstruction note |
| Linear partial charging ≠ nonlinear charging-speed benefits | **Clarify in text** | Model uses constant rate; PR effect is just-enough quantity, not nonlinear \(r(SoC)\) |
| FR vs PR search implementations / iteration throughput differ | **Verified** | primary288 `alns_iterations` means above |
| Absolute ALNS-HCO: limited Phase-II aggregate improvement vs Phase I | **Verified** | 350/360 frozen; eps=0 distance gate (`phase2_het_diagnosis/DIAGNOSIS.md`) |

**Economic provenance:** Observed public `charging_expense` ≠ reconstructed depot energy × hypothetical tariff ≠ wage×duration scenarios. Post-hoc only (`economic_sensitivity/ASSUMPTIONS.json`).

---

## 5. Manuscript materials produced

| Deliverable | Path |
|---|---|
| This report | `ROUND5_FINAL_REPORT.md` |
| Manuscript revision guide (ready-to-insert text) | `MANUSCRIPT_REVISION_GUIDE.md` |
| Point-by-point reviewer responses | `RESPONSE_TO_REVIEWERS.md` |
| Table/figure manifest | `FINAL_TABLE_FIGURE_MANIFEST.csv` |
| Gap-change & claim JSON | `final_stats/` |
| Full pytest log | `final_stats/pytest_full_suite.txt` |

No `.tex`/`.docx` manuscript was present in the repository; insertion text is organized by section headings in the guide.

**Figure 5 attribution:** Confirmed from `outputs/figures/final_figure_manifest.md` — Figures 4 and 5 are **PhaseII-NoHCO → PhaseII-Heterogeneous** (HCO effect), not FR→PR.

---

## 6. Validation executed this round

| Check | Result |
|---|---|---|
| Primary 288 / cell 24 / matched seeds = 2 | Asserted in Round 4; used exclusively for gap Δ |
| Hanif 2520 / Schneider 336 / 264 feasible / 44 paired | Re-asserted Round 5 |
| PhaseII≡PhaseI aggregates 350/360 | Re-asserted |
| Energy residual & depot fields present | Re-asserted |
| Manifest paths exist | 21/21 |
| Protected algo files vs `origin/main` | **OK** (`verify_no_algo_diff.py`) |
| Full pytest | **52 passed, 10 skipped** (`pytest_full_suite.txt`) |

**Infrastructure-only fix:** `src/__init__.py` now executes hyphenated-package `__init__.py` so `BENCHMARK` / `SOLVER_VERSION` export correctly. This was a pre-existing import bug blocking `tests/test_hanif_carefi.py` collection on `main` as well. **No** protected solver/HCO files changed.

---

## 7. Explicit unresolved issues (not manufactured)

1. **Route identity** PhaseII-Het vs Phase I: aggregate metrics match 350/360; final node sequences / route checksums are **not** in the publication CSV → route identity remains **unproven**.
2. **Schneider distance-rounding sensitivity** vs published BKS: not performed; numerical BKS matching **not** claimed.
3. **Lex-conditional distance CIs** for \(K\)-tie-both: sample size 1–2 instances — descriptive only.
4. **Manuscript `.tex` file** not in repo — authors must paste from `MANUSCRIPT_REVISION_GUIDE.md`.
5. **Wage / depot-tariff optimization:** post-hoc scenarios only; solver does not optimize those objectives.

---

## 8. Round 5 checklist

| Task | Status |
|---|---|
| Matched 30→300 FR–PR gap CIs by TW | **Done** |
| Archive/label obsolete status | **Done** |
| Schneider 56/336/264/72/44 + search≠proof | **Done** |
| Hanif claim verification vs CSV | **Done** |
| Detour wording correction note | **Retained** |
| Manuscript guide + Response to Reviewers | **Done** |
| Full test suite + algo-diff | **Done** |
| Commit + push `reviewer-revision-2026` | After this report |
