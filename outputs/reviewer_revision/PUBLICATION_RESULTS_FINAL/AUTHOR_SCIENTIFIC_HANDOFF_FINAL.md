# AUTHOR SCIENTIFIC HANDOFF — FINAL

**Branch:** `reviewer-revision-2026`  
**Purpose:** Scientific evidence handoff only (not manuscript text; not a response-to-reviewers draft).

**Coverage note:** Reviewer coverage checked against the repository's documented reviewer-concern summary; exact original reviewer-letter wording not independently verified.

---

## 1. Confirmed results you may report

### Hanif (from `outputs/raw/solution_results.csv`)
- Raw records: **6720**
- Seven-method panel: **3720** (= 120 GI + 6×600)
- Stochastic methods: seeds 0–4; GI: seed 0 only
- Original and Relaxed TW analyzed separately

### FR/PR convergence
- Primary runs: **288** at budgets 30/120/300 s
- Fleet-size-first lexicographic interpretation in T8 lex tables

### HCO
- PhaseI→Heterogeneous aggregate identity (K, distance, charged energy): **584/600** (route identity not claimed)
- NoHCO→Heterogeneous charging expense: **16 / 501 / 83**
- PhaseII-Plain vs Heterogeneous: Plain lower distance on **60/60** instances each TW; equal K — **retain this unfavorable finding**
- Fixed-route policy: **240** / **42** both feasible / **198** forced-full infeasible on fixed routes (not a free FR failure claim)

### Schneider
- Primary FR/PR 30 s: **336** / **264** feasible / **72** unsuccessful
- FR168 archive: **168** / **132** feasible / **132** externally valid (Java **full-recharge**)
- Lex BKS among valid FR runs: {'FLEET_TIE': 34, 'FLEET_WORSE_THAN_BKS': 98} (generally worse than inventory; not competitive-performance claims)
- Reproducibility vs primary FR metrics: exact **144**; discrepancies **24**

---

## 2. Reviewer concern → primary artifact

| Concern | Table / figure |
|---|---|
| Equal-budget FR/PR + convergence | Figure 2; T8_*; FigS_* |
| Fleet before distance | T8 lex; FR168 BKS tables |
| TW not i.i.d. | `paired_comparisons_all.csv` |
| Energy | T6_* |
| HCO attribution | Figures 3–5; T4_*; T5_*; Plain vs Het |
| Detour wording | `HCO_DETOUR_VERIFICATION.md` |
| FR/PR engines | `FR_PR_FAIRNESS_CORRECTED.md` |
| Schneider coverage/failures | T7_*; FR168 tables |
| Schneider FR external + BKS | `Schneider_FR168_*`; `Schneider_BKS_FR_provenance_56.csv` |
| Economics post-hoc | T9_* |
| Benchmark description | `benchmark_characteristics/*` |

Full checklist: `FINAL_REVIEWER_EVIDENCE_CHECKLIST.csv`

---

## 3. Necessary qualifications
- FR/PR full-solver comparison ≠ pure charging-policy isolation (use fixed-route policy experiment for controlled ablation).
- FR168 external verification covers **FR full-recharge routes only**, not the primary 336 FR/PR panel and not ALNS-PR.
- Instance-level FR168 BKS summary = **best-of-three valid seeds (K then distance)**, **not** averaging.
- Inventory BKS from Erdelić & Carić (2022) consolidation; residual distance-convention uncertainty remains.
- 5% discrepancy distance split is descriptive, not a significance test.
- Lower ALNS iteration counts support wall-clock throughput sensitivity; not sole-cause CPU proof.

---

## 4. Explicit unfavorable findings
- HCO Heterogeneous does **not** dominate PhaseII-Plain on distance (Plain wins 60/60 each TW).
- NoHCO→Het charging expense: **83** worsenings.
- Schneider 30 s FR: **0** fleet wins vs consolidated BKS among valid runs; **98** fleet-worse; fleet ties still show positive distance gaps.
- 24/168 FR metric discrepancies vs primary under equal nominal wall-clock.

---

## 5. Remaining limitations (cannot remove without changing the study)
1. No independent PR route-level external verifier campaign (unavailable; not asserted as original-letter requirement without letter verification).
2. Primary 336 FR/PR not fully route-archived/externally verified.
3. 30 s FR not BKS-competitive.
4. Wall-clock ALNS reproducibility under load.
5. Author-owned manuscript/declarations/letters.

---

## 6. Obsolete reports — do not copy
See `SUPERSEDED_REPORTS_INDEX.md`. Especially avoid **350/360**, **4200**, and “seed-0-only / 3-diagnostic” Schneider claims.

---

## 7. Exact paths
- Package root: `outputs/reviewer_revision/PUBLICATION_RESULTS_FINAL/`
- This handoff: `.../AUTHOR_SCIENTIFIC_HANDOFF_FINAL.md`
- Numbers: `.../FINAL_VERIFIED_RESULTS_FOR_AUTHOR.md`
- Manifest: `.../RESULTS_SOURCE_MANIFEST.csv`
- FR168: `.../FINAL_TABLES/Schneider_FR168_*.csv`
- BKS provenance: `.../FINAL_TABLES/Schneider_BKS_FR_provenance_56.csv`
- QC: `.../QC_LOGS/final_release_audit.json`

Figure source verification: **True** (`QC_LOGS/figure_source_verification.json`).
Source-count lock: **True**.
