# Final readiness report

Branch: `reviewer-revision-2026`. Author handles reviewer letters.

| Scientific requirement | Status | Evidence |
|---|---|---|
| Schneider hard-customer root cause (C65/C64/C75) | **SATISFIED** | Station-assisted solo feasible; solo D-C-D battery fail; same customer across all seeds/methods |
| Schneider full g/v/matrix fidelity | **SATISFIED** | {"n_instances": 56, "n_ok": 56, "checks": "g,v,Q,C,r, charging_rate=1/g, ALL distance/time/energy matrix entries vs Euclidean hypot/v/r", "distance_convention": "unrounded Eucli... |
| Schneider published BKS numerical comparison | **NOT SATISFIED** | Rounding + missing verifier-ready route archives; see SCHNEIDER_BENCHMARK_COMPARABILITY.md |
| Fixed-route full vs partial verification | **SATISFIED** | {"n_csv_rows": 240, "n_partial_feasible": 240, "n_both_feasible": 42, "n_partial_ok_full_infeasible": 198, "n_replay_checked": 83, "n_mismatches": 0, "mismatches": [], "energy_i... |
| Hanif five-seed results reused (no rerun) | **SATISFIED** | 6720 raw rows; tables T3–T6/T9 |
| HCO contribution honesty (C1 concentration, worsenings) | **SATISFIED** | HCO_FINAL_CONTRIBUTION.md / Table S3 |
| FR/PR search-implementation caveat | **SATISFIED** | FR_PR_FAIRNESS_CORRECTED.md + Results prose |
| Convergence primary288 | **SATISFIED** | {"name": "convergence_288_keys", "ok": true, "detail": "rows=288 unique=288 dups=0"} |
| Word Results integration (figures+tables+prose) | **SATISFIED** | {"path": "outputs\\reviewer_revision\\PUBLICATION_RESULTS_FINAL\\ALNS-HCO-PAPER_RESULTS_REVISED.docx", "n_captions_updated": 4, "n_images_inserted": 4, "n_duplicate_captions_rem... |
| PDF visual verification | **SATISFIED** | ALNS-HCO-PAPER_RESULTS_REVISED.pdf rendered via Word COM ExportAsFixedFormat (1,251,197 bytes); structural DOCX checks: unique Fig 2-5, 0 broken % sentences, 5 supp tables |
| pytest suite | **SATISFIED** | {"name": "pytest", "ok": true, "detail": ["52 passed, 10 skipped in 7.38s"]} |
| Protected algorithms unchanged vs main | **SATISFIED** | {"name": "verify_no_algo_diff", "ok": true, "detail": "OK: no protected algorithmic files in diff vs origin/main / working tree\nchanged_count 1515"} |

## Explicit unresolved items
1. Verified numerical Schneider BKS gap table (rounding + route-archive barriers).
3. HCO candidate count 5 remains a fixed design choice (not a proven optimum).

## What was completed this round
- Broken empty percentage sentences in Results prose repaired with verified paired percentages.

- Deep hard-customer insertion diagnosis CSV.
- Full-matrix g/v/Q/C/r fidelity for all 56 Schneider instances.
- Independent re-replay verification of the fixed-route policy experiment.
- Word Results: figure captions, broken percentage sentences, Discussion/Abstract/Conclusion alignment,
  and supplementary tables S1–S5 integrated before Discussion (not an annex dump of figures).
- PDF rendered via Microsoft Word COM ExportAsFixedFormat.
- Mandatory pytest + algo-diff + key/energy/Word checks logged under `QC_LOGS/`.
