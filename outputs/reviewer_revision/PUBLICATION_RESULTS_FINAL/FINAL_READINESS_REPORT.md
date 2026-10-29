# Final readiness report

Branch: `reviewer-revision-2026`. Author handles reviewer letters.

**Mandatory QC:** 11/11 checks passed (zero failures).

| Scientific requirement | Status | Evidence |
|---|---|---|
| Word figures unique + no broken % sentences | **SATISFIED** | Revalidated on final DOCX |
| Supplementary tables S1–S5 complete sources | **SATISFIED** | `FINAL_TABLES/WORD_SUPPLEMENTARY/` + in-document tables |
| Matching PDF from final DOCX | **SATISFIED** | OK:1282953 size=1282953 |
| Schneider BKS numerical comparison | **NOT SATISFIED** | Primary Schneider checkpoints store metrics without complete node-string route archives (sample has_routes=False). Distance convention remains unrounded Euclide |
| Hanif five-seed / HCO honesty / FR-PR caveats | **SATISFIED** | Manuscript prose + T3–T6/T8/T9 |
| pytest + algo-diff vs main | **SATISFIED** | `QC_LOGS/` |

## Remaining scientific limitations
1. Schneider published BKS gap table not claimed (route archives + rounding).
2. HCO candidate count 5 is a fixed design choice, not a proven optimum.
3. Full-solver FR/PR is not a pure recharge-policy ablation; isolation uses the fixed-route counterfactual.

## Contradiction reconciliation (25fa734 audit)
Prior `mandatory_verification.json` recorded `word_figures_unique: false` because validation ran **before** broken-percentage prose patches and used a Python PDF path that failed, while a later PowerShell Word COM PDF was pushed and readiness was updated manually. This round rebuilds the DOCX, re-validates the exact final file, regenerates the PDF from that DOCX, and writes a fresh QC log with no manual status overrides.
