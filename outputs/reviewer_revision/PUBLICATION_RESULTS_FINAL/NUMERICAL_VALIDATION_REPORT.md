# Numerical validation report

## Protocol checks
- Hanif raw rows: **6720** (expect 6720) ✓  
- Seven-method × 60 × 2 TW × 5 seeds: **4200** (expect 4200) ✓  
- GI rows: 120 (seed 0 only; deterministic) ✓  
- Schneider: 336 / feas 264 / infeas 72 ✓  
- Convergence primary: 288 seeds [0, 1] ✓  
- HCO five-seed identical aggregates: **584/600** ✓  

## Archive reconciliation (`outputs/tables/` vs recomputed)
- Matched cells: **144**  
- Mismatches: **0**  
- Detail: `validation_vs_archived_tables.csv`  

All checked overall/partial cells match archived five-seed publication tables.

## Energy identities
- Reconstruction: \(E_{\mathrm{drive}}=D\cdot r\), \(E_{\mathrm{en}}=E_{\mathrm{ch}}\), \(E_{\mathrm{depot}}=E_{\mathrm{drive}}-E_{\mathrm{en}}\), \(E_{\mathrm{return}}=K B_{\max}+E_{\mathrm{en}}-E_{\mathrm{drive}}\).  
- Coverage/flags: `FINAL_TABLES/T6_energy_coverage.csv`  
- Observed vs reconstructed clearly labelled in `T6_energy_accounting.csv`.

## Duplicates / keys
- Matched (instance, tw, method, seed) uniqueness asserted for Hanif stochastic methods (5 seeds).  
- No mixing of 3-seed and 5-seed estimates in primary Hanif tables.

## Algorithm protection
- `scripts/verify_no_algo_diff.py` must remain OK vs `origin/main`.

## Word manuscript
- {"updated": true, "path": "outputs\\reviewer_revision\\PUBLICATION_RESULTS_FINAL\\ALNS-HCO-PAPER_RESULTS_REVISED.docx", "updated_cells": 40}

## Figure visual inspection
- Figure 2: Forest plots OK; all charging metrics left of zero; shared xlim; CIs present.
- Figure 3: Regenerated from five-seed fixed-route paired table.
- Figure 4: Original TW shows modest NoHCO?Het reductions in Cch/Twait/Tch with D�0; Relaxed near zero � consistent with limited HCO effect.
- Figure 5: Family means only; C1 Original large negative Cch; no fabricated family CIs.
