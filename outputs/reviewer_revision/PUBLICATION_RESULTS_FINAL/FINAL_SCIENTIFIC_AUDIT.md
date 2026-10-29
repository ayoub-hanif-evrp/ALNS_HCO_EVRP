# Final scientific audit checklist

Branch: `reviewer-revision-2026`. Core ALNS/HCO logic unchanged vs `main`.

## Evidence status
See `REVIEWER_EXPERIMENT_COVERAGE.csv` for SATISFIED / PARTIAL / NOT SATISFIED marks.

## Protocols reconfirmed
- Hanif five-seed: 6720 rows; 4200 seven-method panel.
- Schneider primary: 336 / 264 / 72 / 44.
- Convergence primary288: 288 runs seeds 0–1.
- HCO identical aggregates: 584/600.

## New computational work this round
- HCO deep contribution + family leave-one-out (offline from raw).
- HCO candidate sensitivity on development subset (new Phase-II runs; shared Phase I reused).
- Schneider source-fidelity audit (all 56).
- Schneider 120 s diagnostic on 12 hard instances (new; primary 30 s preserved).
- FR/PR fairness audit + fixed-route probe.
- Manuscript results Word revision pass.

## Limitations (explicit)
1. HCO does not show broad improvement vs NoHCO; many unchanged/worsened cases retained.
2. Parameter studies are supplementary; frozen primary settings not retuned.
3. FR vs PR is not a pure charging-policy ablation.
4. Schneider BKS numerical comparison withheld (distance convention).
5. Aggregate PhaseII≡PhaseI ≠ proven route identity.

## Execution summary (this correction round)

| Work | Result |
|---|---|
| HCO deep 5-seed audit | 584/600 PhaseI=Het aggregates; NoHCO?Het Cch 16/501/83 |
| Candidate sensitivity k?{2,5,8} | 72/72; **identical metrics**; k=5 fixed choice |
| Existing e / 15-15 panels | Ties (?Cch=0) reused; not retuned |
| FR/PR fairness | Documented non-isolated stacks; iteration disparity |
| Convergence 288 | Reconfirmed; no rerun |
| Schneider fidelity | 56/56 converter OK |
| Schneider 120s hard set | 72/72; **0 feasible** (unserved=1) |
| BKS gaps | NOT claimed (convention) |
| Word revised copy | Table 3 + captions/annex + embedded figures |
| PDF render | Blocked (no win32com); DOCX delivered |
| Tests | 52 passed, 10 skipped |
| Algo vs main | Protected files unchanged |
