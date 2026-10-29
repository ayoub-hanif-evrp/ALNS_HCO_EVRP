# Schneider FR168 — final analysis (release audit)

## Scope distinction
- **FR168**: 56 instances × 3 seeds ALNS-FR with route archives + Java **full-recharge** external verification.
  Complete for its **132** complete/valid routes (of 168 attempts).
- **Primary 336**: original FR+PR panel (264 feasible / 72 unsuccessful) — **not** independently route-verified.
- Do **not** label FR168 as verification of the full 336 FR/PR panel.

## Counts (from archives)
- Attempts **168**; feasible **132**; unsuccessful **36**
- Externally valid **132**
- Exact vs primary **144**; discrepancies **24**
- Lex BKS among valid runs: {'FLEET_TIE': 34, 'FLEET_WORSE_THAN_BKS': 98}

## Instance-level aggregation
Instance summary uses **best externally valid seed by lexicographic (K, then distance)**.
This is **best-of-three selection**, **not** seed averaging.
Optional descriptive means over feasible seeds are reported separately in the same CSV.

## BKS provenance
- Inventory: Erdelić & Carić (2022) consolidating earlier FR literature (`Schneider_BKS_FR_provenance_56.csv`, n=56).
- Consolidated BKS = lex min(K,D) of literature row vs Erdelić ALNS.
- Our verified distances: unrounded Euclidean; verifier total to 3 dp. Gap % assumes inventory D is comparable; residual rounding uncertainty remains.
- Incomplete/invalid runs remain visible and are excluded from numerical gap calculations.

## PR
Independent PR route-level external verification is **not claimed**.
Not explicitly requested in the available repository reviewer-concern summary; original reviewer-letter wording was not independently verified in this audit.
Java verifier must not be used to claim PR validation.

## Primary 336
Official FR/PR coverage: **336** attempts, **264** feasible, **72** unsuccessful.
