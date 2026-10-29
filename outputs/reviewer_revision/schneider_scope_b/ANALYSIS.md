# Schneider Scope B — 56 ALNS-FR (seed 0, 30 s)

## Protocol
- Instances: all **56** large Schneider (LARGE_56)
- Method: ALNS-FR; seed=0; GI deadline=20.0s; ALNS budget=30.0s
- Matches `scripts/run_schneider_experiments.py` FR path (primary config).
- **Does not replace** the original primary 336-run archive.

## Reproducibility vs primary seed-0 FR checkpoints
- Exact match (feasible+K+distance): **51/56**
- Near distance match: **0/56**
- Discrepancy: **5/56**
- Scope-B feasible: **44/56** (primary seed0 FR was 44/56)

## External verifier (ghiermann/evrptw-verifier)
- Distance convention in verifier source: **unrounded Euclidean arcs**; total rounded to **3** decimals.
- Station visit ⇒ full recharge (matches FR policy).
- External valid (raw exports): **44/56**
- External valid (collapsed consecutive identical IDs): **44/56**
- Report variant valid: **44/56**
- Collapse conceals violation (forbidden for claims): **0**

## BKS comparison (lexicographic; externally valid complete solutions only)
- Fleet better / tie / worse vs inventory BKS: **0 / 10 / 34**
- Distance gaps reported for fleet ties; marked secondary when fleet differs.
- Inventory sources: Schneider et al. (2014) / Schiffer & Walther as coded in `schneider_bks.py`.

## Recommendation
Scope B reproducibility and verification coverage support optionally extending to **168 FR runs (56×3 seeds)** before considering the full 336 (do not rerun PR for this verifier).

## Artifacts
- Table: `FINAL_TABLES/Schneider_ScopeB_56_FR_seed0.csv`
- Routes: `outputs/reviewer_revision/schneider_scope_b/checkpoints/`
- Verifier logs: `QC_LOGS/scopeB_verifier_*.txt`

## Reproducibility discrepancies (5/56)
Same feasibility label for all five; stochastic ALNS search under identical seed/budget did not reproduce exact distance/K on:
- r104_21: ScopeB K=13 vs primary 12; delta_d=+22.7930
- r107_21: ScopeB K=13 vs primary 13; delta_d=+5.7523
- r110_21: ScopeB K=13 vs primary 13; delta_d=+133.2042
- r112_21: ScopeB K=13 vs primary 13; delta_d=+5.1454
- r207_21: ScopeB K=3 vs primary 3; delta_d=+7.3996

These Scope-B results are retained and reported; primary archive is not overwritten.

## Fleet-tie BKS distance gaps (externally valid)
See `FINAL_TABLES/Schneider_ScopeB_BKS_fleet_ties.csv` (10 instances). All reported gaps are positive (worse than inventory BKS distance) under 30 s seed-0 FR.

## Collapse study
Removing consecutive identical station IDs never converted an infeasible raw solution into a feasible one (n_conceal=0). Distance change under collapse is zero when consecutive stations share coordinates (same node). Raw and collapsed external validity counts match (44/56).

## Rounding evidence
Java verifier source uses unrounded Euclidean arc lengths and rounds only the **sum** to 3 decimals — compatible with our Euclidean reporting after 3-dp formatting. This does **not** prove identity with every published table that may use 2-decimal arc rounding; inventory BKS distances are compared after external validity under the verifier convention.

