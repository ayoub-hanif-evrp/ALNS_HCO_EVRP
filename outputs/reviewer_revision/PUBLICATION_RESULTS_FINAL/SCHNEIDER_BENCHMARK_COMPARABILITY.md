# Schneider FR×3 seeds (168) — final external verification

## Protocol
- 56 large instances × seeds {0,1,2} × ALNS-FR 30 s (GI 20 s) = **168** attempts
- Seed 0 reused from Scope B; seeds 1–2 newly executed with route archives
- Primary FR/PR **336** dataset **not** overwritten
- Java verifier = full-recharge only → **does not validate ALNS-PR**

## Counts
- Feasible: **132/168**
- Unsuccessful: **36/168**
- Externally valid (report variant): **132/168**
- Exact reproduce vs primary FR checkpoints: **144/168**
- Metric discrepancies vs primary: **24/168**
- Collapse conceals violation: **0**

### By seed
| Seed | Feasible | External valid | Exact repro |
|---:|---:|---:|---:|
| 0 | 44/56 | 44/56 | 51/56 |
| 1 | 44/56 | 44/56 | 45/56 |
| 2 | 44/56 | 44/56 | 48/56 |

## Lexicographic BKS (externally valid FR only)
- Fleet better / tie / worse: **0 / 34 / 98**
- Distance gaps only meaningful under fleet ties; secondary when fleets differ
- Inventory: Schneider et al. (2014) / Schiffer & Walther via `schneider_bks.py`
- Verifier distance: unrounded Euclidean arcs; total rounded to 3 decimals

## Reproducibility
See `SCHNEIDER_REPRODUCIBILITY_DIAGNOSIS.md`. Wall-clock ALNS ⇒ load-dependent iteration counts.

## Artifacts
- `FINAL_TABLES/Schneider_FR168_external_verification.csv`
- `FINAL_TABLES/Schneider_FR168_BKS_fleet_ties.csv`
- Checkpoints: `outputs/reviewer_revision/schneider_fr168/checkpoints/`
- Logs: `QC_LOGS/fr168_verifier_*.txt`

## PR route verification
Documented reviewer concerns (repository RESPONSE summary) require Schneider **coverage and honest failure reporting**, not an independent PR-specific route verifier. The Java tool is **full-recharge only** and must not be used to claim PR verification.

## Reproducibility
All 24 FR168 vs-primary discrepancies audited in `Schneider_FR168_all24_discrepancies.csv` and summarized in `SCHNEIDER_REPRODUCIBILITY_ALL24.md`.
