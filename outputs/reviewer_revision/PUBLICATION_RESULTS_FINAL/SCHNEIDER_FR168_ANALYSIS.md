# Schneider FR168 — final analysis

## Counts (authoritative)
- 168 attempts; feasible **132**; unsuccessful **36**
- Externally valid **132** (Java **full-recharge** verifier)
- Exact vs primary **144/168**; discrepancies **24/168** (all audited)
- Lex BKS per valid run: better/tie/worse = **0 / 34 / 98**
- Primary 336 FR/PR unchanged; this is **FR-only** route verification (not PR; not the full 336 panel)

## Tables
- `Schneider_FR168_external_verification.csv`
- `Schneider_FR168_BKS_by_run.csv`
- `Schneider_FR168_BKS_instance_summary.csv` (seeds within instance)
- `Schneider_FR168_all24_discrepancies.csv`

## PR
Independent PR route verification is **not explicitly required** by the documented reviewer summary; retained as a transparent limitation. Do not use the full-recharge Java verifier on PR.
