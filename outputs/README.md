# Clean Hanif ALNS-HCO revision outputs

## Reproduce

```bash
python scripts/select_phase2_epsilon.py
python scripts/run_clean_hanif_study.py --workers 8 --fresh
```

## Method names

| Name | Meaning |
|---|---|
| GI / VNS / SA | Routing baselines |
| ALNS-FR | Full recharge ALNS, 30 s |
| ALNS-PR | Partial recharge ALNS, 30 s |
| PhaseI-PR-15 | Shared Phase-I solution X_PR_15 |
| FixedRoute-HCO-Detour | HCO on frozen customer routes; detour-only selection |
| FixedRoute-HCO-Heterogeneous | HCO on same routes; detour+price+wait+rate |
| PhaseII-Detour | 15 s charging-aware ALNS + HCO-Detour from shared Phase I |
| PhaseII-Heterogeneous | Same with HCO-Heterogeneous (final ALNS-HCO) |

## Locations

- Raw: `outputs/raw/solution_results.csv`, `station_visits.csv`, `route_results.csv`
- Shared Phase I: `outputs/shared_phase1/`
- Config: `outputs/config/frozen_experiment_config.json`
- Stats: `outputs/statistics/`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/` (PNG only)
- Validation: `outputs/validation/validation_report.txt`
- RQs: `outputs/analysis/research_questions.md`
- Report: `outputs/final_experiment_report.md`

Prior manuscript package: `outputs/legacy_pre_revision/` and `archive/pre_clean_revision_*`.
