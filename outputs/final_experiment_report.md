# Final experiment report — corrected Hanif ALNS-HCO study

## Code changes (final HCO fix)
- Explicit charging plans `(s, e_s)`; no `None → full recharge` inside HCO.
- Full sequential battery propagation through all prior charging stops.
- **HCO-Detour is fully geometry-only:** candidate ranking by `(detour, station_id)` and
  non-worsening acceptance by `(D, N_ch, station-id sequence)` — never uses
  \(\bar P_s, \bar W_s, r_s, C_{ch}, T_{wait}, T_{ch}\).
- Heterogeneous / Phase-II charging criterion: lex `(C_ch, T_wait, T_ch, D)` at `K=K*`.
- Phase-II repair shortlists insertions, then HCO-evaluates each (or NoHCO without oracle).
- Matched control: `PhaseII-NoHCO` (same Phase-II operators as Het, no oracle).
- Corrected HCO activity from explicit input→output plan diffs.
- Cache keys: `(customers, mode, k)`; Detour and Heterogeneous never share entries.

## Frozen parameters
Embedded below from `outputs/config/frozen_experiment_config.json` at compile time
(single source of truth — do not mix with older report snapshots).

- **ε = 0.0**:
  conservative tie choice among fully feasible development-subset candidates
  `{0, 0.01, 0.02, 0.05}` (all identical on the 12-case subset). **Not** claimed superior.
- **Time allocation 15/15**: equal split of the common 30 s budget.
  Development sensitivity for 10/20, 15/15, 20/10 also tied at mean ΔCch = 0;
  **not** claimed optimal. See `outputs/analysis/time_allocation_sensitivity.csv`.

```json
{
  "benchmark": "Hanif 1000-customer EVRPTW",
  "study": "final_hco_fix_20260906",
  "n_instances_per_tw": 60,
  "customers_per_instance": 1000,
  "tw_variants": [
    "original",
    "relaxed"
  ],
  "seeds": [
    0,
    1,
    2,
    3,
    4
  ],
  "n_runs": 5,
  "alns_fr_budget_s": 30.0,
  "alns_pr_budget_s": 30.0,
  "shared_phase1_s": 15.0,
  "phase2_s": 15.0,
  "gi_deadline_s": 40.0,
  "hco_candidates_per_criterion": 5,
  "phase2_distance_tolerance": 0.0,
  "epsilon_candidates": [
    0.0,
    0.01,
    0.02,
    0.05
  ],
  "development_instances": [
    "C1_10_1",
    "C2_10_1",
    "R1_10_1",
    "R2_10_1",
    "RC1_10_1",
    "RC2_10_1"
  ],
  "methods": [
    "GI",
    "VNS",
    "SA",
    "ALNS-FR",
    "ALNS-PR",
    "PhaseI-PR-15",
    "FixedRoute-HCO-Detour",
    "FixedRoute-HCO-Heterogeneous",
    "PhaseII-NoHCO",
    "PhaseII-Detour",
    "PhaseII-Heterogeneous",
    "PhaseII-Plain"
  ],
  "python": "3.13.9",
  "platform": "Windows-11-10.0.26200-SP0",
  "processor": "Intel64 Family 6 Model 143 Stepping 8, GenuineIntel",
  "machine": "AMD64",
  "created_utc": "2026-09-05T19:18:57.299495+00:00",
  "tie_tolerance": 1e-06,
  "percentage_definition": "paired instance-level 100*(B-A)/A, then mean of those percentages",
  "no_schneider": true,
  "epsilon_selection_rule": "min_mean_delta_Cch_among_fully_feasible",
  "epsilon_selection_summary": {
    "0.0": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "n": 12
    },
    "0.01": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "n": 12
    },
    "0.02": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "n": 12
    },
    "0.05": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "n": 12
    }
  },
  "provisional": false,
  "development_seed": 0,
  "phase1_time_limit_s": 15.0,
  "phase2_time_limit_s": 15.0,
  "time_allocation": "15/15",
  "time_allocation_rule": "min_mean_delta_Cch_among_fully_feasible_then_prefer_15_15",
  "time_allocation_summary": {
    "10/20": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "imbalance": 10.0,
      "n": 12,
      "phase1_s": 10.0,
      "phase2_s": 20.0
    },
    "15/15": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "imbalance": 0.0,
      "n": 12,
      "phase1_s": 15.0,
      "phase2_s": 15.0
    },
    "20/10": {
      "ok": true,
      "mean_delta_Cch": 0.0,
      "mean_distance_ratio": 1.0,
      "imbalance": 10.0,
      "n": 12,
      "phase1_s": 20.0,
      "phase2_s": 10.0
    }
  },
  "time_allocation_note": "Equal split of the common 30s budget; not claimed optimal.",
  "phase2_repair_shortlist_size": 5,
  "fixed_route_objective": "lexicographic (C_ch, T_wait, T_ch, D); non-worsening"
}
```

## Validation
- Failed jobs: 0
- Feasible successful rows: 6720
- Validation errors listed: 0
- See `outputs/validation/validation_report.txt` (includes `pytest tests/test_hco_final_fix.py`).

## Required tables
- `outputs/tables/table_overall_methods.csv`
- `outputs/tables/table_partial_recharging.csv`
- `outputs/tables/table_fixed_route_heterogeneity.csv`
- `outputs/tables/table_phase2_nohco_vs_hco.csv`
- `outputs/tables/table_phase2_detour_vs_heterogeneous.csv`
- `outputs/tables/table_phase1_to_phase2.csv`
- `outputs/tables/table_station_selection.csv`
- `outputs/tables/table_family_results.csv`

## Required figures (PNG only)
- `outputs/figures/figA_partial_recharging_effect.png`
- `outputs/figures/figB_fixed_route_heterogeneity.png`
- `outputs/figures/figC_station_characteristics.png`
- `outputs/figures/figD_phase1_to_phase2.png`
- `outputs/figures/figE_nohco_vs_hco.png`
- `outputs/figures/figS1_phase2_detour_vs_heterogeneous.png` (supplement)
- `outputs/figures/figF_family_heterogeneous_effect.png`

## Research questions
See `outputs/analysis/research_questions.md`. Report negative results as negative.

## HCO activity diagnostics
Solution-level results are frozen and trusted. Do **not** quote
`improvement_rate` / tie / reject / insert-replace totals from the current
`hco_activity.csv` — see `outputs/analysis/HCO_ACTIVITY_CAVEAT.md`.
The counter bug is fixed in code; regenerating that CSV is optional and does
not change routes or Figures A–F.
