# HCO activity CSV caveat

`outputs/raw/hco_activity.csv` and `outputs/analysis/hco_activation_analysis.csv`
were produced **before** a small activity-counter fix (per-call outcome flags
were not reset before the final input/output comparison, so some calls could be
counted as both an improvement and a tie).

**Do not quote in the manuscript:**

- `improvement_rate`
- `objective_improvements` / `objective_ties` / `rejected_worse_plans`
- insertion / replacement / removal totals derived from those rows

**Unaffected (safe to use):** final routes and all solution-level metrics
(`C_ch`, `T_wait`, `T_ch`, `D`, station-selection tables, paired comparisons,
Figures A–F / S1).

The counter logic is fixed in `src/alns-hco/hco.py` (`_reset_call_outcome_flags`
before finalize; structural counts taken only from `diff_charging_plans`).
Regenerating `hco_activity.csv` would require re-instrumented HCO runs and is
optional for the paper rewrite.
