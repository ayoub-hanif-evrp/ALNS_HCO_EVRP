# Schneider source and route validation

## Field-level source fidelity
Checked all 56 large instances: customer/station coordinates, demands, time windows,
service times, depot bounds, Q/C/r/g/v, and unrounded Euclidean arc probe.
**OK: 56 / 56.**
Distance convention: euclidean unrounded (float32 matrices); travel=d/v; energy=d*r.

Detail: `FINAL_TABLES/Schneider_source_field_fidelity.csv`.

## Unserved-customer diagnosis (30 s, seeds 0–2, FR & PR)
Runs: 72; feasible among them: 0.
See `Schneider_hard_unserved_customer_ids.csv` and `Schneider_hard_missing_consistency.csv`.

120 s diagnostic (prior round): 72/72 still infeasible with unserved=1 — longer time did not resolve failures.
