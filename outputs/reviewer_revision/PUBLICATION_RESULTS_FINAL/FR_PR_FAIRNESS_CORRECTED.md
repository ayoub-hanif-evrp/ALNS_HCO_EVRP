# FR versus PR — corrected fairness audit

## Code-path facts (verified)

### ALNS-PR (`src/alns-hco/alns_pr.py`)
- If `initial is None`, calls `solve_greedy(instance, deadline=40.0)` **before** the ALNS timer.
- ALNS search timer starts after GI; returned `runtime_s = gi_runtime_s + search_s`.
- Repair uses `greedy_reinsert_just_enough` (just-enough partial recharge).
- Phase I (`solve_phase1`) wraps `solve_alns_pr`.

### ALNS-FR (`src/care-fi/benchmarks/alns.py` via experiment runners)
- Experiment runners call `solve_greedy(..., deadline=40.0)` then `solve_alns(..., time_limit_s=budget)`.
- Charge policy: full recharge when a station is required.
- Reported solver runtime typically **includes** GI + ALNS wall-clock.

### Correction of prior audit error
The previous statement that ALNS-PR has “no separate GI stage” was **incorrect**.
Both FR and PR use greedy initialization with a 40 s GI deadline when no start routes are supplied.

## Budget interpretation
| Quantity | Evidence |
|---|---|
| Nominal ALNS search limit | Equal by protocol (`time_limit_s` / Phase-I budget argument) |
| End-to-end including GI | Both include GI; not claimed identical unless measured split is reported |
| Iteration throughput | **Not equal** — primary288 means: `{'ALNS-FR|30s': 3897.3541666666665, 'ALNS-FR|120s': 15954.666666666666, 'ALNS-FR|300s': 44398.270833333336, 'ALNS-PR|30s': 1146.75, 'ALNS-PR|120s': 4800.5625, 'ALNS-PR|300s': 12156.083333333334}` |

FR and PR also differ in destroy/repair implementations (care-fi ALNS vs alns-hco Phase-I operators). The primary FR/PR benchmark therefore compares **two search configurations**, not a pure recharge-policy intervention.

## Controlled same-route recharge-policy experiment (NEW)
File: `FR_PR_TRUE_FIXED_ROUTE_POLICY_EXPERIMENT.csv`

Design: shared Phase-I node sequences (customers **and** station visit locations fixed). Only station charge alphas change:
- **Partial:** archived Phase-I charge labels (authoritative just-enough decisions)
- **Full:** identical sequences with alpha = 1.0 at every true charging-station stop

Forward `just_enough_labels()` recomputation does not reproduce archived feasible Phase-I labels; archives are the partial plan.

### Spanning DEV subset (['C1_10_1', 'C2_10_1', 'R1_10_1', 'R2_10_1', 'RC1_10_1', 'RC2_10_1'] × TW × seeds 0–1; n=24)
| Outcome | Count |
|---|---|
| Partial feasible | 24 |
| Both feasible | 1 |
| Partial OK but full infeasible (counterfactual) | 23 |

### All seed-0/1 Phase-I archives (predeclared; n=240)
| Outcome | Count |
|---|---|
| Partial feasible | 240 |
| Both feasible | 42 |
| Partial OK but full infeasible (counterfactual TW/horizon) | 198 |

When full recharge is infeasible on a partial-feasible route, that is reported as **counterfactual infeasibility**, not a zero-cost result.
Paired differences/CIs: `FINAL_TABLES/FR_PR_TRUE_POLICY_paired_summaries.csv` (both-feasible only).

**This experiment replaces** the misleading `FR_PR_fixed_route_recharge_probe.csv` (which compared HCO-Detour vs HCO-Heterogeneous).

Keep this controlled policy study **separate** from the full-solver five-seed FR/PR benchmark.
