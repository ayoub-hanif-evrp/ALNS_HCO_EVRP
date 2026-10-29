# Fixed-route full vs partial — independent verification

## Design (kept separate from full-solver ALNS-FR/PR)
- Same Phase-I **customer order and station visit locations**.
- Partial = archived Phase-I charge labels; Full = α=1.0 at every true CS stop.
- A free full-recharge ALNS could select **different** routes; this experiment isolates recharge quantity only.

## CSV outcomes (predeclared seed-0/1 archives)
| Outcome | Count |
|---|---:|
| Rows | 240 |
| Partial feasible | 240 |
| Both feasible | 42 |
| Partial OK, full infeasible (counterfactual) | 198 |

## Independent re-replay
- Stratified re-replay checks: 83
- Feasibility/label mismatches: 0
- Energy identity E_depot=E_drive−E_enroute on checked columns: 166 OK cells
- Instance-level paired CI recomputation match: 10/10

Inference unit for CIs: **base instances** (seed-averaged where both seeds both-feasible).
