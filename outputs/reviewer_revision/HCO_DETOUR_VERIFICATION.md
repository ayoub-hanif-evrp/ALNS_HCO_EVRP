# HCO-Detour scientific verification (Round 2)

**Rule:** HCO decision logic was **not** modified. This note verifies and documents behaviour.

## Manuscript-style claims vs implementation

| Claim | Status | Evidence |
|---|---|---|
| Detour candidate shortlist uses `(detour, station_id)` only | **Match** | `station_candidates` in `src/alns-hco/hco.py` (detour branch); tests `test_detour_candidates_ignore_price_and_wait`, `test_detour_hco_plan_invariant_to_price_wait_permutation` |
| Detour never uses \(\bar P, \bar W, r\) for ranking / acceptance | **Match for shortlist + final non-worsening** | Final score `_geometric_score` = `(D, N_ch, station-id sequence)` |
| Feasibility still respects physical charge rate and waiting | **Match** | `replay_route` / `replay_explicit` use `charging_rate` and `expected_wait`; test `test_feasibility_replay_uses_actual_wait_and_rate` |
| Replacement / improve pass ranks by detour only | **Does not fully match wording** | `_select_best_replace` uses distance-led keys `(D, N_ch, …)` for detour mode (Round-1 audit, lines ~493–545). Documented by `test_document_replacement_uses_geometric_distance_not_pure_detour` |

## Interpretation for the revision

- **Preference decisions** among geometrically ranked candidates are invariant to permuting station prices and waits under `mode="detour"`.
- **Feasibility and duration accounting** still depend on actual station rates and waits (and prices for expense), because they go through the authoritative replay.
- The internal **replacement tournament** optimizes a geometric objective dominated by total route distance (and visit count), with insertion-arc detour only as a tertiary key. Authors should either (a) revise the manuscript text to match this implementation, or (b) change the oracle in a later round (not Round 2).

## Tests

```text
pytest tests/test_hco_detour_science.py tests/test_hco_final_fix.py -q
```
