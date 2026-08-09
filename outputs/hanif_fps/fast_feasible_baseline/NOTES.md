# FastFeasibleBaseline (frozen)

Frozen snapshot of the first full 120-instance Hanif Det-FPS run
(dense EDF construct + coarse shrink + light repair).

**Do not call this fleet-minimized.** Feasibility/scalability baseline only.

## Semantics (OUR_EXTENDED)

| Attribute | Status |
|-----------|--------|
| Q=600, h=1 | used |
| Heterogeneous `charging_rate` | used |
| `wait_mu` as deterministic expected wait | used |
| `wait_sigma` stochastic wait | **not** used |
| Charging price / `price_sigma` | **not** in objective (`lambda_charging_price=0`) |
| `density_score` | **not** used in search |

Published Hanif paper feasibility verifier averages at Q=600 (reference, may omit wait_mu):
A ≈ 74.2 routes, B ≈ 63.7 routes.

## This baseline

| Variant | Feasible | Mean K | Mean runtime |
|---------|----------|--------|--------------|
| A original | 60/60 | 96.72 | ~35 s |
| B relaxed | 60/60 | 80.35 | ~39 s |

Next milestone: `HanifFleetIntensification` with route elimination + A→B warm starts.
