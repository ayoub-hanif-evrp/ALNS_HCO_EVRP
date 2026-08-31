# Hanif metaheuristic benchmark

No published solver has a numerical table on this 1000-customer Hanif set.
GI, ALNS, VNS, and SA are **our implementations** on the same instances, replay, and vehicle data as CARE-FI.
They are not reproductions of Schneider VNS–TS or Keskin ALNS.

Budget: 15 s per search method after a shared greedy start. Seed: 0.

## Feasibility

Every accepted solution is checked by the same sequential replay as CARE-FI (`src/hanif/replay.py`).
A solution is feasible only if **all** of the following hold:

- every customer is served **exactly once** (no unserved, no duplicates)
- battery never goes negative on any arc
- every customer start is inside `[a_i, b_i]`
- load never exceeds vehicle capacity
- return to the depot is inside the planning horizon
- charging labels are a portion in `[0, 1]`

Moves that fail replay are discarded. GI, VNS, and SA never keep an infeasible incumbent.
ALNS discards any repair with unserved customers or hard violations; simulated-annealing steps
may accept a **worse feasible** distance, never a constraint violation.
These methods are **not exact**: they can miss a better feasible solution, but they do not
report a solution that violates the hard constraints above.

GI / ALNS / VNS / SA use **full recharge** when a station is inserted.
CARE-FI numbers below are **Stage 2** (customer sequences frozen, charge portions refined).
CARE-FI Stage 1 uses the same full-recharge policy as the four methods; those charging
figures are in the Stage-1 note after each TW block.

## Original TW

| Method | n | Feasible | Unserved | Hard viol. | Mean K | Mean distance | Mean duration | Mean ch. visits | Mean charged energy | Mean ch. time | Mean ch. cost | Mean station wait | Mean customer wait | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CARE-FI | 60 | 60/60 | 0 | 0 | 76.82 | 58771.1 | 175830.6 | 60.1 | 19002.1 | 17289.8 | 9870.1 | 831.0 | 62272.1 | 1.14 |
| GI | 60 | 60/60 | 0 | 0 | 77.17 | 59123.7 | 181686.8 | 61.0 | 27763.3 | 24313.0 | 14963.7 | 908.9 | 60674.6 | 0.97 |
| ALNS | 60 | 60/60 | 0 | 0 | 71.33 | 48951.8 | 173948.6 | 75.9 | 23829.1 | 20454.8 | 13122.0 | 1163.2 | 66693.9 | 15.01 |
| VNS | 60 | 60/60 | 0 | 0 | 76.32 | 50402.8 | 180041.6 | 66.7 | 23224.9 | 20272.1 | 12545.4 | 1000.8 | 71697.5 | 15.00 |
| SA | 60 | 60/60 | 0 | 0 | 76.73 | 51016.0 | 182900.4 | 75.2 | 24433.6 | 21174.4 | 13305.4 | 1139.4 | 72890.1 | 15.00 |

CARE-FI Stage 1 (full recharge, same charge policy as GI/ALNS/VNS/SA) vs Stage 2 (table above):

- mean charging time 24410.8 → 17289.8
- mean charging cost 15013.0 → 9870.1
- mean charged energy 27865.0 → 19002.1
- mean duration 180973.7 → 175830.6
- mean distance 59055.5 → 58771.1 (K frozen at 76.82)

### Mean K by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| CARE-FI | 124.1 | 49.7 | 112.3 | 29.6 | 105.7 | 39.5 |
| GI | 124.4 | 50.4 | 112.8 | 29.7 | 105.8 | 39.9 |
| ALNS | 112.4 | 45.1 | 108.4 | 29.5 | 95.0 | 37.6 |
| VNS | 122.0 | 50.0 | 111.9 | 29.7 | 104.4 | 39.9 |
| SA | 123.0 | 50.3 | 112.4 | 29.7 | 105.1 | 39.9 |

### Mean charging cost by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| CARE-FI | 1942.8 | 5280.5 | 10984.7 | 22488.4 | 6486.8 | 12037.6 |
| GI | 6864.6 | 8540.8 | 21695.2 | 24821.0 | 13825.6 | 14035.0 |
| ALNS | 5512.8 | 6035.2 | 19616.0 | 22534.7 | 13707.7 | 11325.3 |
| VNS | 6136.1 | 6123.2 | 18975.9 | 21107.1 | 12067.4 | 10863.0 |
| SA | 6264.9 | 6417.1 | 19894.4 | 22081.2 | 13123.4 | 12051.5 |

## Relaxed TW

| Method | n | Feasible | Unserved | Hard viol. | Mean K | Mean distance | Mean duration | Mean ch. visits | Mean charged energy | Mean ch. time | Mean ch. cost | Mean station wait | Mean customer wait | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CARE-FI | 60 | 60/60 | 0 | 0 | 74.72 | 67586.3 | 174088.7 | 91.9 | 26549.2 | 24044.9 | 13767.2 | 1292.4 | 44498.4 | 1.55 |
| GI | 60 | 60/60 | 0 | 0 | 77.00 | 68841.6 | 184023.6 | 97.8 | 35285.6 | 30693.2 | 19126.4 | 1457.5 | 46364.6 | 1.29 |
| ALNS | 60 | 60/60 | 0 | 0 | 62.83 | 49274.0 | 151201.2 | 103.8 | 27269.8 | 23653.0 | 14793.6 | 1561.2 | 40035.8 | 15.01 |
| VNS | 60 | 60/60 | 0 | 0 | 75.07 | 51201.1 | 170030.9 | 99.6 | 26856.0 | 23533.1 | 14424.8 | 1487.9 | 57137.5 | 15.00 |
| SA | 60 | 60/60 | 0 | 0 | 75.25 | 52792.8 | 172727.3 | 103.7 | 27941.5 | 24444.1 | 15039.9 | 1549.8 | 57266.9 | 15.00 |

CARE-FI Stage 1 (full recharge, same charge policy as GI/ALNS/VNS/SA) vs Stage 2 (table above):

- mean charging time 30640.4 → 24044.9
- mean charging cost 19031.5 → 13767.2
- mean charged energy 35177.6 → 26549.2
- mean duration 179977.3 → 174088.7
- mean distance 67811.9 → 67586.3 (K frozen at 74.72)

### Mean K by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| CARE-FI | 139.6 | 61.2 | 99.9 | 23.6 | 100.1 | 23.9 |
| GI | 142.4 | 63.1 | 103.6 | 24.1 | 103.9 | 24.9 |
| ALNS | 103.5 | 43.5 | 93.5 | 22.3 | 92.0 | 22.2 |
| VNS | 134.9 | 61.5 | 102.5 | 24.1 | 102.5 | 24.9 |
| SA | 135.0 | 61.8 | 102.9 | 24.1 | 102.8 | 24.9 |

### Mean charging cost by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| CARE-FI | 4417.2 | 7115.5 | 20018.5 | 22699.8 | 13575.0 | 14777.3 |
| GI | 11894.1 | 10575.3 | 30185.6 | 24115.8 | 21879.8 | 16107.9 |
| ALNS | 8396.4 | 5511.1 | 24233.6 | 19082.1 | 19065.1 | 12473.4 |
| VNS | 9853.4 | 6612.5 | 22075.5 | 18547.2 | 16877.8 | 12582.1 |
| SA | 10025.5 | 6817.6 | 22748.1 | 19705.4 | 17718.4 | 13224.6 |

## Hyperparameters

### Greedy insertion (GI)

- `customer_order`: `increasing due time b_i, then a_i, then index`
- `nearby_routes`: `8`
- `insert_positions`: `3`
- `nearby_stations`: `3`
- `charge_policy`: `full recharge only when a nearby station is required for battery feasibility`
- `random_seed`: `None`
- `time_limit_s`: `None`

### Adaptive large neighbourhood search (ALNS)

- `random_seed`: `0`
- `time_limit_s`: `15.0`
- `destroy_size_min`: `4`
- `destroy_size_max`: `12`
- `destroy_operators`: `['random', 'worst_detour', 'related_shaw', 'short_route']`
- `repair`: `greedy cheapest feasible insertion (nearby_routes=8, positions=3)`
- `adaptive_reaction_factor_rho`: `0.2`
- `segment_length`: `20`
- `score_new_best`: `33`
- `score_better`: `9`
- `score_accepted`: `5`
- `sa_start_temperature_factor`: `0.05`
- `sa_cooling`: `0.9995`
- `charge_policy`: `full recharge when a station is required`
- `accept_infeasible`: `False`

### Variable neighbourhood search (VNS)

- `random_seed`: `0`
- `time_limit_s`: `15.0`
- `k_max`: `3`
- `neighbourhoods`: `['relocate', 'swap', 'intra_route_2opt']`
- `shake`: `k random relocations, k = 1..k_max`
- `first_improvement`: `True`
- `attempts_per_neighbourhood`: `12`
- `charge_policy`: `full recharge when a station is required`

### Simulated annealing (SA)

- `random_seed`: `0`
- `time_limit_s`: `15.0`
- `initial_temperature`: `200.0`
- `cooling_rate`: `0.995`
- `minimum_temperature`: `0.5`
- `move_probabilities`: `{'relocate': 0.5, 'swap': 0.3, 'two_opt': 0.2}`
- `accept_infeasible`: `False`
- `charge_policy`: `full recharge when a station is required`

CARE-FI uses the two-stage method in `src/hanif/` (due-time insertion, most-constrained-first
route elimination, then station/portion refinement). See `METHOD.md`.
