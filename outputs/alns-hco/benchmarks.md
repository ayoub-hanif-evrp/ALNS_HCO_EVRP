# ALNS-HCO vs ALNS, VNS, SA, and GI

Same 120 Hanif instances (60 Original TW + 60 Relaxed TW), same replay.
GI, ALNS, VNS, SA, and ALNS-HCO are **our implementations**.
None of these is a reproduction of Schneider VNS–TS, Keskin ALNS, or Wang HCO.

Budgets: GI ~1 s; ALNS / VNS / SA = 15 s after a shared greedy start;
ALNS-HCO = the same 15 s ALNS, then just-enough fleet compression (~8 s) and 15 s Phase II (seed 0).

## Feasibility

A solution is feasible only if every customer is served exactly once, battery never goes
negative, every time window and the capacity/horizon limits hold. One violation makes the
instance unsolved. All five methods solved **120/120** with 0 unserved and 0 hard violations.

GI / ALNS / VNS / SA use **full recharge** when a station is required.
ALNS-HCO Phase I copies that ALNS, then relabels with just-enough partial charge,
compresses the fleet, and continues ALNS with lex (K, distance, charging).

## Original TW

| Method | n | Feasible | Mean K | Mean distance | Mean duration | Mean ch. time | Mean ch. cost | Mean station wait | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALNS-HCO | 60 | 60/60 | 68.68 | 47270.0 | 163958.9 | 12713.6 | 7838.6 | 864.7 | 32.53 |
| ALNS | 60 | 60/60 | 71.33 | 48951.8 | 173948.6 | 20454.8 | 13122.0 | 1163.2 | 15.01 |
| VNS | 60 | 60/60 | 76.32 | 50402.8 | 180041.6 | 20272.1 | 12545.4 | 1000.8 | 15.00 |
| SA | 60 | 60/60 | 76.73 | 51016.0 | 182900.4 | 21174.4 | 13305.4 | 1139.4 | 15.00 |
| GI | 60 | 60/60 | 77.17 | 59123.7 | 181686.8 | 24313.0 | 14963.7 | 908.9 | 0.97 |

ALNS-HCO vs each method (strictly better / tie / worse), lower is better:

| vs | K | Distance | Charging cost |
|---|---:|---:|---:|
| ALNS | 39/21/0 | 59/0/1 | 57/0/3 |
| VNS | 46/12/2 | 51/0/9 | 47/0/13 |
| SA | 47/13/0 | 54/0/6 | 55/0/5 |
| GI | 47/13/0 | 60/0/0 | 60/0/0 |

### Mean K by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| ALNS-HCO | 108.2 | 40.8 | 106.8 | 28.4 | 94.6 | 33.3 |
| ALNS | 112.4 | 45.1 | 108.4 | 29.5 | 95.0 | 37.6 |
| VNS | 122.0 | 50.0 | 111.9 | 29.7 | 104.4 | 39.9 |
| SA | 123.0 | 50.3 | 112.4 | 29.7 | 105.1 | 39.9 |
| GI | 124.4 | 50.4 | 112.8 | 29.7 | 105.8 | 39.9 |

### Mean charging cost by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| ALNS-HCO | 898.0 | 1960.1 | 6850.5 | 22115.4 | 4702.6 | 10504.9 |
| ALNS | 5512.8 | 6035.2 | 19616.0 | 22534.7 | 13707.7 | 11325.3 |
| VNS | 6136.1 | 6123.2 | 18975.9 | 21107.1 | 12067.4 | 10863.0 |
| SA | 6264.9 | 6417.1 | 19894.4 | 22081.2 | 13123.4 | 12051.5 |
| GI | 6864.6 | 8540.8 | 21695.2 | 24821.0 | 13825.6 | 14035.0 |

## Relaxed TW

| Method | n | Feasible | Mean K | Mean distance | Mean duration | Mean ch. time | Mean ch. cost | Mean station wait | Mean runtime (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ALNS-HCO | 60 | 60/60 | 59.98 | 46500.6 | 142001.8 | 15072.6 | 9183.3 | 1104.6 | 33.27 |
| ALNS | 60 | 60/60 | 62.83 | 49274.0 | 151201.2 | 23653.0 | 14793.6 | 1561.2 | 15.01 |
| VNS | 60 | 60/60 | 75.07 | 51201.1 | 170030.9 | 23533.1 | 14424.8 | 1487.9 | 15.00 |
| SA | 60 | 60/60 | 75.25 | 52792.8 | 172727.3 | 24444.1 | 15039.9 | 1549.8 | 15.00 |
| GI | 60 | 60/60 | 77.00 | 68841.6 | 184023.6 | 30693.2 | 19126.4 | 1457.5 | 1.29 |

ALNS-HCO vs each method (strictly better / tie / worse), lower is better:

| vs | K | Distance | Charging cost |
|---|---:|---:|---:|
| ALNS | 46/14/0 | 55/0/5 | 54/0/6 |
| VNS | 59/1/0 | 47/0/13 | 47/0/13 |
| SA | 59/1/0 | 57/0/3 | 54/0/6 |
| GI | 59/1/0 | 60/0/0 | 60/0/0 |

### Mean K by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| ALNS-HCO | 96.1 | 38.4 | 93.0 | 20.7 | 91.4 | 20.3 |
| ALNS | 103.5 | 43.5 | 93.5 | 22.3 | 92.0 | 22.2 |
| VNS | 134.9 | 61.5 | 102.5 | 24.1 | 102.5 | 24.9 |
| SA | 135.0 | 61.8 | 102.9 | 24.1 | 102.8 | 24.9 |
| GI | 142.4 | 63.1 | 103.6 | 24.1 | 103.9 | 24.9 |

### Mean charging cost by family

| Method | C1 | C2 | R1 | R2 | RC1 | RC2 |
|---|---:|---:|---:|---:|---:|---:|
| ALNS-HCO | 1537.9 | 1514.1 | 11708.8 | 18798.3 | 8827.1 | 12713.8 |
| ALNS | 8396.4 | 5511.1 | 24233.6 | 19082.1 | 19065.1 | 12473.4 |
| VNS | 9853.4 | 6612.5 | 22075.5 | 18547.2 | 16877.8 | 12582.1 |
| SA | 10025.5 | 6817.6 | 22748.1 | 19705.4 | 17718.4 | 13224.6 |
| GI | 11894.1 | 10575.3 | 30185.6 | 24115.8 | 21879.8 | 16107.9 |

## How to read this

EVRPTW ranking is feasible, then min \(K\), then min distance. Charging cost is third.

- Original TW mean K: ALNS-HCO 68.68 vs ALNS 71.33 (HCO smaller); paired K wins/tie/losses 39/21/0.
- Relaxed TW mean K: ALNS-HCO 59.98 vs ALNS 62.83 (HCO smaller); paired K wins/tie/losses 46/14/0.
- ALNS-HCO is the 15 s ALNS plus extra search, so wall-clock is longer. The paired K count
  above is vs the archived 15 s ALNS, not vs the ALNS prefix inside the same HCO run.

## Hyperparameters

### ALNS-HCO

- `random_seed`: `0`
- `phase1_time_limit_s`: `15.0`
- `phase2_time_limit_s`: `15.0`
- `fleet_compress_s`: `8.0`
- `phase1_engine`: `same GI + 15 s ALNS as the ALNS baseline, then just-enough route elimination`
- `phase1_charge_policy`: `baseline ALNS uses full recharge; compression/Phase II use just-enough partial charge`
- `phase1_objective`: `lexicographic (unserved, K, distance)`
- `phase2_objective`: `lexicographic (K, distance, charging expense); fleet may decrease`
- `epsilon_distance`: `0.01`
- `epsilon_duration`: `0.0`
- `fleet_constraint`: `K never increases above the Phase-I incumbent; route elimination is allowed`
- `destroy_size_min`: `4`
- `destroy_size_max`: `12`
- `phase1_destroy_operators`: `['random', 'worst_detour', 'related_shaw', 'short_route']`
- `phase2_destroy_operators`: `['random', 'worst_detour', 'related_shaw', 'short_route', 'expensive_charge', 'slow_charge_wait']`
- `repair_phase1`: `greedy cheapest feasible insertion (nearby_routes=8, positions=3, stations=3)`
- `repair_phase2`: `greedy insertion with just-enough partial charge on every station`
- `nearby_routes`: `8`
- `insert_positions`: `3`
- `nearby_stations`: `3`
- `adaptive_reaction_factor_rho`: `0.2`
- `segment_length`: `20`
- `score_new_best`: `33`
- `score_better`: `9`
- `score_accepted`: `5`
- `sa_start_temperature_factor`: `0.05`
- `sa_cooling`: `0.9995`
- `exact_oracle_threshold_theta`: `0.02`
- `fast_station_candidates`: `4`
- `exact_station_replacements`: `4`
- `accept_infeasible`: `False`
- `cache_by_customer_sequence`: `True`

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

### Greedy insertion (GI)

- `customer_order`: `increasing due time b_i, then a_i, then index`
- `nearby_routes`: `8`
- `insert_positions`: `3`
- `nearby_stations`: `3`
- `charge_policy`: `full recharge only when a nearby station is required for battery feasibility`
- `random_seed`: `None`
- `time_limit_s`: `None`
