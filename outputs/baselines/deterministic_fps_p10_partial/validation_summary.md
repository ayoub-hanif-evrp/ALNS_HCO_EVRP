# Deterministic-FPS validation (6 instances)

Decision: **PARTIAL** — The solver works, but specific fleet/feasibility/runtime problems remain.

Budget: 720.0s / instance | Total runtime: 4242.6s | pytest rc=0

| Instance | Class | LO | HI* | m* | Feas | Served | Used | Dist | Charge visits | Charge t | Dur | J | Runtime | Term |
|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| c103_21 | C/type1 | 19 | 37 | 37 | True | 100/100 | 37 | 3100.6 | 19 | 2582.2 | 33967.6 | 46716.9 | 281.2s | completed |
| c201_21 | C/type2 | 19 | 37 | 31 | True | 100/100 | 31 | 2971.7 | 7 | 723.7 | 76386.6 | 48341.3 | 812.4s | budget_exhausted |
| r103_21 | R/type1 | 15 | 34 | 32 | True | 100/100 | 32 | 2829.9 | 34 | 540.3 | 6930.4 | 36049.1 | 770.1s | budget_exhausted |
| r201_21 | R/type2 | 15 | 34 | 18 | True | 100/100 | 18 | 2601.1 | 1 | 3.6 | 15009.6 | 23244.3 | 617.5s | completed |
| rc103_21 | RC/type1 | 18 | 36 | 29 | True | 100/100 | 29 | 3353.7 | 33 | 556.2 | 6512.1 | 33443.2 | 811.8s | budget_exhausted |
| rc201_21 | RC/type2 | 18 | 36 | 29 | True | 100/100 | 29 | 2812.0 | 0 | 0.0 | 22420.9 | 35914.9 | 949.4s | budget_exhausted |

## Aggregate

- feas rate: 1.000
- mean served: 1.000
- mean used vehicles: 29.33
- mean distance: 2944.8
- mean paper J: 37285.0
- mean runtime: 707.1s

Holdouts untouched. Transformer search-guide training not started.
