# ALNS-HCO results

Adaptive Large Neighborhood Search with Heterogeneous Charging Oracle.
Phase I is the same GI + routing ALNS as the 15 s ALNS baseline, then just-enough
fleet compression. Phase II keeps searching with lex (K, distance, charging);
the fleet may decrease and may not increase.

- Instances: 120
- Feasible: 120/120

## Original TW

- Mean K (own ALNS → HCO): 71.27 → 68.68
- Instances with K < own ALNS: 38/60
- Mean K after Phase I compress: 69.88
- Mean distance: 48301.4 → 47270.0
- Mean charging cost: 7975.6 → 7838.6
- Mean charging time: 12713.6
- Mean runtime (s): 32.53

## Relaxed TW

- Mean K (own ALNS → HCO): 62.85 → 59.98
- Instances with K < own ALNS: 46/60
- Mean K after Phase I compress: 61.07
- Mean distance: 47733.3 → 46500.6
- Mean charging cost: 9686.6 → 9183.3
- Mean charging time: 15072.6
- Mean runtime (s): 33.27

Files: `results.csv`, `summary.json`. Hyperparameters are in `summary.json`.

This is our implementation. It is not a reproduction of Keskin, Wang, or Kullman.
