# Hanif ALNS-HCO results (factual)

Interpretation locked from the master CSV. Do not claim that HCO increases fleet size,
broadly reorganizes charging plans, or consistently selects cheaper/faster stations.

- Master rows: 3720; feasible rows: 3720.
- ALNS-PR vs separately run ALNS-PR+HCO: $K$ differs on 16 of 600 instance–seed pairs. That comparison is not a perfectly controlled fixed-route ablation.
## Original TW (n=60 paired instances)
- ALNS-FR → ALNS-PR charged-energy change (ratio of means): -25.20%.
- ALNS-FR → ALNS-PR charging-expense change (ratio of means): -24.84%.
- ALNS-PR mean K: 69.88
- ALNS-HCO mean K: 71.01 (Phase I is about 15 s then K is frozen; this is not an HCO fleet-size effect).
- ALNS-PR mean distance: 48555.3
- ALNS-HCO mean distance: 50082.7
- Charging-cost mean paired %: 2.43 (95% CI [-1.50, 6.36])
- Charging-cost wins/ties/losses: 20/0/40
- ALNS-HCO within-run Phase II changed distance or charging expense in 20 of 300 seed-runs (K changed: 0; C1:16, C2:4).
- ALNS-PR+HCO within-run HCO changed distance or charging expense in 12 of 300 seed-runs (K changed: 0; C1:11, C2:1).
- ALNS-PR mean runtime (s): 30.90
- ALNS-HCO mean runtime (s): 31.03

## Relaxed TW (n=60 paired instances)
- ALNS-FR → ALNS-PR charged-energy change (ratio of means): -16.74%.
- ALNS-FR → ALNS-PR charging-expense change (ratio of means): -16.57%.
- ALNS-PR mean K: 61.80
- ALNS-HCO mean K: 62.81 (Phase I is about 15 s then K is frozen; this is not an HCO fleet-size effect).
- ALNS-PR mean distance: 49276.6
- ALNS-HCO mean distance: 50998.0
- Charging-cost mean paired %: 7.16 (95% CI [5.32, 9.35])
- Charging-cost wins/ties/losses: 4/0/56
- ALNS-HCO within-run Phase II changed distance or charging expense in 0 of 300 seed-runs (K changed: 0; no families).
- ALNS-PR+HCO within-run HCO changed distance or charging expense in 0 of 300 seed-runs (K changed: 0; no families).
- ALNS-PR mean runtime (s): 31.19
- ALNS-HCO mean runtime (s): 31.32

## Reporting rules
- Large energy / charging-time / charging-cost gains are ALNS-FR → ALNS-PR.
- ALNS-PR → ALNS-HCO mainly reduces charging visits and station waiting, often with more energy per visit.
- Do not write that HCO selects significantly cheaper or faster chargers: the master CSV has no selected-station $\bar P_s$, $\bar W_s$, or $r_s$ logs.
- Do not use customer-waiting percentage change as a headline result.
