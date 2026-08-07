# Final method comparison (full-9 holdout)

Selected after P10.5 + P11 validation gates: **Deterministic-FPS** (fleet-minimized) and optional **Det-FPS + Transformer search-move guide**.

Budget: 900.0s/instance | Seed: 0 | Holdouts: c101_21, c104_21, c208_21, r101_21, r104_21, r211_21, rc101_21, rc104_21, rc208_21

## Frozen baselines (immutable)

| Method | Served | Feas | Mean J | Mean used | Source |
|---|---:|---:|---:|---:|---|
| Expert | 0.984 | 0.667 | 96950.5 | — | full9_20260731 |
| Static BC | 0.920 | 0.667 | 161077.6 | — | full9_20260731 |
| Old RL-best | 0.903 | 0.667 | 177774.5 | — | full9_20260731 |
| FixedFleet-Expert+Repair+ALNS (m=40) | 0.998 | 0.889 | 62371.0 | 39.78 | expert_repair_alns_20260801 / fixed_fleet_expert_repair_alns |

## Live fleet-minimized solvers (this freeze)

| Method | Served | Feas | Mean J | Mean used | Mean runtime | Source |
|---|---:|---:|---:|---:|---:|---|
| Deterministic-FPS | 0.956 | 0.889 | 84949.6 | 24.33 | 652.4s | live_full9_holdout |
| Det-FPS+SearchMoveGuide | 0.996 | 0.889 | 44303.3 | 23.67 | 671.8s | live_full9_holdout |

### Per-instance — Deterministic-FPS

| Instance | Class | Feas | Served | m* | Used | Dist | J | Runtime | Within5% | Term |
|---|---|:---:|---:|---:|---:|---:|---:|---:|:---:|---|
| c101_21 | C/type1 | True | 100/100 | 23 | 23 | 2342.7 | 29725.2 | 638.9s | True | completed |
| c104_21 | C/type1 | True | 100/100 | 31 | 31 | 2756.3 | 38131.6 | 650.6s | True | completed |
| c208_21 | C/type2 | True | 100/100 | 19 | 19 | 2428.4 | 29686.8 | 650.1s | True | completed |
| r101_21 | R/type1 | False | 60/100 | None | 31 | 1710.5 | 533678.5 | 690.7s | True | completed |
| r104_21 | R/type1 | True | 100/100 | 21 | 21 | 2008.5 | 23657.5 | 640.3s | True | completed |
| r211_21 | R/type2 | True | 100/100 | 16 | 16 | 1768.6 | 19321.5 | 636.5s | True | completed |
| rc101_21 | RC/type1 | True | 100/100 | 31 | 31 | 3058.5 | 34962.1 | 656.3s | True | completed |
| rc104_21 | RC/type1 | True | 100/100 | 23 | 23 | 2727.8 | 26467.5 | 642.3s | True | completed |
| rc208_21 | RC/type2 | True | 100/100 | 24 | 24 | 3077.3 | 28916.0 | 666.3s | True | completed |

### Per-instance — Det-FPS+SearchMoveGuide

| Instance | Class | Feas | Served | m* | Used | Dist | J | Runtime | Within5% | Term |
|---|---|:---:|---:|---:|---:|---:|---:|---:|:---:|---|
| c101_21 | C/type1 | True | 100/100 | 22 | 22 | 2173.0 | 28308.7 | 664.0s | True | completed |
| c104_21 | C/type1 | True | 100/100 | 24 | 24 | 2240.2 | 29944.2 | 657.8s | True | completed |
| c208_21 | C/type2 | True | 100/100 | 25 | 25 | 2733.1 | 36598.2 | 673.1s | True | completed |
| r101_21 | R/type1 | False | 96/100 | None | 29 | 2342.4 | 172306.4 | 719.4s | True | completed |
| r104_21 | R/type1 | True | 100/100 | 21 | 21 | 2008.5 | 23657.5 | 657.7s | True | completed |
| r211_21 | R/type2 | True | 100/100 | 15 | 15 | 2041.8 | 18468.3 | 664.1s | True | completed |
| rc101_21 | RC/type1 | True | 100/100 | 31 | 31 | 3058.5 | 34962.1 | 667.1s | True | completed |
| rc104_21 | RC/type1 | True | 100/100 | 23 | 23 | 2706.5 | 26428.1 | 660.0s | True | completed |
| rc208_21 | RC/type2 | True | 100/100 | 23 | 23 | 3280.5 | 28056.2 | 682.5s | True | completed |

## Honesty notes

- FixedFleet-Expert+Repair+ALNS uses **fixed m=40** and is an ablation, not minimized fleet.
- Deterministic-FPS / Det-FPS+Guide perform genuine descending fleet minimization.
- Validation carve-out (6 instances) was used for method selection; this table is holdout-only.
- P11 guide checkpoint: `outputs/tgfps_p11/transformer_search_guide.pt` (not BC).
