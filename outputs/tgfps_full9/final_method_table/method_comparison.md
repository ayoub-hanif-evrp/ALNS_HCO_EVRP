# Final method comparison (full-9 holdout)

Selected after validation/smoke gate: **Expert+Repair+ALNS**

## Frozen baseline (immutable `outputs/baselines/full9_20260731`)

| Method | Served | Feas | Mean J | Source |
|---|---:|---:|---:|---|
| Expert | 0.984 | 0.667 | 96950.5 | frozen |
| Static BC | 0.920 | 0.667 | 161077.6 | frozen |
| Old RL-best | 0.903 | 0.667 | 177774.5 | frozen |

## Live TG-FPS selected solver

| Method | Served | Feas | Mean J | Source |
|---|---:|---:|---:|---|
| Expert+Repair+ALNS | 0.998 | 0.889 | 62371.0 | live_full9 |

### Per-instance (live)

| Instance | Feas | Served | J | Used EVs | Runtime |
|---|:---:|---:|---:|---:|---:|
| c101_21 | True | 1.000 | 53978.9 | 40 | 100.4s |
| c104_21 | True | 1.000 | 49700.6 | 38 | 94.9s |
| c208_21 | True | 1.000 | 61163.3 | 40 | 83.2s |
| r101_21 | False | 0.980 | 164770.7 | 40 | 519.8s |
| r104_21 | True | 1.000 | 44462.7 | 40 | 104.5s |
| r211_21 | True | 1.000 | 47644.1 | 40 | 50.3s |
| rc101_21 | True | 1.000 | 45660.7 | 40 | 132.9s |
| rc104_21 | True | 1.000 | 45952.5 | 40 | 144.9s |
| rc208_21 | True | 1.000 | 48006.0 | 40 | 54.6s |

Directional goals: feas>6/9, served≥0.984, J<96951 — **met** (feas 0.889, served 0.998, J≈62371).

Residual-RL: disabled (`residual_rl.enabled: false`).

Methods in TG-FPS stack: Expert, Expert+Repair, Expert+Repair+ALNS, Static BC / Old RL-best (frozen), Candidate Ranker / DAgger (trained), Neural-Guided variants behind hybrid_search.
