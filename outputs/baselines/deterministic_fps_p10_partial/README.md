# Deterministic-FPS P10 PARTIAL baseline (immutable)

Frozen from the first P10 validation run.

- Decision: **PARTIAL**
- Method: Deterministic-FPS via `solve_with_fleet_search`
- 6/6 validation feasible, 100% served, exact replay OK
- Mean used vehicles: 29.33; mean runtime: ~707s
- Weaknesses: `c103_21` stuck at upper bound m*=37; post-fleet stages skipped on 4/6; budget overruns

Do not overwrite. P10.5 improvements live under `outputs/baselines/deterministic_fps/` (new run) or `outputs/baselines/deterministic_fps_p105/`.

Holdouts were not evaluated.
