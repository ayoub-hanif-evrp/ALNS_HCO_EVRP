# Paper full-9 holdout freeze (immutable) — 2026-08-07

Equal-budget fleet-minimized evaluation on the reserved Schneider 9 holdouts.

- Budget: **900s / instance**
- Seed: **0**
- Methods: Deterministic-FPS, Det-FPS + Transformer search-move guide
- Do **not** overwrite

## Headline aggregates

| Method | Served | Feas | Mean J | Mean used | Notes |
|---|---:|---:|---:|---:|---|
| Expert (frozen) | 0.984 | 0.667 | 96950.5 | — | greedy |
| Static BC (frozen) | 0.920 | 0.667 | 161077.6 | — | greedy |
| Old RL-best (frozen) | 0.903 | 0.667 | 177774.5 | — | greedy |
| FixedFleet-Expert+Repair+ALNS | 0.998 | 0.889 | 62371.0 | 39.78 | **fixed m=40** |
| **Deterministic-FPS** | 0.956 | 0.889 | 84949.6 | **24.33** | fleet-min |
| **Det-FPS+SearchMoveGuide** | **0.996** | 0.889 | **44303.3** | **23.67** | fleet-min + P11 |

## Residual failure
Both fleet-min methods miss full feasibility on **`r101_21`** only (Det-FPS 60% served; guided 96% served). Same instance was the FixedFleet miss (feas 8/9).

## Selected paper solver
**Det-FPS + Transformer search-move guide** — best mean J and served among fleet-minimized methods, same feas rate as FixedFleet, ~40% fewer vehicles than fixed m=40.

See `method_comparison.md` for per-instance rows.
