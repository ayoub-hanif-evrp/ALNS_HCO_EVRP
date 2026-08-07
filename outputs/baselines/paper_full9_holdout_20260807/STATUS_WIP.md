# Paper full-9 holdout — COMPLETE (seed 0)

Frozen immutable copy: `outputs/baselines/paper_full9_holdout_20260807/`

Working copy: `outputs/baselines/paper_full9_holdout/seed_0/`

## Results (900s, seed 0)

| Method | Served | Feas | Mean J | Mean used |
|---|---:|---:|---:|---:|
| Deterministic-FPS | 0.956 | 8/9 | 84950 | 24.33 |
| Det-FPS+SearchMoveGuide | 0.996 | 8/9 | 44303 | 23.67 |
| FixedFleet-Expert+Repair+ALNS (frozen) | 0.998 | 8/9 | 62371 | 39.78 |

Only `r101_21` remains incompletely served under fleet minimization.

Optional next: `python scripts/run_paper_full9_holdout.py --budget 900 --seed 1 --methods both` for dual-seed robustness.
