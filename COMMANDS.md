# Commands — Hanif Fleet Intensification

Frozen FastFeasibleBaseline: `outputs/hanif_fps/fast_feasible_baseline/` (do not overwrite).

## Intensified A + B (A→B warm starts)

```powershell
python scripts/run_hanif_fps.py --all --variant both --budget 180 --out outputs/hanif_fps_intensify
```

## Pilot

```powershell
python scripts/run_hanif_fps.py --pilot --variant both --budget 180 --out outputs/hanif_fps_intensify_pilot
```

Milestone engineering targets (not paper claims):
- A mean K ≤ 80, 60/60 feasible
- B mean K ≤ 70, 60/60 feasible
- no B instance worse than its A incumbent
