# Deterministic Hanif EVRPTW Solver

Classical **Deterministic-FPS** solver for the Hanif Electric Vehicle Routing
dataset (1000-customer Homberger instances, variants A original TW and B
relaxed TW). This tree is **not** currently a Transformer MARL training
codebase — neural/BC/RL modules were removed from the active path.

## What this project does now

- Fast dense construction → coarse fleet shrink → **Hanif fleet intensification**
  (scalable route elimination / ejection + light regret repair)
- Exact route replay acceptance
- Both TW variants; **A→B warm starts** so relaxed-B can inherit A's incumbent
- Seed = 0 (no multi-seed construction portfolio)

## Hanif attributes (honest)

| Attribute | Used? |
|-----------|-------|
| Battery Q=600, energy h=1 | yes |
| Heterogeneous charging rates | yes |
| Station `wait_mu` as deterministic expected wait | yes |
| Stochastic `wait_sigma` | no |
| Charging price in objective | no (`lambda_charging_price=0`) |
| `density_score` in search | no |

Frozen scalability-only baseline (not fleet-minimized):
`outputs/hanif_fps/fast_feasible_baseline/`.

## Quick start

```powershell
# Intensified run on both TW variants (writes routes + metrics)
python scripts/run_hanif_fps.py --all --variant both --budget 180 --out outputs/hanif_fps_intensify

# Pilot (6 instances × 2 variants)
python scripts/run_hanif_fps.py --pilot --variant both --budget 180 --out outputs/hanif_fps_intensify_pilot
```

See `COMMANDS.md`.

## Layout

```
configs/hanif_fps.yaml
scripts/run_hanif_fps.py
src/search/deterministic_fps.py
src/search/hanif_fleet_intensify.py
src/search/fast_construct.py
dataset/Hanif/
outputs/hanif_fps/fast_feasible_baseline/
```

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (numpy, etc.)

## Tests

```powershell
pytest -q
```
