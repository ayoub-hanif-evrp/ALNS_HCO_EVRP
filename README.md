# CARE-FI: Deterministic Hanif EVRPTW Solver

Classical **CARE-FI (Constraint-Aware Regret Elimination for Fleet
Intensification)** solver for the Hanif Electric Vehicle Routing
dataset (1000-customer Homberger instances, variants A original TW and B
relaxed TW). This tree is **not** currently a Transformer MARL training
codebase — neural/BC/RL modules were removed from the active path.

Research status: the paper headline is the two-stage Classical fleet search
plus heterogeneous station refinement under corrected log-normal waiting.
Archive: `outputs/` (`classical/` + `proposed/`).

## What this project does now

- Fast dense construction → coarse fleet shrink → **fleet intensification**
  (route elimination + station-aware repair; CARE ranking is a lazy fallback)
- Exact route replay acceptance under the **EXTENDED** protocol
- Optional **NO_WAIT_REPLAY** of the same saved routes with station waits off
- Both TW variants; **A→B warm starts** so relaxed-B can inherit A's incumbent
- Seed = 0 (no multi-seed construction portfolio)

## Hanif attributes (honest)

| Attribute | Used? |
|-----------|-------|
| Battery Q=600, energy h=1 | yes |
| Heterogeneous charging rates | yes |
| Station `wait_mu`/`wait_sigma` | yes: log-normal parameters; feasibility uses E[W]=exp(μ+½σ²) |
| Stochastic `wait_sigma` | search: no (expected wait only); post-optimization robustness: yes |
| Charging price | expected/risk-aware Φ modes; not in classical fleet-distance |
| `density_score` in search | no (generator feature only; would double-count) |

Frozen 60x2 archive (never overwrite):
`outputs/classical/` and `outputs/proposed/`.

## Comparison methods

Published EVRPTW metaheuristics (ALNS, VNS/TS) are re-run on this benchmark.
No published table exists for these 1000-customer instances, so numbers are
produced here under matched Q, h, and time windows.

## Quick start

```powershell
python scripts/run_hanif_fps.py --all --variant both --budget 180 --workers 4 --out outputs/hanif_fps_run_new
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
outputs/classical/
outputs/proposed/
outputs/figures/
outputs/tables/
```

## Requirements

- Python 3.10+
- `pip install -r requirements.txt` (numpy, etc.)

## Tests

```powershell
pytest -q
```
