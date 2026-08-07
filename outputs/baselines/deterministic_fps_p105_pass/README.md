# Deterministic-FPS P10.5 PASS baseline (immutable)

Frozen dual-seed validation after structural repairs.

- Decision: **PASS**
- Method: Deterministic-FPS (`solve_deterministic_fps` + genuine fleet minimization)
- Budget: 900s / instance
- Validation set (6): `c103_21, c201_21, r103_21, r201_21, rc103_21, rc201_21`
- Holdouts (9) were **not** evaluated
- Tests: 55 passed (`pytest -q`)

## Seed results

| Seed | Feas rate | Mean used | Mean runtime | Within 5% | Stuck at upper |
|---:|---:|---:|---:|:---:|:---:|
| 0 | 6/6 | 25.00 | 654.1s | Yes | No |
| 1 | 6/6 | 25.33 | 654.7s | Yes | No |

Both seeds: 100% served, exact replay OK, fleet minimization performed (no fixed m), mandatory stages invoked.

## vs P10 PARTIAL

| | P10 PARTIAL | P10.5 PASS |
|---|---|---|
| Feasibility | 6/6 | 6/6 (dual seed) |
| `c103_21` m* | 37 (stuck at upper) | 26–30 with descent |
| `r103_21` | incomplete / budget fail | m*=28, within budget |
| Post-fleet stages | often skipped | charging / repair / ALNS / pool / rescue invoked |
| Budget adherence | overruns | within 5% on all 12 runs |

Do not overwrite this folder. Working copies may continue under `outputs/baselines/deterministic_fps_p105/`.

P11 (Transformer search-guide) is now allowed.
