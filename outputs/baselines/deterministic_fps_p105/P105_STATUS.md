# P10.5 status — **PASS**

## Frozen immutable baseline
`outputs/baselines/deterministic_fps_p105_pass/` (seed_0 + seed_1)

P10 PARTIAL preserved at `outputs/baselines/deterministic_fps_p10_partial/`.

## Acceptance gates (both seeds)
- pytest: **55 passed**, rc=0
- 6/6 instances feasible, 100% served, exact replay OK
- Genuine fleet minimization (no fixed fleet)
- Not stuck at upper after one fail
- Mandatory stages invoked (charging, repair, ALNS, route pool, critical rescue check)
- Runtime within 5% of 900s budget on every instance

## Seed 0 (budget 900s)

| Instance | m* | Feas | Served | Runtime | Within5% |
|---|---:|:---:|---:|---:|:---:|
| c103_21 | 30 | True | 100/100 | 657.4s | True |
| c201_21 | 30 | True | 100/100 | 666.8s | True |
| r103_21 | 28 | True | 100/100 | 646.9s | True |
| r201_21 | 16 | True | 100/100 | 647.2s | True |
| rc103_21 | 26 | True | 100/100 | 646.5s | True |
| rc201_21 | 20 | True | 100/100 | 659.8s | True |

## Seed 1 (budget 900s)

| Instance | m* | Feas | Served | Runtime | Within5% |
|---|---:|:---:|---:|---:|:---:|
| c103_21 | 26 | True | 100/100 | 641.0s | True |
| c201_21 | 30 | True | 100/100 | 674.6s | True |
| r103_21 | 28 | True | 100/100 | 641.8s | True |
| r201_21 | 16 | True | 100/100 | 640.7s | True |
| rc103_21 | 26 | True | 100/100 | 644.9s | True |
| rc201_21 | 26 | True | 100/100 | 685.4s | True |

## Key fixes that unlocked PASS
1. Upper-bound probes get a large first time slice (~380s) so expert episodes on type-1 R can finish (r103).
2. Canonical `(edf, seed=0)` always leads upper probes so dual-seed offsets cannot erase the reliable constructor (c201 seed-1).
3. Cooperative deadlines + lighter insertion/repair keep wall-clock within 5%.

## Decision
**PASS** — Deterministic-FPS is a valid no-regression backbone.

P11 may start. BC / DAgger / RL remain blocked until explicitly scheduled after P11 gates.
