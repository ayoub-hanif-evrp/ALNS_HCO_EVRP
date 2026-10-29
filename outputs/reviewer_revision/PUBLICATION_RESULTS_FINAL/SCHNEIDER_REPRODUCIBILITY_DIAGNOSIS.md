# Schneider seed-0 reproducibility diagnosis (5 discrepancies)

Instances: r104_21, r107_21, r110_21, r112_21, r207_21

## Findings
- GI construction is deterministic across repeats: **True**
- ALNS-FR uses `random.Random(seed)` and a **wall-clock** `time.perf_counter()` budget (not iteration count).
- On all five mismatches, Scope B (4-worker parallel) recorded **fewer ALNS iterations** than primary.
- Serial diagnostic exact match to primary: **2/5**

## Cause assessment
Wall-clock ALNS budget: under parallel CPU contention Scope B completed fewer iterations than the primary archive on the five mismatched instances, producing different (usually worse) incumbents despite identical seed and GI.

Serial re-runs (one job at a time) raised iteration counts toward the primary range and **restored exact primary K/distance for 2/5** (`r104_21`, `r207_21`). The other three still differed: even serial runs did not reach the historical primary iteration counts, so the RNG trajectory diverged after a different number of accepted moves. This is **evidence of wall-clock / throughput sensitivity**, not of a seed-propagation bug or altered algorithm.

| Instance | Primary iters | ScopeB parallel iters | Serial iters | Serial ≡ primary |
|---|---:|---:|---:|:---:|
| r104_21 | 12947 | 10608 | 11450 | yes |
| r107_21 | 11660 | 9787 | 9970 | no (≡ ScopeB) |
| r110_21 | 11348 | 8046 | 8976 | no |
| r112_21 | 10140 | 7880 | 8857 | no (≡ ScopeB) |
| r207_21 | 11129 | 8258 | 10115 | yes |

## Comparability
Primary and Scope-B/fr168 runs share the same algorithm, seed, GI deadline, and nominal 30 s budget, but wall-clock search makes throughput machine/load-dependent. Metrics are scientifically comparable as equal-wall-clock outcomes, not as bit-identical deterministic trajectories. Both versions are preserved.

Algorithm was **not** changed to force matching. Primary archive and new archives are both retained.
Table: `FINAL_TABLES/Schneider_repro_discrepancy_diagnosis.csv`
