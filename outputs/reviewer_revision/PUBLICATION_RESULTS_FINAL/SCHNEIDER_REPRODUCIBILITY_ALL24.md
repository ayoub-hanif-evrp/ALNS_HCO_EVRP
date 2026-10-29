# Schneider FR168 — all 24 reproducibility discrepancies

Source: comparison of FR168 route-archive metrics to primary Schneider FR checkpoints.
Primary 336 archive unchanged. Algorithms unchanged.

n = **24**. Feasibility matched primary in **24/24**.

| Instance | Seed | ΔK | Δdistance | Iter ratio | Impact | Cause class |
|---|---:|---:|---:|---:|---|---|
| c101_21 | 2 | 0 | +37.1811 | 0.597 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c103_21 | 1 | 0 | +4.7466 | 0.819 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c103_21 | 2 | 0 | +20.4663 | 0.584 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c104_21 | 2 | 0 | +69.0778 | 0.566 | DISTANCE_MATERIAL | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c105_21 | 2 | 0 | +22.8804 | 0.627 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c107_21 | 1 | 0 | +16.7022 | 0.668 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c107_21 | 2 | 0 | +1.5377 | 0.625 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c108_21 | 1 | 0 | +0.1390 | 0.671 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c108_21 | 2 | 0 | +58.1285 | 0.625 | DISTANCE_MATERIAL | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c109_21 | 1 | 1 | -23.3948 | 0.639 | FLEET_CHANGED | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| c207_21 | 2 | 1 | -15.7909 | 0.604 | FLEET_CHANGED | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r104_21 | 0 | 1 | +22.7930 | 0.819 | FLEET_CHANGED | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r107_21 | 0 | 0 | +5.7523 | 0.839 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r107_21 | 1 | 0 | +49.7283 | 0.727 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r110_21 | 0 | 0 | +133.2042 | 0.709 | DISTANCE_MATERIAL | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r111_21 | 1 | 0 | +11.3430 | 0.773 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r112_21 | 0 | 0 | +5.1454 | 0.777 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r204_21 | 1 | 0 | +6.3801 | 0.685 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r207_21 | 0 | 0 | +7.3996 | 0.742 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| r208_21 | 1 | 0 | +2.5196 | 0.705 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| rc102_21 | 1 | 1 | +82.3913 | 0.656 | FLEET_CHANGED | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| rc102_21 | 2 | 1 | +57.5368 | 0.751 | FLEET_CHANGED | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| rc202_21 | 1 | 0 | +13.3802 | 0.716 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |
| rc204_21 | 1 | 0 | +3.0816 | 0.729 | DISTANCE_MINOR | SUPPORTED_LOWER_ITERATION_THROUGHPUT |

## Interpretation
- Do **not** claim every discrepancy is solely CPU contention.
- Lower iteration ratio supports wall-clock throughput sensitivity as a likely contributor.
- Similar-iteration discrepancies remain **uncertain** (timing path dependence).
- Scientific fleet/feasibility conclusions vs primary change only where impact is `FLEET_CHANGED` (feasibility never changed in this set).


## Threshold note
The 5% distance split (`DISTANCE_MATERIAL` vs `DISTANCE_MINOR`) is a **descriptive categorization** for reporting, not an independently established statistical significance threshold.
Lower iteration throughput is an **evidence-supported contributing explanation**, not proof that CPU contention was the sole cause.
