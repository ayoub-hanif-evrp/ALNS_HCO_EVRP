# Schneider published-benchmark comparability

## Reviewer requirement interpretation
Transparent **external validation** of the Schneider EVRPTW set (feasibility, fidelity, hard-instance
diagnosis) was completed. A **verified numerical BKS gap table** additionally requires matching
distance rounding, charging policy, objective, and independently checkable route archives.
That last requirement remains **unresolved**.

## What exists
- Primary protocol unchanged: **336** attempts, **264** feasible, **72** unsuccessful.
- Hard customers C65/C64/C75 diagnosed; station-assisted solo routes exist.
- Full g/v/matrix fidelity: 56/56 OK.
- BKS inventory in code: 56 FR entries (`schneider_bks.py`) — reference only.
- Independent verifier project: https://github.com/ghiermann/evrptw-verifier (3-decimal total rounding;
  Schneider full-recharge semantics).

## Targeted route-export diagnostic (NEW, does not alter primary 336)
Saved complete routes for 3 ALNS-FR 30 s runs on easy instances:
c101_21, c201_21, r201_21.
- Independent internal replay match: **3/3**.
- Artifacts: `outputs/reviewer_revision/schneider_route_export_diag/` and
  `FINAL_TABLES/Schneider_route_export_diag.csv`.

## Why numerical BKS gaps are still NOT claimed
1. Our distances are **unrounded Euclidean** (float32); many published tables use **2-decimal arc**
   rounding before summing; the verifier rounds **total** cost to 3 decimals.
2. Primary experiment checkpoints did not freeze string-ID routes for all 264 feasible attempts.
3. Inventory `compare_row` distances are therefore **not** treated as publication BKS gaps.
4. Homogeneous Schneider charging does **not** validate heterogeneous price/wait HCO claims.

**Status: NOT SATISFIED — MISSING EVIDENCE** for a primary numerical BKS gap table.
External feasibility validation and hard-instance diagnosis: **SATISFIED**.
