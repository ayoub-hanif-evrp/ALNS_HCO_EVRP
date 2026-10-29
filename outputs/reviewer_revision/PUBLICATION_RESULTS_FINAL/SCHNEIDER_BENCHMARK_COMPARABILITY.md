# Schneider published-benchmark comparability

## Reviewer requirement interpretation
Transparent **external validation** of the Schneider EVRPTW set (feasibility, fidelity, hard-instance
diagnosis) was completed for the primary protocol. A **full primary numerical BKS gap table** for all
264 feasible attempts additionally requires archived routes for every primary run.

## Primary protocol (unchanged)
- **336** attempts, **264** feasible, **72** unsuccessful; **44** jointly FR+PR feasible.
- Hard customers C65/C64/C75 diagnosed; station-assisted solo routes exist (solo ≠ instance proof).
- Full g/v/matrix fidelity: 56/56 OK.
- Homogeneous Schneider charging does **not** validate heterogeneous price/wait HCO claims.

## Independent external verifier (ghiermann/evrptw-verifier)
Built and executed locally from https://github.com/ghiermann/evrptw-verifier
(jar `evrptw-verifier-0.2.0.jar`, OpenJDK 17 portable under `tools/jdk17`).

Verifier semantics (from source): unrounded Euclidean arcs; **total** distance rounded to **3** decimals;
station visit ⇒ **full** recharge to battery capacity; customers must all be covered.

Diagnostic ALNS-FR 30 s exports (5 instances: c101_21, c201_21, r101_21, r201_21, rc101_21):
- Raw exported sequences (may include consecutive identical station tokens): external valid **3/5**
- After collapsing consecutive identical node IDs (Schneider-style single full-recharge visit): external valid **3/5**

Detailed logs: `QC_LOGS/evrptw_verifier_*.txt`
Table: `FINAL_TABLES/Schneider_external_verifier_results.csv`

## Per-instance verifier outcomes (collapsed variant)

| Instance | Our K | Sol cost (3dp) | External valid | Verifier cost | BKS K | BKS D | Dist gap % | Note |
|---|---:|---:|:---:|---:|---:|---:|---:|---|
| c101_21 | 12 | 1197.084 | YES | 1197.084 | 12 | 1053.83 | +13.59 | distance_gap_vs_inventory_BKS_after_external_valid |
| c201_21 | 5 | 739.248 | YES | 739.248 | 4 | 645.16 | +14.58 | fleet_diff our_k=5 bks_k=4; distance_gap secondary |
| r101_21 | 25 | 2262.128 | NO | None | 18 | 1663.04 | — | not_claimed_solution_invalid |
| r201_21 | 5 | 1325.483 | YES | 1325.483 | 3 | 1264.32 | +4.84 | fleet_diff our_k=5 bks_k=3; distance_gap secondary |
| rc101_21 | 24 | 2702.059 | NO | None | 16 | 1723.79 | — | not_claimed_solution_invalid |

## Status: **PARTIAL**
External verifier executed on diagnostic FR full-recharge routes; primary 264 feasible attempts still lack complete route archives for a full BKS gap table.

### What is SATISFIED
- Independent Java verifier installed, built, and run against original Schneider `.txt` instances.
- Full-recharge (FR) diagnostic routes tested; PR routes are not judged by this full-recharge verifier.

### What remains missing for a full primary BKS table
1. Primary experiment checkpoints did not freeze string-ID routes for all 264 feasible attempts.
2. Diagnostic 30 s solutions are not claimed as competitive BKS-quality tours.
3. Where externally valid, inventory BKS gaps may be reported for the diagnostic set only,
   with fleet size primary and distance secondary — **not** extrapolated to the 336-run panel.

### Representation note
Some ALNS-FR exported routes contain long runs of consecutive identical station IDs
(internal waiting/charge-loop encoding). The Schneider verifier models **one** station visit as a
full recharge. The `collapse_consecutive_identical` variant is therefore the scientifically
appropriate mapping for external full-recharge verification; raw multi-token station runs are
also tested and logged for transparency.


### External confirmation of missing customers
On predeclared hard-family diagnostics, the Java verifier independently reported:
- `r101_21`: INVALID — customers missing: **C65**
- `rc101_21`: INVALID — customers missing: **C75**
This corroborates the existing hard-customer diagnosis; solo-route station feasibility still does not prove full-instance feasibility.
