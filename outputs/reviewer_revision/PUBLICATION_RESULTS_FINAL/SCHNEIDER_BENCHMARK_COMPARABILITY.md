# Schneider published-benchmark comparability

## Independent verifier
[ghiermann/evrptw-verifier](https://github.com/ghiermann/evrptw-verifier) verifies Schneider-format
instances and complete route sequences (node string IDs). Total cost is rounded to **3 decimals**
(sum rounded, not each arc). Full-recharge E-VRPTW semantics of Schneider et al. (2014).

## Our converter (verified this round)
- Euclidean distances **unrounded** (float32); travel = d/v; energy = d×r.
- Parameters g, v, Q, C, r and **all** matrix entries cross-checked: 56/56 OK.
- Homogeneous charging (rate 1/g); waits disabled; neutral price placeholders.

## Literature BKS inventory
`src/care-fi/evaluation/schneider_bks.py` transcribes Erdelić & Carić (2022) FR BKS consolidations
(56 FR entries).

## Feasible-route independent verification status
Primary Schneider experiment stores feasibility/metrics per attempt. Full node-string route archives
suitable for the external verifier were **not** frozen for all 264 feasible attempts in the publication
package. Hard instances (72 attempts) are incomplete (unserved=1) and cannot be verifier-validated.

## Why a verified BKS gap table is NOT claimed
1. Arc-distance rounding: our unrounded Euclidean vs common 2-decimal arc rounding before summing.
2. Verifier rounds **total** distance/cost to 3 decimals — comparable only after exporting routes under
   a documented rounding convention.
3. Incomplete hard-instance solutions cannot be exported as verifier-valid tours.
4. FR vs PR BKS sets assume specific charging policies; mixing with our ALNS search configurations
   without independently verified route export would be misleading.

**Status: NOT SATISFIED — MISSING EVIDENCE** for a primary numerical BKS gap table.
This is an explicit scientific limitation, not an oversight of report writing.
