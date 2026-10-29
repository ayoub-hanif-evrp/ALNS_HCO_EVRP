# Schneider published-benchmark comparability

## Independent verifier
[ghiermann/evrptw-verifier](https://github.com/ghiermann/evrptw-verifier) verifies Schneider-format
instances and complete route sequences (node string IDs). Total cost is rounded to **3 decimals**
(sum rounded, not each arc). Full-recharge E-VRPTW semantics of Schneider et al. (2014).

## Our converter
- Euclidean distances **unrounded** (float32).
- Travel = distance / v; energy = distance × r.
- Homogeneous charging; waits disabled; neutral price placeholders (not used for hetero claims).

## Literature BKS table in repo
`src/care-fi/evaluation/schneider_bks.py` transcribes Erdelić & Carić (2022) FR/PR BKS consolidations
(56 FR entries loaded; error=None).

## Why numerical BKS gaps are not reported as primary claims
1. Arc-distance rounding conventions differ (unrounded vs typical 2-decimal arcs).
2. Primary hard-instance solutions are incomplete (unserved customers) — cannot export verifier-valid tours.
3. Feasible ALNS runs lack archived full node-string route files in the publication freeze for verifier export.
4. FR vs PR BKS sets assume specific charging policies; mixing them with our search configurations without
   verified route export would be misleading.

**Status:** BKS inventory is available for reference, but a verified common-formulation gap table is
**NOT** produced. If route archives with string IDs are later exported for feasible instances, the
external verifier can be applied under documented rounding caveats.
