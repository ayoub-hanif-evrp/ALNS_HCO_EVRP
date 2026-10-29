# Schneider published-benchmark comparability

## Status: **NOT SATISFIED — MISSING EVIDENCE** for a primary numerical BKS gap table

## Independent verifier
[ghiermann/evrptw-verifier](https://github.com/ghiermann/evrptw-verifier) verifies Schneider-format
instances and complete route sequences. Total cost rounded to 3 decimals. Full-recharge E-VRPTW semantics.

## What we verified
- Source + full matrix fidelity (g, v, Q, C, r, all distance/time/energy entries): 56/56 OK.
- Hard-customer diagnosis: C65/C64/C75 admit station-assisted solo routes; solo D–C–D fails on battery.
- Identifying missing customers does **not** prove official complete-instance infeasibility.

## Why BKS gaps are not claimed
1. Primary Schneider checkpoints store metrics without complete node-string route archives (sample has_routes=False). Distance convention remains unrounded Euclidean vs common 2-decimal arc rounding; verifier rounds total cost to 3 decimals. Without exported verifier-valid tours under a documented shared rounding convention, numerical BKS gaps are not claimed.
2. FR vs PR BKS sets assume specific charging policies incompatible with mixing unverified search configs.
3. Hard instances (72 attempts) are incomplete and cannot be verifier-validated.

BKS inventory remains in `src/care-fi/evaluation/schneider_bks.py` for reference only.
