# Schneider 12-instance failure root cause

## Facts
- Primary 30 s: 12 instances × 3 seeds × 2 methods = 72 unsuccessful attempts (unserved=1).
- Targeted 120 s: 72/72 still unserved=1 → not a short-timeout artefact.
- Full matrix/g/v fidelity: 56/56 OK.
- Solo depot–customer–depot fails on **battery** for C65/C64/C75 (round-trip energy > Q).
- Station-assisted solo routes are feasible for all 12 hard customers (see diagnosis CSV).

## Missing-customer consistency
Same single customer missing in every FR/PR×seed run for **12/12** instances:

| Customer | Instances |
|---|---|
| C65 | r101_21, r102_21, r103_21, r105_21, r106_21 |
| C64 | r108_21 |
| C75 | rc101_21, rc103_21–rc107_21 |

## Classification
**Search / construction / initialization limitation** under the current solver stack:
the missing customers are individually feasible with charging, but the solver repeatedly fails
to construct a complete covering. Converter incompatibility is not supported by the full-matrix audit.
Individual-route feasibility does **not** prove complete-instance feasibility under every rounding convention.

Algorithm changes to force success are out of scope for this revision.
