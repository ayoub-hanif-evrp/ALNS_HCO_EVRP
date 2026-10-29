# Schneider external validation diagnostic

## Source fidelity
Validated all **56** instances against `.txt` parameters: **56 OK**.
Distance convention: euclidean_unrounded; travel=distance/v; energy=distance*r.
Heterogeneous Hanif prices/waits **not** imported.

## Primary 30 s protocol (unchanged)
336 attempts; 264 feasible; 72 unsuccessful; 44 both-feasible for FR/PR paired stats.
Failures classified as construction/search limits under current solver (Round-4 `replay_route` probes), **not** proven mathematical infeasibility.

## Targeted 120 s diagnostic (12 hard instances)
Planned/completed: 72/72.
Feasible runs at 120 s: 0.
Instances with any feasible 120 s run: 0.
**Primary 30 s table is not replaced** by longer-budget successes.

## BKS comparison
**Not performed as a numerical claim.** Reason: Published Schneider BKS typically use rounded distances and/or different objective aggregations; our converter uses unrounded Euclidean. Numerical BKS gaps would be misleading without verified convention match.

## Addendum — 120 s diagnostic outcome
All 72 longer-budget attempts on the 12 hard instances remained infeasible with exactly one unserved customer. Additional wall-clock beyond 30 s is unlikely to be informative for these cases under the current implementation; further multi-hour runs were not launched.
