# P10 decision: **PARTIAL**

The solver works, but specific fleet/feasibility/runtime problems remain.

## Why not PASS
- c103_21 finished at m*=37 (= expanded upper bound) after failing only m=36; this is high-fleet feasible, not demonstrated fleet minimization.
- On 4/6 instances, post-fleet stages (charging DP, repair, ALNS, route-pool, rescue) were skipped due to budget exhaustion.
- 4/6 instances exceeded the equal 720s budget (up to ~949s); runtimes are reported honestly.

## What works
- No fixed fleet in official results; solve_with_fleet_search used throughout.
- Fleet descent ran on all instances; strong on 
201_21 (34→18) and several others.
- 6/6 complete feasibility, 100% served, exact replay OK on all accepted solutions.
- 43 pytest passed from clean suite.
- Holdouts untouched; Transformer search-guide (P11) not started.

`json
{
  "decision": "PARTIAL",
  "text": "The solver works, but specific fleet/feasibility/runtime problems remain.",
  "checks": {
    "fleet_minimization": true,
    "no_fixed_fleet": true,
    "descent_ran": true,
    "genuine_descent_weak_on": [
      "c103_21"
    ],
    "exact_ok_on_feasible": true,
    "all_customers_served": true,
    "route_pool_or_rescue_invoked": true,
    "route_pool_rescue_skipped_due_to_budget": [
      "c201_21",
      "r103_21",
      "rc103_21",
      "rc201_21"
    ],
    "pytest_ok": true,
    "feasibility_rate": 1.0,
    "budget_overruns": [
      "c201_21",
      "r103_21",
      "rc103_21",
      "rc201_21"
    ]
  },
  "missing_or_skipped": [
    "c103_21: m* equals expanded upper bound (37); descent failed at 36 \u2014 not successful fleet minimization for this instance (earlier probes showed ~30 feasible under tighter hi).",
    "c201_21,r103_21,rc103_21,rc201_21: charging/repair/ALNS/route-pool/rescue skipped after fleet budget exhaustion.",
    "Four instances exceeded the equal 720s wall budget (honestly reported)."
  ],
  "note": "Auto-gate had PASS; overridden to PARTIAL per acceptance rule: do not treat high-fleet feasible as successful fleet minimization. P11 not started."
}
`
