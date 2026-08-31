# Components removed from the active CARE-FI path

These modules still exist in the repository for historical inspection. They are **not** part of the CARE-FI method used for the paper.

| Component | Location | Why it left the active path |
|---|---|---|
| Transformer / attention routing | `legacy/training/`, `legacy/env/policy_action.py` | Not used. CARE-FI is deterministic search. |
| MARL / RL fine-tuning | `legacy/training/rollout*.py` | Not the paper method. |
| Behaviour cloning / DAgger | `legacy/expert/`, `legacy/training/` | No learned policy. |
| ALNS (adaptive destroy–repair, SA) | `legacy/search/alns.py` | Random neighbourhoods and simulated annealing. |
| VNS / tabu search | `legacy/search/vns_ts.py` | Internal baseline only, not CARE-FI. |
| Multi-seed / random restarts | fleet search seeds | CARE-FI is deterministic; ties use IDs. |
| Route pool + set partitioning | `legacy/search/route_pool.py` | Extra recombination layer. |
| Cold fleet rebuilds | `legacy/search/hanif_fleet_intensify.py` | Fallback reconstruction. |
| CARE / regret ranking portfolios | intensification rankers | Replaced by \((n_r,\delta_r,r)\) and fewest-compatible-route insertion. |
| Generic rescue solvers | `legacy/search/critical_rescue.py` | Large fallback chain. |
| CVaR / sampled waiting | `legacy/evaluation/paper_objective.py` risk modes | Feasibility and \(\Phi\) use \(\mathbb{E}[W]\), \(\mathbb{E}[P]\) only. |

`src` is only `src/hanif/` and `src/schneider/`. Each package has its own copy of search/env/data. Schneider wraps `src/schneider/search/schneider_solve.py`. The Hanif paper path is `src/hanif/solver.py` (CARE-FI).
