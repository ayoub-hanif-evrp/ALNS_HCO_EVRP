# MARL EVRP Training Dashboard — Figure Index

Training metrics only. No validation or held-out test instances.
Moving average window: **13** (= training instances per cycle).

| Graph | File | Required | Metrics | Status | Explanation |
|-------|------|----------|---------|--------|-------------|
| Final objective over epochs | `01_objective.png` | yes | final_objective | generated | Real EVRP objective J per epoch (lower is better). MA window = one full training cycle. |
| Raw objective return vs shaped training return | `02_raw_vs_shaped_return.png` | yes | raw_objective_return, shaped_training_return | generated | Compares the true objective signal (-J/scale) with the shaped return used for policy updates. |
| Cycle-level objective | `03_cycle_objective.png` | yes | final_objective | generated | Mean objective over all 13 training instances per cycle (most interpretable learning curve). |
| Served-customer ratio | `04_served_ratio.png` | yes | served_rate | generated | Fraction of customers served (target 1.0). |
| Unserved customers | `05_unserved_customers.png` | yes | unserved_customers | generated | Remaining unserved customers per episode (lower is better). |
| Feasibility rate | `06_feasibility_rate.png` | yes | feasibility_rate | generated | Feasible solution indicator smoothed over epochs. |
| Reward decomposition | `07_reward_decomposition.png` | yes | served_bonus_total, early_depot_penalty_total, terminal_unserved_penalty_scaled, mean_pen_unserved, mean_pen_infeasible | generated | Training reward components — check whether terminal penalties still dominate. |
| Terminal penalty schedule | `08_terminal_penalty_schedule.png` | no | terminal_penalty_coef | generated | Ramp of terminal unserved penalty during training. |
| Objective mismatch | `09_objective_mismatch.png` | yes | relative_objective_mismatch | generated | Relative |J - reconstructed J| / |J|. Values > 0.05 mean reward misalignment. |
| Action type fractions | `10_action_type_fractions.png` | yes | fraction_customer_actions, fraction_station_actions, fraction_depot_actions | generated | Policy action mix — rising customer fraction and falling early depot use indicate progress. |
| Early depot returns | `11_early_depot_returns.png` | yes | early_depot_returns | generated | Depot returns while customers remain unserved (should decrease). |
| Actor and critic loss | `12_actor_critic_loss.png` | yes | actor_loss, critic_loss | generated | Actor-critic optimization stability. |
| Critic explained variance | `13_explained_variance.png` | yes | critic_explained_variance | generated | Near zero means the critic is not learning useful values. |
| Advantage statistics | `14_advantage_stats.png` | no | advantage, advantage_std | generated | Scale and spread of GAE advantages. |
| Gradient and parameter update norms | `15_gradient_param_norms.png` | yes | actor_grad_norm, critic_grad_norm, actor_param_delta_norm, critic_param_delta_norm, grad_norm | generated | Confirms the actor and critic weights are actually changing each update. |
| Per-instance objective heatmap | `16_instance_objective_heatmap.png` | yes | final_objective | generated | Objective per RC _21 instance across cycles (13 rows). |
| Per-instance served-ratio heatmap | `17_instance_served_ratio_heatmap.png` | yes | served_rate | generated | Served ratio per instance across cycles — detects repeating bad patterns. |
| Used EVs | `18_used_evs.png` | no | avg_used_evs | generated | Fleet size used per episode. |
| Distance and charging behavior | `19_distance_charging.png` | no | avg_distance, avg_charging_time, charging_station_visits | generated | Minimal operational routing metrics. |
| 20 solution map random final cycle | `20_solution_map_random_final_cycle.png` | yes | random_final_cycle | generated | Representative episode: random_final_cycle |
| 21 solution map best final cycle | `21_solution_map_best_final_cycle.png` | yes | best_final_cycle | generated | Representative episode: best_final_cycle |
| 22 solution map worst final cycle | `22_solution_map_worst_final_cycle.png` | yes | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 23 solution map best overall | `23_solution_map_best_overall.png` | no | best_overall | generated | Representative episode: best_overall |
| 24 route timeline random final cycle | `24_route_timeline_random_final_cycle.png` | yes | random_final_cycle | generated | Representative episode: random_final_cycle |
| 25 route timeline best final cycle | `25_route_timeline_best_final_cycle.png` | yes | best_final_cycle | generated | Representative episode: best_final_cycle |
| 26 route timeline worst final cycle | `26_route_timeline_worst_final_cycle.png` | yes | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 27 battery trajectory random final cycle | `27_battery_trajectory_random_final_cycle.png` | yes | random_final_cycle | generated | Representative episode: random_final_cycle |
| 28 battery trajectory best final cycle | `28_battery_trajectory_best_final_cycle.png` | yes | best_final_cycle | generated | Representative episode: best_final_cycle |
| 29 battery trajectory worst final cycle | `29_battery_trajectory_worst_final_cycle.png` | yes | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 30 unserved over decisions random | `30_unserved_over_decisions_random.png` | yes | random_final_cycle | generated | Representative episode: random_final_cycle |
| 31 unserved over decisions best | `31_unserved_over_decisions_best.png` | yes | best_final_cycle | generated | Representative episode: best_final_cycle |
| 32 unserved over decisions worst | `32_unserved_over_decisions_worst.png` | yes | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 33 action sequence random | `33_action_sequence_random.png` | yes | random_final_cycle | generated | Representative episode: random_final_cycle |
| 34 action sequence best | `34_action_sequence_best.png` | yes | best_final_cycle | generated | Representative episode: best_final_cycle |
| 35 action sequence worst | `35_action_sequence_worst.png` | yes | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 36 feasible actions random | `36_feasible_actions_random.png` | yes | random_final_cycle | generated | Representative episode: random_final_cycle |
| 37 feasible actions best | `37_feasible_actions_best.png` | yes | best_final_cycle | generated | Representative episode: best_final_cycle |
| 38 feasible actions worst | `38_feasible_actions_worst.png` | yes | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 39 service slack random | `39_service_slack_random.png` | no | random_final_cycle | generated | Representative episode: random_final_cycle |
| 40 service slack best | `40_service_slack_best.png` | no | best_final_cycle | generated | Representative episode: best_final_cycle |
| 41 service slack worst | `41_service_slack_worst.png` | no | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| 42 ev route balance best | `42_ev_route_balance_best.png` | no | best_final_cycle | generated | Representative episode: best_final_cycle |
| 43 ev route balance worst | `43_ev_route_balance_worst.png` | no | worst_final_cycle | generated | Representative episode: worst_final_cycle |
| Final-cycle instance summary table | `44_final_cycle_instance_table.html` | yes | final_cycle | generated | Final cycle per-instance metrics. |
