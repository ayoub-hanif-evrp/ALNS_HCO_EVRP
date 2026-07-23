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
