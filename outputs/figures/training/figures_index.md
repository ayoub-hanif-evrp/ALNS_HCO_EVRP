# MARL EVRP Training Figures Index

Training-only metrics (no validation or held-out test instances).

| Graph | File | Metrics | Status | Explanation |
|-------|------|---------|--------|-------------|
| Training objective over epochs | `01_training_objective.png` | cost | generated | Shows whether the MARL policy is reducing the EVRP solution cost. |
| Episode return over epochs | `02_episode_return.png` | reward | generated | Shows whether the actor-critic learning signal improves over time. |
| Feasibility moving average | `03_feasibility_moving_average.png` | feasibility_rate | generated | Shows whether the model produces feasible EVRP solutions more often. |
| Served-customer ratio over epochs | `04_served_customer_ratio.png` | served_rate | generated | Shows whether the policy learns to serve all customers. |
| Constraint violations over training | `05_constraint_violations.png` | unserved_customers, avg_battery_violation, avg_lateness, avg_capacity_violation, avg_depot_violation, avg_duplicate_visits, mean_pen_unserved, mean_pen_battery, mean_pen_late, mean_pen_capacity, mean_pen_depot, mean_pen_duplicate | generated | EVRP constraint learning: battery, capacity, time windows, depot return. |
| Actor-critic losses | `06_actor_critic_losses.png` | actor_loss, critic_loss, total_loss, loss | generated | Training stability of the MARL actor-critic system. |
| Policy entropy over epochs | `07_policy_entropy.png` | entropy | generated | Exploration — entropy should not collapse too early. |
| Critic explained variance | `08_critic_explained_variance.png` | critic_explained_variance | generated | Whether the critic learns useful value estimates. |
| Distance and duration | `09a_distance_duration.png` | avg_distance, avg_duration | generated | Route distance and duration over training. |
| Charging and waiting | `09b_charging_waiting.png` | avg_charging_time, avg_charging_price, avg_waiting_time | generated | Charging and waiting effort over training. |
| Number of used EVs | `10_used_evs.png` | avg_used_evs | generated | Whether the model learns efficient fleet usage. |
| Charging-station visits over training | `11a_station_visits.png` | charging_station_visits | generated | EV recharging station usage during training. |
| Charging effort over training | `11b_charging_effort.png` | avg_charging_time, avg_charging_price | generated | Total charging time and monetary charging cost. |
| Minimum battery level over training | `11c_min_battery.png` | min_remaining_battery | generated | Minimum remaining battery at episode end. |
| Fleet coordination and route balance | `12_fleet_coordination_balance.png` | route_balance_cv | generated | Std/mean of per-EV route distance — load balance across agents. |
| Feasible action ratio | `13_feasible_action_ratio.png` | feasible_action_ratio | generated | How restrictive environment masks are during training. |
| Cycle-level objective progress | `14a_cycle_objective.png` | cost | generated | Mean objective per full pass over training instances. |
| Cycle-level feasibility and served ratio | `14b_cycle_feasibility_served.png` | feasibility_rate, served_rate | generated | Aggregate feasibility and coverage per cycle. |
| Cycle-level violations | `14c_cycle_violations.png` | avg_violation, unserved_customers, avg_battery_violation, avg_lateness | generated | Mean violations per training cycle. |
| Per-instance objective across cycles | `15a_per_instance_objective_heatmap.png` | cost | generated | Objective heatmap: instances × cycles. |
| Per-instance feasibility across cycles | `15b_per_instance_feasibility_heatmap.png` | feasibility_rate | generated | Feasibility heatmap: instances × cycles. |
| Final-cycle objective by training instance | `16_final_cycle_instance_objective.png` | cost | generated | Hardest training instances in the last cycle. |
| Training stability diagnostics | `17_training_stability_diagnostics.png` | grad_norm, cuda_max_mem_allocated_mb, cuda_mem_allocated_mb | generated | Gradient norms and GPU memory during training. |
