# Multi-Agent Transformer Solver for E-VRPTW

A research-grade solver for the **Electric Vehicle Routing Problem with Time
Windows and Recharging Stations (E-VRPTW)**. It combines a static Transformer
graph encoder with an **asynchronous, event-driven multi-agent decoder**, hard
feasibility masks, a learned charging-portion policy, REINFORCE training, and a
fleet-size search.

## Project goal

Given an instance (a depot, customers with demands and time windows, and
reusable charging stations), produce routes for a fleet of identical electric
vehicles that serve every customer exactly once while respecting:

- vehicle load capacity and battery capacity,
- energy consumption while driving and (partial) recharging at stations,
- customer time windows and service times,
- depot return for every active vehicle,
- continuous, per-vehicle (event-driven) time.

The most important modelling idea: **each EV has its own clock**. Vehicles do
not act in fixed global rounds. At every decision event the EV that becomes
available earliest (smallest `current_time`) makes the next decision. Because
all vehicles start at the depot at time 0, the first `m` decisions naturally
form a sequential initial dispatch; afterwards the schedule is fully
asynchronous and event-driven.

## Data

The solver reads only the converted, EVRP-ready instances:

```
dataset/evrptw_instances/<instance_name>/
    evrp_instance.json   # metadata, vehicle params, index sets, nodes
    nodes.csv            # per-node features incl. normalized ML features
    matrices.npz         # distance_matrix, travel_time_matrix, energy_matrix
    summary.json         # compact per-instance summary
```

The raw `dataset/vrptw_instances/` folder is the original Schneider-style data
and is **never read** by the solver. Place new converted instances under
`dataset/evrptw_instances/` to use them.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and PyTorch 2.x (CPU is sufficient; CUDA is used
automatically if available).

## Quick start

```bash
# Run the unit tests first (feasibility-critical logic).
pytest tests/ -v

# Train the policy.
python main_train.py --config configs/default.yaml

# Evaluate the trained model + baselines, with fleet-size search.
python main_evaluate.py --config configs/default.yaml
```

Useful overrides:

```bash
python main_train.py --config configs/default.yaml --epochs 50 --device cpu
python main_evaluate.py --config configs/default.yaml --checkpoint outputs/checkpoints/best.pt
```

To iterate quickly, restrict to the small reduced instances by setting
`dataset.train_size_filter: ["C5"]` (and `eval_size_filter`) in the config.

## Architecture

```
                 +-----------------------------+
 node features → |   Static Transformer        |  encode once per instance
 (all nodes)     |   Node Encoder              | → H = [h_0, ..., h_N]
                 +-----------------------------+
                                |
   event loop (repeat):         |  encoder always sees the whole graph
   ┌─────────────────────────────────────────────────────────────────┐
   │ 1. select EV with smallest current_time (ties: route len, ev_id)  │
   │ 2. compute hard feasibility mask for that EV                      │
   │ 3. EV State Encoder      → ev_context                             │
   │ 4. EV Communication      → other_ev_context                      │
   │ 5. Global Context MLP    → global_context                        │
   │ 6. Pointer decoder scores all nodes; infeasible → logit -1e9      │
   │ 7. sample / argmax a feasible node                               │
   │ 8. if station: charging-portion head picks {25,50,75,100}% charge │
   │ 9. environment applies physics, advances that EV's clock          │
   └─────────────────────────────────────────────────────────────────┘
```

Key invariant: the **encoder always sees all nodes** (served / reserved nodes
are never removed); feasibility is enforced **only** at the decoder softmax.

### Modules

- `src/data/instance_loader.py` — typed loading of converted instances.
- `src/data/evrptw_dataset.py` — dataset, train/eval split, padded batching.
- `src/env/ev_state.py` — per-EV state (clock, battery, capacity, traces).
- `src/env/event_queue.py` — earliest-available EV selection.
- `src/env/masks.py` — hard feasibility masks (the most safety-critical code).
- `src/env/cost.py` — objective `J` and penalty breakdown.
- `src/env/evrptw_env.py` — asynchronous event-driven environment + physics.
- `src/models/node_encoder.py` — static Transformer encoder.
- `src/models/ev_state_encoder.py` — acting-EV state encoder.
- `src/models/ev_communication.py` — inter-agent attention.
- `src/models/decoder.py` — global context, pointer head, charging head.
- `src/models/policy.py` — orchestrating policy (`select_action`).
- `src/training/` — rollout, moving-average baseline, REINFORCE loop.
- `src/evaluation/` — fleet search, baselines, metrics, evaluation pipeline.
- `src/utils/` — seeding, config, logging, tensors, plotting.

## Event-driven multi-agent decoding

At each event the environment exposes the EV with the smallest clock. That EV
decides where to go next; applying the action advances only that EV's clock, so
a different EV (still earlier in time) acts next. This continues until all
customers are served and all active EVs have returned to the depot, or a safety
step cap is reached.

## Served and reserved masks

A customer is in exactly one of three states:

- **unserved & unreserved** — selectable by any EV;
- **reserved** — chosen by an EV but not yet completed; not selectable by others
  (`reserved_mask`, `reserved_by`);
- **served** — completed; never selectable again (`served_mask`).

Charging stations are **reusable** and are never placed in the served mask: any
EV may visit any station any number of times (a zero-distance self-loop is the
only station move that is masked).

## Feasibility masks

`compute_feasible_mask(instance, env_state, ev)` returns a boolean mask over all
nodes plus a per-node reason dictionary for debugging. A customer `j` is
feasible from node `i` only if **all** of the following hold:

- `j` is neither served nor reserved;
- `demand[j] <= remaining_capacity`;
- `remaining_battery >= energy[i][j]` (can reach `j`);
- `remaining_battery >= energy[i][j] + min_r energy[j][r]` over rechargeable
  nodes `r` (will not be stranded after `j`);
- `arrival_j = current_time + travel_time[i][j] <= due_time[j]`
  (early arrival is allowed and incurs waiting; late arrival is infeasible);
- optionally, the EV can still return to the depot before the planning horizon.

A station is feasible if reachable on the current battery (and not a self-loop).
The depot is feasible if reachable; it is blocked at the start of a route while
useful work remains, which prevents the degenerate "return immediately without
serving anyone" behaviour. Infeasible nodes receive logit `-1e9`, so they can
never be sampled or chosen greedily.

## Charging-portion head

When the selected node is a station, the same policy chooses a charging portion
from `{0.25, 0.50, 0.75, 1.00}` of the *missing* battery:

```
missing        = battery_capacity - battery_after_arrival
charged_energy = portion * missing
new_battery    = min(battery_capacity, battery_after_arrival + charged_energy)
charging_time  = inverse_charging_rate * charged_energy
```

## Cost / reward

```
J = λ_v·V + λ_d·D + λ_t·T + λ_c·C_charge + λ_w·W + penalties
R = -J
```

`V` used EVs, `D` distance, `T` route duration, `C_charge` charging time,
`W` waiting time. Penalties cover unserved/duplicate customers, lateness,
battery/capacity violations, depot-return failures, unresolved reservations, and
a global infeasibility flag. All weights are configurable in
`configs/default.yaml`.

## Fleet-size search

```
m_min = ceil(total_customer_demand / load_capacity)
m_max = min(num_customers, m_min + extra_margin)
```

The solver runs decoding for each `m` in `[m_min, m_max]` and selects the best:

- **hierarchical**: feasible first, then fewest EVs, then shortest distance,
  then shortest duration, then least charging time;
- **scalar**: minimum `J` among feasible solutions.

If no feasible solution exists, the least-infeasible one is returned with its
reasons. Training uses a fixed `m = m_min` for speed (`training.use_fleet_search`).

## Training

REINFORCE with a moving-average baseline:

```
advantage = reward - baseline
loss      = -mean(advantage.detach() * sum_log_probs) - entropy_coef * entropy
```

Adam (`lr 1e-4`), gradient clipping (`1.0`), entropy regularization (`0.01`).
Per-epoch metrics are written to `outputs/logs/train_log.jsonl`; the best and
periodic checkpoints go to `outputs/checkpoints/`.

## Outputs

```
outputs/
    checkpoints/   best.pt, last.pt, epoch_*.pt
    logs/          train_log.jsonl
    results/       results.csv  (one row per instance × method)
    routes/        <instance>_routes.json  (per-EV detailed traces)
    figures/       training curves, fleet-search curves, route maps,
                   feasibility / distance / objective comparisons
```

`results.csv` columns include the selected fleet size, used EVs, feasibility,
distance, duration, waiting/charging time, station visits, every violation
count, the objective value, and runtime.

## Decoding modes

- **greedy** — pick the highest-probability feasible node / charging portion;
- **sampling** — sample from the masked distribution (used in training);
- **multi_sample** — draw `N` sampled solutions and keep the best feasible one.

## Tests

```bash
pytest tests/ -v
```

- `test_loader.py` — shapes, indices, vehicle params, tensors.
- `test_masks.py` — energy, stranding, time windows, capacity, reservation,
  station self-loop, depot blocking.
- `test_env_step.py` — reservation lifecycle, station reuse, charging math,
  depot return, asynchronous (smallest-clock) EV selection, sequential dispatch.
- `test_cost.py` — objective and penalty components, reward sign.
- `test_fleet_search.py` — fleet range, hierarchical/scalar selection, fallback.

## Limitations

- Training quality depends heavily on compute; the defaults are a starting
  point, not tuned for state-of-the-art results. Begin on reduced instances
  (`C5`/`C10`) and scale up.
- A small number of converted instances have customers whose direct
  depot→customer→recharge energy exceeds battery capacity (reachability
  pressure); these legitimately require intermediate charging and may be harder
  to solve feasibly.
- Decoding is per-instance and sequential by construction (event-driven), so
  throughput is bounded by the number of decision events rather than by GPU
  batch width.
- The horizon check is a conservative single-hop depot-return test, not a full
  look-ahead feasibility guarantee.
#   T r a n s f o r m e r _ M A R L _ E V R P  
 