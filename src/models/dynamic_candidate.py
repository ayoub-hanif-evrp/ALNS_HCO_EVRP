"""Per-candidate dynamic features for the pointer decoder (Section 7)."""
from __future__ import annotations

import numpy as np

from src.env.action_masks import ActionMaskResult
from src.env.masks import EPS, depot_time_window
from src.models.feature_specs import DYNAMIC_CANDIDATE_FEAT_DIM


def compute_dynamic_candidate_features(
    env,
    ev,
    mask_result: ActionMaskResult,
) -> np.ndarray:
    """Per-node dynamic features [N, DYNAMIC_CANDIDATE_FEAT_DIM] for the acting EV."""
    inst = env.instance
    n = inst.num_nodes
    horizon = inst.planning_horizon if inst.planning_horizon > 0 else 1.0
    max_dist = float(inst.distance_matrix.max()) * max(n, 1) or 1.0
    batt_cap = float(inst.vehicle.battery_capacity or 1.0)
    cap = float(inst.vehicle.load_capacity or 1.0)
    dep = inst.depot_index
    i = ev.current_node

    travel_t = inst.travel_time_matrix[i]
    travel_d = inst.distance_matrix[i]
    energy_i = inst.energy_matrix[i]

    ready = inst.ready_time_array()
    due = inst.due_time_array()
    svc_t = inst.service_time_array()
    demand = inst.demand_array()

    is_customer = np.zeros(n, dtype=bool)
    is_customer[inst.customer_indices] = True
    is_station = np.zeros(n, dtype=bool)
    is_station[inst.charging_station_indices] = True
    is_depot = np.zeros(n, dtype=bool)
    is_depot[dep] = True

    arrival_t = ev.current_time + travel_t
    travel_dist_n = travel_d / max_dist
    travel_time_n = travel_t / horizon
    energy_n = energy_i / batt_cap if batt_cap > 0 else 0.0
    arrival = arrival_t / horizon
    batt_after = np.clip((ev.remaining_battery - energy_i) / batt_cap if batt_cap > 0 else 0.0, -1.0, 1.0)
    cap_after = np.where(
        is_customer,
        np.clip((ev.remaining_capacity - demand) / cap if cap > 0 else 1.0, -1.0, 1.0),
        1.0,
    )

    served = env.served_mask
    reserved = env.reserved_mask
    use_tw = is_customer & ~served
    waiting = np.where(use_tw, np.maximum(0.0, ready - arrival_t), 0.0)
    service_start_t = np.where(use_tw, np.maximum(arrival_t, ready), arrival_t)
    service_end_t = np.where(
        is_customer,
        np.where(use_tw, service_start_t + svc_t, arrival_t + svc_t),
        arrival_t,
    )
    slack = np.where(use_tw, due - service_start_t, 0.0)

    se = np.where(is_customer, np.maximum(arrival_t, ready) + svc_t, arrival_t)
    travel_j_dep = inst.travel_time_matrix[:, dep]
    dep_due = float(inst.nodes[dep].due_time)
    depot_slack = (
        np.maximum(0.0, dep_due - (se + travel_j_dep)) / horizon
        if dep_due > 0 else np.zeros(n, dtype=np.float32)
    )

    hard_mask = mask_result.hard_node_mask
    if hard_mask is None:
        hard_mask = mask_result.node_mask
    useful_station_count = int(np.sum(mask_result.node_mask[inst.charging_station_indices]))
    useful_station_norm = useful_station_count / max(len(inst.charging_station_indices), 1)

    # Safe direct depot return from customer j after service:
    # battery_after_customer >= energy[j, depot]  (ONCE — do not subtract twice).
    need_safe = is_customer & ~served & ~reserved
    battery_after_customer = ev.remaining_battery - energy_i
    e_j_dep = inst.energy_matrix[:, dep]
    arr_dep = se + travel_j_dep
    _, due_dep, _ = depot_time_window(inst)
    safe_direct = (
        need_safe
        & (battery_after_customer + EPS >= e_j_dep)
        & (arr_dep <= due_dep + EPS)
    )
    if env.use_horizon_check:
        safe_direct &= arr_dep <= inst.planning_horizon + EPS

    safe_overall = need_safe & hard_mask
    safe_station = need_safe & (useful_station_count > 0)

    n_portions = max(len(env.charging_portions), 1)
    n_cust = max(inst.num_customers, 1)
    charge_feasible = np.zeros(n, dtype=np.float32)
    enabled_after = np.zeros(n, dtype=np.float32)
    for j, cm in mask_result.charge_masks.items():
        charge_feasible[j] = float(np.sum(cm)) / n_portions
        details = mask_result.charge_portion_details.get(j, [])
        if details:
            enabled_after[j] = float(max(d.enabled_customer_count for d in details if d.feasible)) / n_cust

    feats = np.column_stack([
        travel_dist_n,
        travel_time_n,
        energy_n,
        arrival,
        service_start_t / horizon,
        service_end_t / horizon,
        waiting / horizon,
        slack / horizon,
        depot_slack,
        batt_after,
        cap_after,
        safe_direct.astype(np.float32),
        np.where(need_safe, useful_station_count > 0, 0.0).astype(np.float32),
        safe_overall.astype(np.float32),
        np.full(n, useful_station_norm, dtype=np.float32),
        charge_feasible,
        enabled_after,
        is_customer.astype(np.float32),
        is_station.astype(np.float32),
        is_depot.astype(np.float32),
        mask_result.node_mask.astype(np.float32),
    ]).astype(np.float32)

    if feats.shape[1] != DYNAMIC_CANDIDATE_FEAT_DIM:
        raise RuntimeError(
            f"dynamic feature dim {feats.shape[1]} != {DYNAMIC_CANDIDATE_FEAT_DIM}",
        )
    return feats
