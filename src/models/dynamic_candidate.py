"""Per-candidate dynamic features for the pointer decoder (Section 7)."""
from __future__ import annotations

import numpy as np

from src.env.action_masks import ActionMaskResult
from src.env.masks import EPS
from src.env.tw_diagnostics import customer_service_start
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
    feats = np.zeros((n, DYNAMIC_CANDIDATE_FEAT_DIM), dtype=np.float32)
    i = ev.current_node
    travel_t = inst.travel_time_matrix
    travel_d = inst.distance_matrix
    energy = inst.energy_matrix

    for j in range(n):
        node = inst.nodes[j]
        t_ij = float(travel_t[i, j])
        d_ij = float(travel_d[i, j])
        e_ij = float(energy[i, j])
        arrival_t = ev.current_time + t_ij
        arrival = arrival_t / horizon
        travel_dist_n = d_ij / max_dist
        travel_time_n = t_ij / horizon
        energy_n = e_ij / batt_cap if batt_cap > 0 else 0.0
        batt_after = (ev.remaining_battery - e_ij) / batt_cap if batt_cap > 0 else 0.0
        cap_after = (ev.remaining_capacity - node.demand) / cap if node.is_customer and cap > 0 else 1.0

        if node.is_customer and not env.served_mask[j]:
            _, wait, ss, slack = customer_service_start(inst, ev, j)
            service_start = ss / horizon
            service_end = (ss + node.service_time) / horizon
            waiting = wait / horizon
            slack_n = slack / horizon
        else:
            service_start = arrival
            service_end = (arrival_t + node.service_time) / horizon if node.is_customer else arrival
            waiting = 0.0
            slack_n = 0.0

        depot_slack = 0.0
        se = arrival_t
        if node.is_customer:
            se = max(arrival_t, node.ready_time) + node.service_time
        ba = ev.remaining_battery - e_ij
        if inst.nodes[dep].due_time > 0:
            depot_slack = max(0.0, inst.nodes[dep].due_time - (se + travel_t[j, dep])) / horizon

        safe_overall = 0.0
        safe_direct_depot = 0.0
        safe_station = 0.0
        useful_station_count = 0
        hard_mask = mask_result.hard_node_mask
        if hard_mask is None:
            hard_mask = mask_result.node_mask
        if node.is_customer and not env.served_mask[j] and not env.reserved_mask[j]:
            safe_overall = float(hard_mask[j])
            se_c = max(ev.current_time + t_ij, node.ready_time) + node.service_time
            ba_c = ev.remaining_battery - e_ij
            from src.env.charge_masks import safe_depot_return_from
            ok_dep, _ = safe_depot_return_from(
                inst, j, se_c, ba_c - energy[j, dep], use_horizon_check=env.use_horizon_check,
            )
            safe_direct_depot = float(ok_dep)
            useful_station_count = int(np.sum(
                mask_result.node_mask[inst.charging_station_indices]
            ))
            safe_station = float(useful_station_count > 0)

        reachable_stations = 0
        if node.is_charging_station or (node.is_depot and env.use_depot_charging):
            reachable_stations = 1
        elif node.is_customer:
            ba_c = ev.remaining_battery - e_ij
            for s in inst.charging_station_indices:
                if ba_c + EPS >= energy[j, s]:
                    reachable_stations += 1

        charge_feasible = 0
        enabled_after = 0
        if j in mask_result.charge_masks:
            cm = mask_result.charge_masks[j]
            charge_feasible = int(np.sum(cm))
            details = mask_result.charge_portion_details.get(j, [])
            if details:
                enabled_after = int(max(d.enabled_customer_count for d in details if d.feasible))

        feats[j] = np.array([
            travel_dist_n,
            travel_time_n,
            energy_n,
            arrival,
            service_start,
            service_end,
            waiting,
            slack_n,
            depot_slack,
            np.clip(batt_after, -1.0, 1.0),
            np.clip(cap_after, -1.0, 1.0),
            safe_direct_depot,
            safe_station,
            safe_overall,
            useful_station_count / max(len(inst.charging_station_indices), 1),
            charge_feasible / max(len(env.charging_portions), 1),
            enabled_after / max(inst.num_customers, 1),
            float(node.is_customer),
            float(node.is_charging_station),
            float(node.is_depot),
            float(mask_result.node_mask[j]),
        ], dtype=np.float32)
    return feats
