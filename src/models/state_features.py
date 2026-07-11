"""Environment feature extraction for actor and critic (Sections 5–10)."""
from __future__ import annotations

import numpy as np

from src.env.action_masks import ActionMaskResult
from src.env.masks import EPS, customer_feasible
from src.env.tw_diagnostics import customer_service_start
from src.models.feature_specs import (
    ACTING_EV_FEATURE_DIM,
    EV_TOKEN_SCALAR_DIM,
    FLEET_BALANCE_FEATURE_DIM,
    GLOBAL_STATE_FEATURE_DIM,
    MASK_SUMMARY_FEATURE_DIM,
)


def _horizon(env) -> float:
    return max(env.instance.planning_horizon, 1.0)


def _max_route_dist(env) -> float:
    inst = env.instance
    return max(float(inst.distance_matrix.max()) * max(inst.num_nodes, 1), 1.0)


def _steps_since_last_service(ev) -> float:
    return min(ev.steps_since_last_customer_service / 50.0, 1.0)


def acting_ev_features(env, ev) -> np.ndarray:
    inst = env.instance
    vp = inst.vehicle
    horizon = _horizon(env)
    max_dist = _max_route_dist(env)
    batt_cap = float(vp.battery_capacity or 1.0)
    cap = float(vp.load_capacity or 1.0)
    battery_frac = ev.remaining_battery / batt_cap if batt_cap > 0 else 0.0
    cap_frac = ev.remaining_capacity / cap if cap > 0 else 0.0
    return np.array([
        ev.current_time / horizon,
        battery_frac,
        cap_frac,
        ev.route_distance / max_dist,
        float(ev.active),
        float(ev.finished),
        1.0 - battery_frac,
        1.0 - cap_frac,
        _steps_since_last_service(ev),
    ], dtype=np.float32)


def ev_token_scalars(env, ev) -> np.ndarray:
    inst = env.instance
    vp = inst.vehicle
    horizon = _horizon(env)
    max_dist = _max_route_dist(env)
    n_cust = max(inst.num_customers, 1)
    return np.array([
        ev.current_time / horizon,
        ev.remaining_battery / vp.battery_capacity if vp.battery_capacity else 0.0,
        ev.remaining_capacity / vp.load_capacity if vp.load_capacity else 0.0,
        float(ev.active),
        float(ev.finished),
        ev.route_distance / max_dist,
        ev.num_customers_served / n_cust,
    ], dtype=np.float32)


def fleet_balance_features(env) -> np.ndarray:
    inst = env.instance
    vp = inst.vehicle
    n = max(len(env.evs), 1)
    dists = np.array([e.route_distance for e in env.evs], dtype=np.float64)
    served = np.array([e.num_customers_served for e in env.evs], dtype=np.float64)
    batt = np.array([
        e.remaining_battery / vp.battery_capacity if vp.battery_capacity else 0.0 for e in env.evs
    ], dtype=np.float64)
    caps = np.array([
        e.remaining_capacity / vp.load_capacity if vp.load_capacity else 0.0 for e in env.evs
    ], dtype=np.float64)

    def _std_norm(x: np.ndarray) -> float:
        if x.size <= 1:
            return 0.0
        m = float(x.mean())
        if abs(m) < 1e-9:
            return float(np.std(x) / max(np.max(np.abs(x)), 1e-9))
        return float(np.std(x) / m)

    return np.array([
        _std_norm(dists),
        _std_norm(served),
        _std_norm(batt),
        _std_norm(caps),
    ], dtype=np.float32)


def global_state_features(env, mask_result: ActionMaskResult | None = None) -> np.ndarray:
    inst = env.instance
    horizon = _horizon(env)
    n_cust = max(inst.num_customers, 1)
    total_demand = inst.total_demand or 1.0
    remaining_demand = sum(
        inst.nodes[c].demand for c in inst.customer_indices if not env.served_mask[c]
    )
    max_dist = _max_route_dist(env)
    active = sum(1 for e in env.evs if not e.finished)
    finished = sum(1 for e in env.evs if e.finished)
    gs = mask_result.global_serviceability if mask_result is not None else None
    gs_ratio = float(gs.globally_serviceable_customers_count) / n_cust if gs else 0.0
    expired = float(getattr(env, "last_expired_count", 0)) / n_cust
    feas_c = float(mask_result.feasible_customer_count) / n_cust if mask_result else 0.0
    n_st = max(len(inst.charging_station_indices), 1)
    feas_s = float(mask_result.feasible_station_count) / n_st if mask_result else 0.0

    slacks: list[float] = []
    ev = env.select_acting_ev()
    if ev is not None:
        travel = inst.travel_time_matrix
        for c in inst.customer_indices:
            if env.served_mask[c]:
                continue
            _, _, _, slack = customer_service_start(inst, ev, c)
            slacks.append(slack / horizon)
    min_slack = min(slacks) if slacks else 0.0
    mean_slack = float(np.mean(slacks)) if slacks else 0.0

    no_progress = 0
    for e in env.evs:
        if e.finished:
            continue
        if _steps_since_last_service(e) > 0.2:
            no_progress += 1
    no_progress_ratio = no_progress / max(env.m, 1)

    fb = fleet_balance_features(env)
    fleet_route_balance = float(fb[0])

    return np.clip(np.array([
        env.num_served_customers / n_cust,
        (n_cust - env.num_served_customers) / n_cust,
        remaining_demand / total_demand,
        active / max(env.m, 1),
        finished / max(env.m, 1),
        env.current_event_time / horizon,
        env.total_distance / max_dist,
        env.total_charging_time / horizon,
        env.total_waiting_time / horizon,
        gs_ratio,
        expired,
        feas_c,
        feas_s,
        min_slack,
        mean_slack,
        no_progress_ratio,
        fleet_route_balance,
        0.0,
        0.0,
        0.0,
    ], dtype=np.float32)[:GLOBAL_STATE_FEATURE_DIM], 0.0, 1.0)


def mask_summary_features(env, mask_result: ActionMaskResult) -> np.ndarray:
    inst = env.instance
    n = inst.num_nodes
    n_cust = max(inst.num_customers, 1)
    n_st = max(len(inst.charging_station_indices), 1)
    counts = mask_result.mask_reason_counts
    total_masked = max(n - int(mask_result.node_mask.sum()), 1)

    def _ratio(key: str) -> float:
        return float(counts.get(key, 0)) / total_masked

    return np.clip(np.array([
        mask_result.feasible_customer_count / n_cust,
        mask_result.feasible_station_count / n_st,
        float(mask_result.feasible_depot_recharge),
        float(mask_result.feasible_depot_finish),
        _ratio("masked_by_time_window"),
        _ratio("masked_by_battery"),
        _ratio("masked_by_capacity"),
        _ratio("masked_by_no_safe_continuation"),
        _ratio("masked_as_useless_station"),
        _ratio("masked_as_no_progress"),
    ], dtype=np.float32), 0.0, 1.0)


def battery_state_features(env, ev) -> np.ndarray:
    vp = env.instance.vehicle
    horizon = _horizon(env)
    batt_cap = float(vp.battery_capacity or 1.0)
    cap = float(vp.load_capacity or 1.0)
    batt_frac = ev.remaining_battery / batt_cap if batt_cap > 0 else 0.0
    cap_frac = ev.remaining_capacity / cap if cap > 0 else 0.0
    return np.array([
        batt_frac,
        1.0 - batt_frac,
        cap_frac,
        1.0 - cap_frac,
    ], dtype=np.float32)
