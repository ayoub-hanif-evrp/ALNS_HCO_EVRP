"""Hard feasibility masks for the E-VRPTW decoder.

The model must never select an infeasible node. These functions compute, for the
currently-acting EV, a boolean mask over all nodes. Infeasible nodes get logit
``-1e9`` before the softmax so they can never be sampled or chosen greedily.

All functions are pure and operate on NumPy arrays so they can be unit-tested in
isolation from the neural network.
"""
from __future__ import annotations

import numpy as np

from .ev_state import EVState

EPS = 1e-6


def depot_time_window(instance) -> tuple[float, float, float]:
    """Return (ready, due, service_time) for the depot closing window."""
    depot = instance.depot_index
    node = instance.nodes[depot]
    return float(node.ready_time), float(node.due_time), float(node.service_time)


def depot_return_arrival(instance, from_node: int, depart_time: float) -> float:
    """Clock time when an EV leaving ``from_node`` at ``depart_time`` reaches the depot."""
    depot = instance.depot_index
    return float(depart_time + instance.travel_time_matrix[from_node, depot])


def depot_return_feasible(
    instance,
    from_node: int,
    depart_time: float,
    *,
    use_horizon_check: bool = True,
) -> tuple[bool, str]:
    """Check whether the EV can reach the depot before its closing time window."""
    arrival = depot_return_arrival(instance, from_node, depart_time)
    _, due, _ = depot_time_window(instance)
    if arrival > due + EPS:
        return False, "depot_window_late"
    if use_horizon_check and arrival > instance.planning_horizon + EPS:
        return False, "horizon_exceeded"
    return True, "ok"


def _safe_continuation_after_customer(
    instance,
    j: int,
    service_end: float,
    battery_after: float,
    capacity_after: float,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, str]:
    """Check depot return or station+charge continuation after serving customer j."""
    from .charge_masks import feasible_charge_portions, safe_depot_return_from

    if battery_after < -EPS:
        return False, "customer_infeasible_after_service_battery"
    if capacity_after < -EPS:
        return False, "customer_capacity_infeasible"

    ok_dep, reason = safe_depot_return_from(
        instance, j, service_end, battery_after, use_horizon_check=use_horizon_check
    )
    if ok_dep:
        return True, "customer_feasible"

    travel = instance.travel_time_matrix
    energy = instance.energy_matrix
    sim = EVState(
        ev_id=0,
        current_node=j,
        current_time=service_end,
        remaining_battery=battery_after,
        remaining_capacity=capacity_after,
    )
    any_station_reachable = False
    for s in instance.charging_station_indices:
        if battery_after + EPS < energy[j, s]:
            continue
        if use_horizon_check and service_end + travel[j, s] > instance.planning_horizon + EPS:
            continue
        any_station_reachable = True
        portion_mask, _ = feasible_charge_portions(
            instance, sim, s, served_mask, reserved_mask,
            portions=charging_portions,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
        )
        if portion_mask.any():
            return True, "customer_feasible"

    if not any_station_reachable:
        if reason.startswith("charge_portion_infeasible"):
            return False, "customer_no_safe_direct_depot"
        return False, "customer_no_reachable_station"
    return False, "customer_station_has_no_feasible_charge"


def customer_feasible(
    instance,
    ev: EVState,
    j: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, str]:
    """Return (feasible, reason) for visiting customer j from the EV's position."""
    from .feasibility_scratch import ev_state_key, get_scratch

    scratch = get_scratch()
    cache_key = None
    if scratch is not None:
        cache_key = (*ev_state_key(ev), int(j))
        cached = scratch.customer.get(cache_key)
        if cached is not None:
            return cached

    def _ret(ok: bool, reason: str) -> tuple[bool, str]:
        out = (ok, reason)
        if scratch is not None and cache_key is not None:
            scratch.customer[cache_key] = out
        return out

    i = ev.current_node
    energy = instance.energy_matrix
    travel = instance.travel_time_matrix

    if served_mask[j]:
        return _ret(False, "customer_already_served")
    if reserved_mask[j]:
        return _ret(False, "customer_reserved")

    demand = instance.nodes[j].demand
    if demand > ev.remaining_capacity + EPS:
        return _ret(False, "customer_capacity_infeasible")

    energy_to_j = energy[i, j]
    if ev.remaining_battery + EPS < energy_to_j:
        return _ret(False, "customer_battery_unreachable")

    arrival = ev.current_time + travel[i, j]
    ready = float(instance.nodes[j].ready_time)
    due = float(instance.nodes[j].due_time)
    service_start = max(arrival, ready)
    if service_start > due + EPS:
        return _ret(False, "customer_time_window_infeasible")

    service_end = service_start + instance.nodes[j].service_time
    battery_after = ev.remaining_battery - energy_to_j
    capacity_after = ev.remaining_capacity - demand

    ok_cont, reason = _safe_continuation_after_customer(
        instance, j, service_end, battery_after, capacity_after,
        served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    if not ok_cont:
        return _ret(False, reason)
    return _ret(True, "customer_feasible")


def station_feasible(
    instance,
    ev: EVState,
    s: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, str]:
    """Return (feasible, reason) for visiting charging station s (physical layer)."""
    from .charge_masks import feasible_charge_portions

    i = ev.current_node
    if s == i:
        return False, "station_loop_blocked"
    energy_to_s = instance.energy_matrix[i, s]
    if ev.remaining_battery + EPS < energy_to_s:
        return False, "station_battery_unreachable"
    if use_horizon_check:
        arrival = ev.current_time + instance.travel_time_matrix[i, s]
        if arrival > instance.planning_horizon + EPS:
            return False, "station_horizon_infeasible"
    portion_mask, _ = feasible_charge_portions(
        instance, ev, s, served_mask, reserved_mask,
        portions=charging_portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
    )
    if not portion_mask.any():
        return False, "station_no_feasible_charge_portion"
    return True, "station_feasible"


def depot_finish_feasible(
    instance,
    ev: EVState,
    use_horizon_check: bool = True,
) -> tuple[bool, str]:
    """Return (feasible, reason) for intentionally finishing at the depot."""
    depot = instance.depot_index
    if ev.current_node == depot:
        if len(ev.route_nodes) <= 1:
            return False, "depot_finish_at_start"
        return True, "depot_finish_at_depot"
    energy_to_depot = instance.energy_matrix[ev.current_node, depot]
    if ev.remaining_battery + EPS < energy_to_depot:
        return False, "battery_cannot_reach_depot"
    ok, reason = depot_return_feasible(
        instance, ev.current_node, ev.current_time, use_horizon_check=use_horizon_check
    )
    if not ok:
        return False, reason
    return True, "depot_finish_feasible"


def depot_recharge_feasible(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    use_depot_charging: bool = True,
    use_horizon_check: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, str]:
    """Return (feasible, reason) for visiting the depot to recharge (free)."""
    from .charge_masks import feasible_charge_portions

    if not use_depot_charging or not instance.depot_has_charging:
        return False, "depot_charging_disabled"
    depot = instance.depot_index
    if ev.current_node == depot:
        return False, "depot_self_loop"
    energy_to_depot = instance.energy_matrix[ev.current_node, depot]
    if ev.remaining_battery + EPS < energy_to_depot:
        return False, "battery_cannot_reach"
    if use_horizon_check:
        ok, reason = depot_return_feasible(
            instance, ev.current_node, ev.current_time, use_horizon_check=use_horizon_check
        )
        if not ok:
            return False, reason
    portion_mask, _ = feasible_charge_portions(
        instance, ev, depot, served_mask, reserved_mask,
        portions=charging_portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
    )
    if not portion_mask.any():
        return False, "depot_no_feasible_charge_portion"
    return True, "depot_recharge_feasible"


def must_close_route(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_depot_charging: bool = True,
    use_horizon_check: bool = True,
    charging_portions: list[float] | None = None,
    num_served_customers: int | None = None,
) -> bool:
    """True when no customer, station, or depot action is feasible for this EV."""
    if num_served_customers is None:
        num_served_customers = int(np.sum(served_mask[instance.customer_indices]))

    class _MaskState:
        def __init__(self):
            self.served_mask = served_mask
            self.reserved_mask = reserved_mask
            self.num_served_customers = num_served_customers

    mask, _ = compute_feasible_mask(
        instance, _MaskState(), ev, use_horizon_check, use_depot_charging, charging_portions,
    )
    return not mask.any()


def compute_feasible_mask(
    instance,
    env_state,
    ev: EVState,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Compute exact feasible mask — delegates to ``compute_full_feasible_actions``."""
    from .action_space import compute_full_feasible_actions

    feasible, reasons, summary, global_svc = compute_full_feasible_actions(
        instance, env_state, ev, use_horizon_check, use_depot_charging, charging_portions,
    )
    if hasattr(env_state, "last_feasible_summary"):
        env_state.last_feasible_summary = summary
    if hasattr(env_state, "last_global_serviceability"):
        env_state.last_global_serviceability = global_svc
    return feasible, reasons


def depot_mode_feasible(
    instance,
    env_state,
    ev: EVState,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, bool]:
    """Return (recharge_feasible, finish_feasible) for depot action sub-choice."""
    from .action_space import (
        compute_depot_finish_validity,
        compute_depot_recharge_usefulness,
        compute_global_serviceability,
    )

    served = env_state.served_mask
    reserved = env_state.reserved_mask
    all_done = env_state.num_served_customers >= instance.num_customers
    evs = getattr(env_state, "evs", [ev])
    global_svc = compute_global_serviceability(
        instance, evs, served, reserved,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    finish_val = compute_depot_finish_validity(
        instance, ev, served, reserved, env_state.num_served_customers,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    ok_f = finish_val.valid and finish_val.category != "invalid"
    if all_done:
        return False, ok_f
    ok_r_raw, _ = depot_recharge_feasible(
        instance, ev, served, reserved, use_depot_charging, use_horizon_check, charging_portions,
    )
    useful_r, _, _ = compute_depot_recharge_usefulness(
        instance, ev, served, reserved,
        all_customers_served=False,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    return ok_r_raw and useful_r, ok_f


def has_any_feasible(feasible_mask: np.ndarray) -> bool:
    return bool(feasible_mask.any())
