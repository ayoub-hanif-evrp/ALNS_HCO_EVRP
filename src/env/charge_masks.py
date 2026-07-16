"""Feasibility of charging portions at stations / depot (shared by mask and policy)."""
from __future__ import annotations

import numpy as np

from .ev_state import EVState

EPS = 1e-6
CHARGE_PORTIONS = (0.25, 0.50, 0.75, 1.0)


def _charging_rate(instance, node: int) -> float:
    if instance.station_charging_rate is not None:
        rate = float(instance.station_charging_rate[node])
        if rate > 1e-9 and not np.isnan(rate):
            return rate
    vp = instance.vehicle
    if vp.inverse_charging_rate > 1e-9:
        return 1.0 / vp.inverse_charging_rate
    return 1.0


def simulate_recharge_departure(
    instance,
    ev: EVState,
    node: int,
    portion: float,
    *,
    free: bool,
) -> tuple[float, float, float, float]:
    """Return (departure_time, battery_after, charging_time, energy_to_node)."""
    i = ev.current_node
    t = float(instance.travel_time_matrix[i, node])
    e = float(instance.energy_matrix[i, node])
    arrival = float(ev.current_time + t)
    battery_after_travel = float(ev.remaining_battery - e)
    vp = instance.vehicle
    missing = max(0.0, vp.battery_capacity - battery_after_travel)
    charged = float(portion) * missing
    battery_after = min(vp.battery_capacity, battery_after_travel + charged)
    charge_time = charged / _charging_rate(instance, node)
    if instance.nodes[node].is_depot:
        ready = float(instance.nodes[node].ready_time)
        charge_start = max(arrival, ready)
    else:
        charge_start = arrival
    departure = charge_start + charge_time
    return departure, battery_after, charge_time, e


def safe_depot_return_from(
    instance,
    from_node: int,
    depart_time: float,
    battery: float,
    *,
    use_horizon_check: bool = True,
) -> tuple[bool, str]:
    """Depot return requires enough battery AND arrival before depot due time."""
    from .masks import depot_return_arrival, depot_time_window

    depot = instance.depot_index
    energy_to_depot = float(instance.energy_matrix[from_node, depot])
    if battery + EPS < energy_to_depot:
        return False, "charge_portion_infeasible_battery_to_depot"
    arrival = depot_return_arrival(instance, from_node, depart_time)
    _, due, _ = depot_time_window(instance)
    if arrival > due + EPS:
        return False, "charge_portion_infeasible_time_to_depot"
    if use_horizon_check and arrival > instance.planning_horizon + EPS:
        return False, "horizon_exceeded"
    return True, "charge_portion_feasible"


def _customer_feasible_from_state(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    use_horizon_check: bool,
    use_depot_charging: bool,
    charging_portions: list[float] | None,
) -> bool:
    from .action_space import _has_feasible_customer

    return _has_feasible_customer(
        instance, ev, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )


def _can_continue_after_recharge(
    instance,
    ev: EVState,
    node: int,
    departure: float,
    battery_after: float,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool,
    use_depot_charging: bool,
    charging_portions: list[float] | None,
) -> tuple[bool, str]:
    """After charging at ``node``, require safe depot return OR a feasible customer path."""
    ok_dep, reason_dep = safe_depot_return_from(
        instance, node, departure, battery_after, use_horizon_check=use_horizon_check
    )
    if ok_dep:
        return True, "charge_portion_feasible"

    sim = EVState(
        ev_id=ev.ev_id,
        current_node=node,
        current_time=departure,
        remaining_battery=battery_after,
        remaining_capacity=ev.remaining_capacity,
    )
    if _customer_feasible_from_state(
        instance, sim, served_mask, reserved_mask, use_horizon_check,
        use_depot_charging, charging_portions,
    ):
        return True, "charge_portion_feasible"

    if reason_dep in (
        "charge_portion_infeasible_battery_to_depot",
        "charge_portion_infeasible_time_to_depot",
    ):
        return False, "charge_portion_infeasible_no_safe_continuation"
    return False, reason_dep if reason_dep != "charge_portion_feasible" else "charge_portion_infeasible_no_safe_continuation"


def feasible_charge_portions(
    instance,
    ev: EVState,
    node: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    portions: list[float] | tuple[float, ...] | None = None,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Boolean mask over charge portions with full continuation checks."""
    from .feasibility_scratch import ev_state_key, get_scratch
    from .analysis_counters import bump_outside_analyze

    bump_outside_analyze("charge_portion_recompute_calls")

    p_list = list(portions or CHARGE_PORTIONS)
    scratch = get_scratch()
    if scratch is not None:
        cache_key = (*ev_state_key(ev), int(node), tuple(p_list))
        cached = scratch.charge_portions.get(cache_key)
        if cached is not None:
            mask, reasons = cached
            return mask.copy(), list(reasons)
    mask = np.zeros(len(p_list), dtype=bool)
    reasons: list[str] = [""] * len(p_list)
    free = instance.nodes[node].is_depot
    i = ev.current_node
    energy_to = float(instance.energy_matrix[i, node])
    if ev.remaining_battery + EPS < energy_to:
        return mask, ["station_battery_unreachable"] * len(p_list)

    for idx, p in enumerate(p_list):
        departure, batt, _, _ = simulate_recharge_departure(instance, ev, node, p, free=free)
        if use_horizon_check and departure > instance.planning_horizon + EPS:
            reasons[idx] = "horizon_exceeded"
            continue
        ok, reason = _can_continue_after_recharge(
            instance, ev, node, departure, batt, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
        )
        if ok:
            mask[idx] = True
            reasons[idx] = "charge_portion_feasible"
        else:
            reasons[idx] = reason
    if scratch is not None:
        scratch.charge_portions[cache_key] = (mask.copy(), list(reasons))
    return mask, reasons
