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


def _min_energy_to_recharge(instance, j: int) -> float:
    """Minimum energy needed to go from node j to any rechargeable node.

    Rechargeable nodes are the depot plus all charging stations. The depot is
    included, so a customer that can be reached and then left toward the depot
    is not considered stranded.
    """
    energy = instance.energy_matrix
    rec = instance.rechargeable_indices
    return float(min(energy[j, r] for r in rec))


def customer_feasible(
    instance,
    ev: EVState,
    j: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    use_horizon_check: bool = True,
) -> tuple[bool, str]:
    """Return (feasible, reason) for visiting customer j from the EV's position."""
    i = ev.current_node
    energy = instance.energy_matrix
    travel = instance.travel_time_matrix

    if served_mask[j]:
        return False, "already_served"
    if reserved_mask[j]:
        return False, "reserved_by_other"

    demand = instance.nodes[j].demand
    if demand > ev.remaining_capacity + EPS:
        return False, "capacity_exceeded"

    energy_to_j = energy[i, j]
    if ev.remaining_battery + EPS < energy_to_j:
        return False, "battery_cannot_reach"

    # Must still be able to reach a rechargeable node after serving j.
    min_after = _min_energy_to_recharge(instance, j)
    if ev.remaining_battery + EPS < energy_to_j + min_after:
        return False, "battery_would_strand"

    arrival = ev.current_time + travel[i, j]
    ready = float(instance.nodes[j].ready_time)
    due = float(instance.nodes[j].due_time)
    service_start = max(arrival, ready)
    if service_start > due + EPS:
        return False, "time_window_late"

    if use_horizon_check:
        departure = service_start + instance.nodes[j].service_time
        ok, reason = depot_return_feasible(instance, j, departure, use_horizon_check=use_horizon_check)
        if not ok:
            return False, reason

    return True, "ok"


def station_feasible(
    instance,
    ev: EVState,
    s: int,
    use_horizon_check: bool = True,
) -> tuple[bool, str]:
    """Return (feasible, reason) for visiting charging station s."""
    i = ev.current_node
    if s == i:
        return False, "station_self_loop"
    energy_to_s = instance.energy_matrix[i, s]
    if ev.remaining_battery + EPS < energy_to_s:
        return False, "battery_cannot_reach"
    if use_horizon_check:
        arrival = ev.current_time + instance.travel_time_matrix[i, s]
        if arrival > instance.planning_horizon + EPS:
            return False, "horizon_exceeded"
    return True, "ok"


def depot_recharge_feasible(
    instance,
    ev: EVState,
    use_depot_charging: bool = True,
    use_horizon_check: bool = True,
) -> tuple[bool, str]:
    """Return (feasible, reason) for visiting the depot to recharge (free).

    The depot is never a learned route-termination action; it behaves like a
    charging station with zero price when ``depot_has_charging`` is enabled.
    """
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
    return True, "ok"


def must_close_route(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_depot_charging: bool = True,
    use_horizon_check: bool = True,
) -> bool:
    """True when the EV cannot serve any customer and recharging cannot help."""
    for j in instance.customer_indices:
        ok, reason = customer_feasible(
            instance, ev, j, served_mask, reserved_mask, use_horizon_check
        )
        if ok:
            return False
        if reason in ("battery_cannot_reach", "battery_would_strand"):
            for s in instance.charging_station_indices:
                ok_s, _ = station_feasible(instance, ev, s, use_horizon_check)
                if ok_s:
                    return False
            ok_d, _ = depot_recharge_feasible(
                instance, ev, use_depot_charging, use_horizon_check
            )
            if ok_d:
                return False
    return True


def compute_feasible_mask(
    instance,
    env_state,
    ev: EVState,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
) -> tuple[np.ndarray, dict[int, str]]:
    """Compute the boolean feasibility mask over all nodes for the acting EV.

    Args:
        instance: EVRPTWInstance.
        env_state: object exposing ``served_mask``, ``reserved_mask`` (bool
            arrays over nodes) and ``num_served_customers`` (int).
        ev: the EV currently making a decision.
        use_horizon_check: whether to enforce the planning-horizon return check.

    Returns:
        (feasible_mask, infeasible_reasons)
        feasible_mask: bool array [N], True where selectable.
        infeasible_reasons: {node_id: reason} for masked nodes (debugging).
    """
    n = instance.num_nodes
    feasible = np.zeros(n, dtype=bool)
    reasons: dict[int, str] = {}

    served = env_state.served_mask
    reserved = env_state.reserved_mask
    all_done = env_state.num_served_customers >= instance.num_customers

    # Customers first (also tells us whether the depot fallback is needed).
    for j in instance.customer_indices:
        ok, reason = customer_feasible(instance, ev, j, served, reserved, use_horizon_check)
        feasible[j] = ok
        if not ok:
            reasons[j] = reason

    # Charging stations: reusable, masked only if unreachable / self-loop.
    # When every customer is done, stop charging and force the EV home.
    for s in instance.charging_station_indices:
        if all_done:
            feasible[s] = False
            reasons[s] = "all_customers_done"
            continue
        ok, reason = station_feasible(instance, ev, s, use_horizon_check)
        feasible[s] = ok
        if not ok:
            reasons[s] = reason

    # Depot: free recharge only (route closure is handled by the environment).
    depot = instance.depot_index
    if all_done:
        feasible[depot] = False
        reasons[depot] = "all_customers_done"
    else:
        ok, reason = depot_recharge_feasible(
            instance, ev, use_depot_charging, use_horizon_check
        )
        feasible[depot] = ok
        if not ok:
            reasons[depot] = reason

    return feasible, reasons


def has_any_feasible(feasible_mask: np.ndarray) -> bool:
    return bool(feasible_mask.any())
