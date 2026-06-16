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
    due = instance.nodes[j].due_time
    if arrival > due + EPS:
        return False, "time_window_late"

    if use_horizon_check:
        ready = instance.nodes[j].ready_time
        service_start = max(arrival, ready)
        departure = service_start + instance.nodes[j].service_time
        # Must be able to return to the depot before the planning horizon.
        back_to_depot = travel[j, instance.depot_index]
        if departure + back_to_depot > instance.planning_horizon + EPS:
            return False, "horizon_exceeded"

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


def depot_feasible(
    instance,
    ev: EVState,
    has_feasible_customer: bool,
    all_customers_done: bool,
) -> tuple[bool, str]:
    """Return (feasible, reason) for returning to (or staying at) the depot.

    Rules:
        - Must be reachable on the remaining battery.
        - Cannot select the depot while already standing on it (no self-loop);
          this also blocks the degenerate "return immediately without serving".
    """
    depot = instance.depot_index
    if ev.current_node == depot:
        return False, "depot_self_loop"
    energy_to_depot = instance.energy_matrix[ev.current_node, depot]
    if ev.remaining_battery + EPS < energy_to_depot:
        return False, "battery_cannot_reach"
    return True, "ok"


def compute_feasible_mask(
    instance,
    env_state,
    ev: EVState,
    use_horizon_check: bool = True,
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

    has_feasible_customer = bool(feasible[instance.customer_indices].any()) if instance.customer_indices else False

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

    # Depot.
    depot = instance.depot_index
    ok, reason = depot_feasible(instance, ev, has_feasible_customer, all_done)
    # Block the depot at the very start of a route when useful work remains,
    # so an EV does not "finish" without serving anyone.
    at_route_start = ev.current_node == depot and len(ev.route_nodes) <= 1
    if at_route_start and has_feasible_customer:
        feasible[depot] = False
        reasons[depot] = "depot_blocked_at_start"
    else:
        feasible[depot] = ok
        if not ok:
            reasons[depot] = reason

    return feasible, reasons


def has_any_feasible(feasible_mask: np.ndarray) -> bool:
    return bool(feasible_mask.any())
