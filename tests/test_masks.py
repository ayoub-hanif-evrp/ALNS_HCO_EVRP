"""Tests for feasibility masks: energy, time windows, capacity, reservation."""
from __future__ import annotations

import numpy as np

from src.env.evrptw_env import EVRPTWEnv
from src.env.ev_state import EVState
from src.env.masks import compute_feasible_mask, customer_feasible, depot_return_feasible


def _fresh_env(instance, m=2):
    return EVRPTWEnv(instance, m=m, use_horizon_check=True)


def test_served_and_reserved_are_masked(small_instance):
    env = _fresh_env(small_instance)
    ev = env.evs[0]
    j = small_instance.customer_indices[0]
    # Initially feasible.
    mask, _ = env.feasible_mask(ev)
    assert mask[j]
    # Reserve it -> infeasible for another EV.
    env.reserve_customer(env.evs[0], j)
    ev2 = env.evs[1]
    mask2, reasons2 = env.feasible_mask(ev2)
    assert not mask2[j]
    assert reasons2[j] == "reserved_by_other"


def test_energy_cannot_reach_is_masked(small_instance):
    env = _fresh_env(small_instance)
    ev = env.evs[0]
    # Drain the battery so almost nothing is reachable.
    ev.remaining_battery = 0.0
    mask, reasons = env.feasible_mask(ev)
    for j in small_instance.customer_indices:
        assert not mask[j]
        assert reasons[j] in {"battery_cannot_reach", "battery_would_strand"}


def test_battery_would_strand_rule(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    i = ev.current_node
    j = inst.customer_indices[0]
    energy_to_j = inst.energy_matrix[i, j]
    min_after = min(inst.energy_matrix[j, r] for r in inst.rechargeable_indices)
    # Enough to reach j, but not enough to then reach any recharge node.
    ev.remaining_battery = energy_to_j + 0.5 * min_after
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, True)
    assert not ok
    assert reason == "battery_would_strand"


def test_capacity_exceeded_is_masked(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    ev.remaining_capacity = inst.nodes[j].demand - 1.0
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, True)
    assert not ok
    assert reason == "capacity_exceeded"


def test_time_window_late_when_service_start_exceeds_due(small_instance):
    """service_start = max(arrival, ready) must be <= due."""
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    node = inst.nodes[j]
    travel = float(inst.travel_time_matrix[ev.current_node, j])
    ev.current_time = float(node.due_time) - travel + 1.0
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, use_horizon_check=False)
    assert not ok
    assert reason == "time_window_late"


def test_time_window_late_is_masked(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    # Push current time well past the due date.
    ev.current_time = inst.nodes[j].due_time + 1000.0
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, use_horizon_check=False)
    assert not ok
    assert reason == "time_window_late"


def test_early_arrival_is_feasible_with_waiting(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    # Find a customer with a strictly positive ready time.
    target = None
    for j in inst.customer_indices:
        if inst.nodes[j].ready_time > 0:
            target = j
            break
    if target is None:
        return  # nothing to assert for this instance
    ev.current_time = 0.0
    ok, reason = customer_feasible(inst, ev, target, env.served_mask, env.reserved_mask, use_horizon_check=False)
    # Early arrival is allowed (waiting), not infeasible due to the window.
    assert ok or reason in {"battery_would_strand", "battery_cannot_reach"}


def test_station_self_loop_masked(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    s = inst.charging_station_indices[0]
    ev.current_node = s
    mask, reasons = env.feasible_mask(ev)
    assert not mask[s]
    assert reasons[s] == "station_self_loop"


def test_depot_return_window_blocks_customer(small_instance):
    inst = small_instance
    ev = EVState.initial(0, inst.depot_index, inst.vehicle.battery_capacity, inst.vehicle.load_capacity)
    j = inst.customer_indices[0]
    depot = inst.depot_index
    # Simulate EV already at customer, too late to return before depot closes.
    ev.current_node = j
    travel_back = inst.travel_time_matrix[j, depot]
    due = inst.nodes[depot].due_time
    ev.current_time = due - travel_back + 1.0
    ok, reason = depot_return_feasible(inst, j, ev.current_time, use_horizon_check=True)
    assert not ok
    assert reason == "depot_window_late"


def test_depot_return_feasible_helper(small_instance):
    inst = small_instance
    j = inst.customer_indices[0]
    ok, _ = depot_return_feasible(inst, j, 0.0, use_horizon_check=True)
    assert ok


def test_depot_blocked_at_start_when_customers_feasible(small_instance):
    inst = small_instance
    env = _fresh_env(small_instance)
    ev = env.evs[0]
    mask, reasons = env.feasible_mask(ev)
    # At route start on depot, recharge is a self-loop and must be blocked.
    assert not mask[inst.depot_index]
    assert reasons[inst.depot_index] == "depot_self_loop"


def test_stations_masked_when_all_served(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    for c in inst.customer_indices:
        env.served_mask[c] = True
    env.num_served_customers = inst.num_customers
    ev = env.evs[0]
    ev.current_node = inst.customer_indices[0]  # away from depot
    mask, reasons = env.feasible_mask(ev)
    for s in inst.charging_station_indices:
        assert not mask[s]
    # Depot recharge blocked when all customers are served (auto-close handles return).
    assert not mask[inst.depot_index]
