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
    assert reasons2[j] == "customer_reserved"


def test_energy_cannot_reach_is_masked(small_instance):
    env = _fresh_env(small_instance)
    ev = env.evs[0]
    # Drain the battery so almost nothing is reachable.
    ev.remaining_battery = 0.0
    mask, reasons = env.feasible_mask(ev)
    for j in small_instance.customer_indices:
        assert not mask[j]
        assert reasons[j] in {
            "customer_battery_unreachable",
            "customer_no_safe_direct_depot",
            "customer_no_reachable_station",
            "customer_station_has_no_feasible_charge",
            "customer_no_safe_continuation",
            "customer_infeasible_after_service_battery",
        }


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
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, True,
                                   env.use_depot_charging, env.charging_portions)
    assert not ok
    assert reason in {
        "customer_no_safe_direct_depot",
        "customer_no_reachable_station",
        "customer_station_has_no_feasible_charge",
        "customer_no_safe_continuation",
        "customer_infeasible_after_service_battery",
    }


def test_capacity_exceeded_is_masked(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    ev.remaining_capacity = inst.nodes[j].demand - 1.0
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, True)
    assert not ok
    assert reason == "customer_capacity_infeasible"


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
    assert reason == "customer_time_window_infeasible"


def test_time_window_late_is_masked(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    # Push current time well past the due date.
    ev.current_time = inst.nodes[j].due_time + 1000.0
    ok, reason = customer_feasible(inst, ev, j, env.served_mask, env.reserved_mask, use_horizon_check=False)
    assert not ok
    assert reason == "customer_time_window_infeasible"


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
    ok, reason = customer_feasible(inst, ev, target, env.served_mask, env.reserved_mask, use_horizon_check=False,
                                   use_depot_charging=env.use_depot_charging, charging_portions=env.charging_portions)
    # Early arrival is allowed (waiting), not infeasible due to the window.
    assert ok or reason in {
        "customer_battery_unreachable",
        "customer_no_safe_direct_depot",
        "customer_no_reachable_station",
        "customer_station_has_no_feasible_charge",
        "customer_no_safe_continuation",
    }


def test_station_self_loop_masked(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    s = inst.charging_station_indices[0]
    ev.current_node = s
    mask, reasons = env.feasible_mask(ev)
    assert not mask[s]
    assert reasons[s] in {"station_loop_blocked", "station_no_useful_continuation", "station_no_progress"}


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
    # At route start on depot, recharge is a self-loop and finish is not allowed yet.
    assert not mask[inst.depot_index]
    assert reasons[inst.depot_index] in {"depot_self_loop", "depot_finish_at_start"}


def test_all_served_stations_allowed_for_depot_return(small_instance):
    """When all customers are served, useful stations must remain for safe return."""
    inst = small_instance
    env = _fresh_env(inst)
    for c in inst.customer_indices:
        env.served_mask[c] = True
    env.num_served_customers = inst.num_customers
    ev = env.evs[0]
    ev.current_node = inst.customer_indices[0]
    # Low battery may require a station before depot finish — mask must not blanket-block stations.
    ev.remaining_battery = max(1.0, inst.vehicle.battery_capacity * 0.05)
    mask, reasons = env.feasible_mask(ev)
    depot = inst.depot_index
    ok_finish, _ = __import__("src.env.masks", fromlist=["depot_finish_feasible"]).depot_finish_feasible(
        inst, ev, use_horizon_check=True,
    )
    if not ok_finish:
        # At least one station should be allowed if it can enable depot return, or finish stays blocked safely.
        any_station = any(mask[s] for s in inst.charging_station_indices)
        assert any_station or not mask[depot], reasons.get(depot, "")
    else:
        assert mask[depot] or any(mask[s] for s in inst.charging_station_indices)


def test_must_close_only_when_mask_empty(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    mask, _ = env.feasible_mask(ev)
    assert env.should_close_route(ev) == (not mask.any())
    from src.env.masks import must_close_route

    assert must_close_route(
        inst, ev, env.served_mask, env.reserved_mask,
        use_depot_charging=env.use_depot_charging,
        use_horizon_check=env.use_horizon_check,
        charging_portions=env.charging_portions,
        num_served_customers=env.num_served_customers,
    ) == (not mask.any())
