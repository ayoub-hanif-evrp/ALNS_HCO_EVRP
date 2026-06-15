"""Tests for environment transitions: reservation lifecycle, stations, depot, async."""
from __future__ import annotations

import numpy as np

from src.env.evrptw_env import EVRPTWEnv


def _env(instance, m=3):
    return EVRPTWEnv(instance, m=m, use_horizon_check=True)


def test_reservation_lifecycle(small_instance):
    inst = small_instance
    env = _env(inst)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    env.apply_action(ev, j)
    # After an atomic customer visit, the customer is served, not reserved.
    assert env.served_mask[j]
    assert not env.reserved_mask[j]
    assert env.num_served_customers == 1
    assert ev.route_nodes[-1] == j
    assert ev.current_time > 0.0
    assert ev.remaining_capacity == inst.vehicle.load_capacity - inst.nodes[j].demand


def test_served_customer_cannot_be_selected_again(small_instance):
    inst = small_instance
    env = _env(inst)
    j = inst.customer_indices[0]
    env.apply_action(env.evs[0], j)
    mask, reasons = env.feasible_mask(env.evs[1])
    assert not mask[j]
    assert reasons[j] == "already_served"


def test_station_reusable_and_charging_math(small_instance):
    inst = small_instance
    vp = inst.vehicle
    env = _env(inst)
    ev = env.evs[0]
    s = inst.charging_station_indices[0]
    # Drop battery to half so charging has an effect.
    ev.remaining_battery = vp.battery_capacity * 0.5
    before_time = ev.current_time
    env.apply_action(ev, s, charge_portion=1.0)
    # Battery increased toward capacity.
    assert ev.remaining_battery > vp.battery_capacity * 0.5 - 1e-6
    assert ev.remaining_battery <= vp.battery_capacity + 1e-6
    assert ev.total_charging_time > 0.0
    assert ev.current_time >= before_time
    assert len(ev.charging_actions) == 1
    # Station can be visited again by another EV (not in served mask).
    assert not env.served_mask[s]
    ev2 = env.evs[1]
    mask, _ = env.feasible_mask(ev2)
    assert mask[s]


def test_partial_charging_portion(small_instance):
    inst = small_instance
    vp = inst.vehicle
    env = _env(inst)
    ev = env.evs[0]
    s = inst.charging_station_indices[0]
    ev.remaining_battery = vp.battery_capacity * 0.2
    energy_to_s = inst.energy_matrix[ev.current_node, s]
    battery_after_arrival = ev.remaining_battery - energy_to_s
    missing = vp.battery_capacity - battery_after_arrival
    env.apply_action(ev, s, charge_portion=0.5)
    expected = battery_after_arrival + 0.5 * missing
    assert abs(ev.remaining_battery - expected) < 1e-4


def test_depot_return_marks_finished(small_instance):
    inst = small_instance
    env = _env(inst)
    ev = env.evs[0]
    # Move to a customer first so depot is not a self-loop.
    j = inst.customer_indices[0]
    env.apply_action(ev, j)
    env.apply_action(ev, inst.depot_index)
    assert ev.finished
    assert ev.route_nodes[-1] == inst.depot_index


def test_event_driven_smallest_time_acts_next(small_instance):
    inst = small_instance
    env = _env(inst, m=3)
    # All EVs start at time 0; tie-break by route length then id -> EV0 first.
    first = env.select_acting_ev()
    assert first.ev_id == 0
    # EV0 acts and advances its clock.
    j = inst.customer_indices[0]
    env.apply_action(first, j)
    assert first.current_time > 0.0
    # Now EV1 (still at time 0) must be selected next, not EV0.
    nxt = env.select_acting_ev()
    assert nxt.ev_id == 1
    assert nxt.current_time == 0.0


def test_initial_dispatch_is_sequential(small_instance):
    inst = small_instance
    m = 3
    env = _env(inst, m=m)
    dispatched = []
    for _ in range(m):
        ev = env.select_acting_ev()
        dispatched.append(ev.ev_id)
        mask, _ = env.feasible_mask(ev)
        # pick first feasible customer
        choice = next(c for c in inst.customer_indices if mask[c])
        env.apply_action(ev, choice)
    # The first m decisions go to EVs 0,1,2 in order (sequential dispatch).
    assert dispatched == [0, 1, 2]
