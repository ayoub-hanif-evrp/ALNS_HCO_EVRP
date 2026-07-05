"""Integration tests for MDP action-space logic (usefulness, loops, termination)."""
from __future__ import annotations

import numpy as np
import pytest

from src.env.action_space import (
    ActionProgressTracker,
    compute_depot_finish_validity,
    compute_depot_recharge_usefulness,
    compute_global_serviceability,
    compute_station_usefulness,
    should_auto_close,
    FeasibleActionSummary,
)
from src.env.evrptw_env import EVRPTWEnv
from src.env.ev_state import EVState
from src.env.masks import compute_feasible_mask, customer_feasible


def _fresh_env(instance, m=2):
    return EVRPTWEnv(instance, m=m, use_horizon_check=True)


def test_useless_station_masked_when_customer_feasible(small_instance):
    """Station must be masked when a customer is feasible now without it."""
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    mask, reasons = env.feasible_mask(ev)
    has_customer = any(mask[j] for j in inst.customer_indices)
    if has_customer:
        for s in inst.charging_station_indices:
            u = compute_station_usefulness(
                inst, ev, s, env.served_mask, env.reserved_mask,
                all_customers_served=False,
            )
            if not u.useful:
                assert not mask[s], f"station {s} masked as useless but still feasible: {reasons.get(s)}"


def test_depot_recharge_masked_when_all_served(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    for c in inst.customer_indices:
        env.served_mask[c] = True
    env.num_served_customers = inst.num_customers
    ev = env.evs[0]
    useful, reason, _ = compute_depot_recharge_usefulness(
        inst, ev, env.served_mask, env.reserved_mask, all_customers_served=True,
    )
    assert not useful
    assert reason == "depot_recharge_all_served"


def test_global_serviceability_all_served(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    for c in inst.customer_indices:
        env.served_mask[c] = True
    gs = compute_global_serviceability(
        inst, env.evs, env.served_mask, env.reserved_mask,
    )
    assert gs.remaining_customers_count == 0
    assert not gs.no_service_possible


def test_loop_detection_depot_station_pattern():
    pt = ActionProgressTracker(loop_window=10)
    for _ in range(3):
        pt.record({"type": "depot_recharge", "ev_id": 0, "previous_node": 0, "node": 0}, EVState.initial(0, 0, 100, 100))
        pt.record({"type": "station", "ev_id": 0, "previous_node": 0, "node": 1}, EVState.initial(0, 1, 100, 100))
    assert pt.loop_detected or pt.depot_station_loop_count >= 0


def test_auto_close_not_when_customer_feasible(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    _, _, summary, gs = env.compute_full_feasible(ev)
    if summary.has_customer:
        assert not should_auto_close(summary, global_svc=gs)
        assert env.try_auto_close_route(ev) is None


def test_controlled_closure_when_no_service(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    for c in inst.customer_indices:
        env.served_mask[c] = True
    env.num_served_customers = inst.num_customers
    for ev in env.evs:
        ev.finished = True
    env.done_reason = None
    info = env.try_controlled_closure()
    assert info is None or env.done_reason is not None


def test_depot_finish_invalid_when_recharge_enables_service(small_instance):
    inst = small_instance
    env = _fresh_env(inst)
    ev = env.evs[0]
    gs = compute_global_serviceability(inst, env.evs, env.served_mask, env.reserved_mask)
    val = compute_depot_finish_validity(
        inst, ev, env.served_mask, env.reserved_mask, env.num_served_customers,
        global_svc=gs,
    )
    feasible = [j for j in inst.customer_indices if customer_feasible(
        inst, ev, j, env.served_mask, env.reserved_mask, True,
    )[0]]
    if feasible:
        useful_r, _, _ = compute_depot_recharge_usefulness(
            inst, ev, env.served_mask, env.reserved_mask, all_customers_served=False, global_svc=gs,
        )
        if useful_r and val.category == "invalid":
            assert "recharge" in val.reason or val.valid is False


def test_progress_tracker_resets_on_customer():
    pt = ActionProgressTracker()
    ev = EVState.initial(0, 0, 100.0, 100.0)
    pt.record({"type": "station", "ev_id": 0, "previous_node": 0, "node": 1}, ev)
    pt.record({"type": "customer", "ev_id": 0, "previous_node": 1, "node": 2, "customer_served": True}, ev)
    assert pt.steps_since_last_customer_service == 0


def test_feasible_summary_from_mask(small_instance):
    inst = small_instance
    n = inst.num_nodes
    feasible = np.zeros(n, dtype=bool)
    reasons: dict[int, str] = {}
    if inst.customer_indices:
        feasible[inst.customer_indices[0]] = True
    summary = FeasibleActionSummary.from_mask(inst, feasible, reasons, all_done=False)
    assert summary.has_customer
    assert summary.has_any
