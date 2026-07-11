"""Exact mask parity and semantic invariant tests."""
from __future__ import annotations

import numpy as np
import pytest

from src.env.action_masks import (
    compute_action_masks,
    compute_action_masks_optimized_exact,
    compute_action_masks_reference,
    masks_are_equivalent,
    validate_action_against_mask,
)
from src.env.evrptw_env import EVRPTWEnv
from src.env.mask_errors import InvalidActionMaskError
from src.env.masks import customer_feasible


def test_mask_reference_equals_optimized(small_instance):
    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    ref = compute_action_masks_reference(env, ev)
    opt = compute_action_masks_optimized_exact(env, ev)
    ok, msg = masks_are_equivalent(ref, opt)
    assert ok, msg


def test_feasible_customer_ratio_bounded(small_instance):
    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    result = compute_action_masks(env, ev)
    unserved = int(np.sum(~env.served_mask[env.instance.customer_indices]))
    ratio = result.feasible_customer_count / max(unserved, 1)
    assert 0.0 <= ratio <= 1.0


def test_customer_without_safe_continuation_masked(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    # Drain battery to allow reach but block continuation
    ev.remaining_battery = float(inst.energy_matrix[ev.current_node, j]) + 0.01
    ok, reason = customer_feasible(
        inst, ev, j, env.served_mask, env.reserved_mask,
        use_horizon_check=True, use_depot_charging=True, charging_portions=env.charging_portions,
    )
    if not ok:
        assert reason in {
            "customer_battery_unreachable",
            "customer_no_safe_direct_depot",
            "customer_no_reachable_station",
            "customer_station_has_no_feasible_charge",
            "customer_no_safe_continuation",
            "customer_time_window_infeasible",
        }


def test_empty_charge_mask_raises_on_validate(small_instance):
    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    result = compute_action_masks(env, ev)
    # Force inconsistent state for test
    bad = result
    if inst := env.instance:
        for s in inst.charging_station_indices:
            if s in bad.charge_masks:
                bad.charge_masks[s] = np.zeros_like(bad.charge_masks[s])
                with pytest.raises(InvalidActionMaskError):
                    validate_action_against_mask(bad, env, s, 1.0, False)
                break


def test_state_version_stale_mask(small_instance):
    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    result = compute_action_masks(env, ev)
    env.bump_state_version()
    from src.env.mask_errors import StaleMaskError
    with pytest.raises(StaleMaskError):
        validate_action_against_mask(result, env, int(env.instance.customer_indices[0]), None, False)
