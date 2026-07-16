"""Parity tests: reference vs EV-state-first optimized decision analysis."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance_loader import load_instance
from src.env.decision_analysis import (
    analyses_are_equivalent,
    analyze_decision_state,
    get_analysis_call_counts,
    reset_analysis_call_counts,
)
from src.env.evrptw_env import EVRPTWEnv


def _find_instance(*names: str) -> Path | None:
    root = Path("dataset/Schneider/evrptw_instances")
    if not root.exists():
        return None
    for name in names:
        direct = root / name
        if direct.exists():
            return direct
        matches = list(root.rglob(f"{name}.txt")) + list(root.rglob(name))
        if matches:
            return matches[0]
    for pat in ("rc101_21*", "c101_21*", "r101_21*", "c101C5*", "rc101*"):
        matches = list(root.rglob(pat))
        if matches:
            return matches[0]
    return None


def _make_env(path: Path) -> EVRPTWEnv:
    inst = load_instance(path)
    env = EVRPTWEnv(
        inst,
        m=2,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
        use_depot_charging=True,
        use_horizon_check=True,
    )
    env.environment_config = {
        "mask_semantics": "exact_original_evrptw",
        "implementation": "optimized",
        "cache": {"enabled": True, "cache_by_state_version": True},
        "recharge_search": {"max_station_hops": 3, "use_dominance_pruning": True},
        "action_pruning": {"enabled": True, "prune_station_when_customer_available": False},
    }
    env.reset()
    return env


@pytest.fixture(scope="module")
def sample_env():
    path = _find_instance("c101C5", "rc101_21", "c101_21", "r101_21", "rc101", "c101")
    if path is None:
        pytest.skip("No Schneider instance found")
    return _make_env(path)


def test_reference_optimized_mask_parity(sample_env):
    env = sample_env
    ev = env.select_acting_ev()
    assert ev is not None
    reset_analysis_call_counts()
    ref = analyze_decision_state(env, ev, implementation="reference")
    env._decision_analysis_cache = {}
    env.last_decision_analysis = None
    env._global_serviceability_cache = None
    env._ev_analysis_cache = {}
    opt = analyze_decision_state(env, ev, implementation="optimized")
    ok, msg = analyses_are_equivalent(ref, opt)
    assert ok, msg
    assert np.array_equal(ref.hard_node_mask, opt.hard_node_mask)
    assert np.array_equal(ref.final_node_mask, opt.final_node_mask)
    assert np.array_equal(ref.depot_mode_mask, opt.depot_mode_mask)
    assert set(ref.final_charge_masks.keys()) == set(opt.final_charge_masks.keys())
    for k in ref.final_charge_masks:
        assert np.array_equal(ref.final_charge_masks[k], opt.final_charge_masks[k])


def test_ev_state_first_serviceability_matches_or_of_ev_masks(sample_env):
    env = sample_env
    env._decision_analysis_cache = {}
    env._ev_analysis_cache = {}
    env._global_serviceability_cache = None
    ev = env.select_acting_ev()
    opt = analyze_decision_state(env, ev, implementation="optimized")
    assert opt.active_ev_analysis is not None
    assert opt.ev_analyses
    cust = np.asarray(env.instance.customer_indices, dtype=np.int64)
    combined = np.zeros(cust.size, dtype=bool)
    for ea in opt.ev_analyses.values():
        combined |= np.asarray(ea.serviceable_customer_mask, dtype=bool)
    svc_ids = set(opt.global_serviceability.serviceable_customer_ids)
    expected = {int(j) for j in cust[combined]}
    assert svc_ids == expected


def test_analysis_assert_fresh(sample_env):
    env = sample_env
    ev = env.select_acting_ev()
    analysis = analyze_decision_state(env, ev, implementation="optimized")
    analysis.assert_fresh(env, ev)
    env.bump_state_version()
    with pytest.raises(RuntimeError, match="stale"):
        analysis.assert_fresh(env, ev)


def test_recharge_search_once_per_ev_not_per_customer(sample_env):
    env = sample_env
    ev = env.select_acting_ev()
    reset_analysis_call_counts()
    env._decision_analysis_cache = {}
    env._ev_analysis_cache = {}
    env._global_serviceability_cache = None
    analysis = analyze_decision_state(env, ev, implementation="optimized")
    counts = get_analysis_call_counts()
    n_active = sum(1 for e in env.evs if not e.finished)
    assert counts["recharge_search"] <= n_active + 1
    assert counts["global_serviceability"] == 1
    assert analysis.active_ev_analysis is not None
    remaining = int((~env.served_mask[env.instance.customer_indices]).sum())
    assert counts["recharge_search"] < max(remaining, 2)
