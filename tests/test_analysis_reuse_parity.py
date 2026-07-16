"""Parity: analysis-first rollout reuses one DecisionStateAnalysis (no mask drift)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.instance_loader import load_instance
from src.env.analysis_counters import (
    reset_all_counters,
    get_cycle_counters,
    enter_analyze,
    exit_analyze,
)
from src.env.decision_analysis import analyze_decision_state, analyses_are_equivalent
from src.env.evrptw_env import EVRPTWEnv


DATASET = Path("dataset/Schneider/evrptw_instances")


def _find(name: str) -> Path | None:
    if not DATASET.exists():
        return None
    hits = list(DATASET.rglob(f"{name}.txt")) + list(DATASET.rglob(name))
    return hits[0] if hits else None


@pytest.fixture
def small_env():
    path = _find("c101_21") or _find("c101C5") or _find("c101")
    if path is None:
        pytest.skip("no Schneider instance available")
    inst = load_instance(path)
    env = EVRPTWEnv(
        inst, m=3,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
        use_depot_charging=True,
        use_horizon_check=True,
    )
    env.environment_config = {
        "implementation": "optimized",
        "cache": {"enabled": True, "cache_by_state_version": True},
        "action_pruning": {"enabled": True},
    }
    env.reset()
    return env


def test_one_analysis_per_decision_no_duplicate_gs(small_env):
    env = small_env
    ev = env.select_acting_ev()
    assert ev is not None
    reset_all_counters()
    analysis = env.analyze_decision(ev)
    c = get_cycle_counters()
    assert c.get("analyze_decision_state_calls", 0) == 1
    assert c.get("global_serviceability_calls", 0) <= 1

    # Closure / auto-close must not re-analyze or recompute GS.
    before_gs = c.get("global_serviceability_calls", 0)
    before_an = c.get("analyze_decision_state_calls", 0)
    env.try_controlled_closure_from_analysis(ev, analysis)
    env.try_auto_close_from_analysis(ev, analysis)
    c2 = get_cycle_counters()
    assert c2.get("analyze_decision_state_calls", 0) == before_an
    assert c2.get("global_serviceability_calls", 0) == before_gs


def test_cached_analysis_masks_stable(small_env):
    env = small_env
    ev = env.select_acting_ev()
    a1 = analyze_decision_state(env, ev, implementation="optimized")
    a2 = analyze_decision_state(env, ev, implementation="optimized")
    ok, msg = analyses_are_equivalent(a1, a2)
    assert ok, msg
    assert np.array_equal(a1.final_node_mask, a2.final_node_mask)
    assert np.array_equal(a1.depot_mode_mask, a2.depot_mode_mask)


def test_outside_analyze_charge_portion_counter():
    """charge_portion_recompute_calls only increments outside analyze."""
    from src.env.analysis_counters import bump_outside_analyze, reset_all_counters, get_cycle_counters

    reset_all_counters()
    bump_outside_analyze("charge_portion_recompute_calls")
    assert get_cycle_counters().get("charge_portion_recompute_calls", 0) == 1
    enter_analyze()
    try:
        bump_outside_analyze("charge_portion_recompute_calls")
    finally:
        exit_analyze()
    assert get_cycle_counters().get("charge_portion_recompute_calls", 0) == 1
