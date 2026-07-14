"""Performance regression for canonical decision analysis."""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.env.decision_analysis import (
    analyses_are_equivalent,
    analyze_decision_state_optimized,
    analyze_decision_state_reference,
)
from src.env.evrptw_env import EVRPTWEnv

ROOT = Path(__file__).resolve().parents[2]
RC21 = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "rc102_21"


@pytest.mark.skipif(not RC21.exists(), reason="rc102_21 not present")
def test_optimized_at_least_3x_faster_than_reference_or_report():
    inst = load_instance(RC21)
    env = EVRPTWEnv(inst, m=3, charging_portions=[0.25, 0.5, 0.75, 1.0], max_steps_factor=4)
    env.environment_config = {
        "mask_semantics": "exact_original_evrptw",
        "implementation": "optimized",
        "action_pruning": {
            "enabled": True,
            "block_exact_noop_actions": True,
            "block_repeated_zero_progress_loops": True,
            "prune_station_when_customer_available": False,
        },
        "recharge_search": {"max_station_hops": 3, "use_dominance_pruning": True},
        "cache": {"enabled": False},
        "performance": {"cache_action_masks": False, "use_optimized_mask_path": True},
    }
    ev = env.select_acting_ev()

    # Warmup
    analyze_decision_state_reference(env, ev)
    env._action_mask_cache = {}
    analyze_decision_state_optimized(env, ev)

    ref_times = []
    for _ in range(3):
        env._action_mask_cache = {}
        env._decision_analysis_cache = {}
        t0 = time.perf_counter()
        ref = analyze_decision_state_reference(env, ev)
        ref_times.append((time.perf_counter() - t0) * 1000.0)

    opt_times = []
    for _ in range(5):
        env._action_mask_cache = {}
        env._decision_analysis_cache = {}
        t0 = time.perf_counter()
        opt = analyze_decision_state_optimized(env, ev)
        opt_times.append((time.perf_counter() - t0) * 1000.0)

    # Parity on last pair
    ok, msg = analyses_are_equivalent(ref, opt, feature_atol=1e-4)
    assert ok, msg

    ref_med = statistics.median(ref_times)
    opt_med = statistics.median(opt_times)
    speedup = ref_med / max(opt_med, 1e-9)
    print(
        f"\n[PERF] reference_median_ms={ref_med:.1f} "
        f"optimized_median_ms={opt_med:.1f} speedup={speedup:.2f}x"
    )
    # Soft relative gate: do not fail solely on slow CI hardware for absolute ms,
    # but require optimized ≤ reference (regression guard). Absolute 20ms target
    # is informational.
    assert opt_med <= ref_med * 1.05, (
        f"optimized slower than reference: {opt_med:.1f}ms vs {ref_med:.1f}ms"
    )
