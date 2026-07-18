"""Gate 1-3 style tests: compile, imports, optimized integration, prefilter."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"


@pytest.mark.skipif(not SMALL.exists(), reason="c101C5 missing")
def test_core_imports_and_compile():
    import src.training.ac_train  # noqa: F401
    import src.training.rollout_ac  # noqa: F401
    import src.env.action_space  # noqa: F401
    import src.models.policy  # noqa: F401
    import src.models.critic  # noqa: F401
    from src.env.decision_analysis import analyze_decision_state  # noqa: F401


@pytest.mark.skipif(not SMALL.exists(), reason="c101C5 missing")
def test_rollout_uses_optimized_analyzer():
    from src.data.instance_loader import load_instance
    from src.env.decision_analysis import (
        get_analysis_call_counts,
        reset_analysis_call_counts,
    )
    from src.env.evrptw_env import EVRPTWEnv

    inst = load_instance(SMALL)
    env = EVRPTWEnv(inst, m=2, charging_portions=[0.25, 0.5, 0.75, 1.0])
    env.environment_config = {
        "mask_semantics": "exact_original_evrptw",
        "implementation": "optimized",
        "cache": {"enabled": True, "cache_by_state_version": True},
        "action_pruning": {
            "enabled": True,
            "block_exact_noop_actions": True,
            "block_repeated_zero_progress_loops": True,
            "prune_station_when_customer_available": False,
        },
        "recharge_search": {"max_station_hops": 3, "use_dominance_pruning": True},
    }
    reset_analysis_call_counts()
    analysis = env.analyze_decision(env.select_acting_ev())
    counts = get_analysis_call_counts()
    assert counts["analyze_decision_state_optimized"] >= 1
    assert counts["analyze_decision_state_reference"] == 0
    assert analysis.implementation == "optimized"
    assert analysis.dynamic_candidate_features is not None
    assert analysis.prefilter_stats.total_customers == inst.num_customers


@pytest.mark.skipif(not SMALL.exists(), reason="c101C5 missing")
def test_prefilter_skips_impossible_customers():
    from src.data.instance_loader import load_instance
    from src.env.decision_analysis import vectorized_basic_customer_prefilter
    from src.env.evrptw_env import EVRPTWEnv

    inst = load_instance(SMALL)
    env = EVRPTWEnv(inst, m=1, charging_portions=[0.25, 0.5, 0.75, 1.0])
    ev = env.evs[0]
    ev.remaining_battery = 0.0
    basic, cust, reject, stats = vectorized_basic_customer_prefilter(
        inst, ev, env.served_mask, env.reserved_mask,
    )
    assert stats.customers_rejected_by_vectorized_prefilter >= 1
    assert stats.customers_sent_to_continuation_analysis == int(basic.sum())
    assert any(r == "customer_battery_unreachable" for r in reject.values())


@pytest.mark.skipif(not SMALL.exists(), reason="c101C5 missing")
def test_policy_reuses_analysis_features():
    from src.data.instance_loader import load_instance
    from src.env.evrptw_env import EVRPTWEnv
    from src.models.policy import TransformerEVRPTWPolicy
    from src.utils.config import load_config

    cfg = load_config(str(ROOT / "configs" / "train.yaml"))
    inst = load_instance(SMALL)
    env = EVRPTWEnv(inst, m=2, charging_portions=cfg["problem"]["charging_portions"])
    env.environment_config = cfg["environment"]
    policy = TransformerEVRPTWPolicy(
        cfg["model"], charging_portions=cfg["problem"]["charging_portions"],
    )
    emb = policy.encode(inst, device="cpu")
    analysis = env.analyze_decision(env.select_acting_ev())
    analysis.dynamic_candidate_features = analysis.dynamic_candidate_features.copy()
    analysis.dynamic_candidate_features[:] = 0.123456
    state = policy.build_decision_state(
        env, env.evs[0], emb, analysis.mask_result, analysis=analysis,
    )
    assert float(state.dynamic_candidate_feats[0, -1].item()) == pytest.approx(
        float(analysis.dynamic_candidate_features[0, -1]), rel=0, abs=1e-5,
    )


@pytest.mark.skipif(not SMALL.exists(), reason="c101C5 missing")
def test_single_serviceability_per_state():
    from src.data.instance_loader import load_instance
    from src.env.decision_analysis import get_analysis_call_counts, reset_analysis_call_counts
    from src.env.evrptw_env import EVRPTWEnv

    inst = load_instance(SMALL)
    env = EVRPTWEnv(inst, m=2, charging_portions=[0.25, 0.5, 0.75, 1.0])
    env.environment_config = {
        "implementation": "optimized",
        "cache": {"enabled": True, "cache_by_state_version": True},
        "debug": {"assert_single_serviceability": True},
        "action_pruning": {"enabled": True, "prune_station_when_customer_available": False},
        "recharge_search": {"max_station_hops": 3},
    }
    reset_analysis_call_counts()
    env.analyze_decision(env.evs[0])
    before = get_analysis_call_counts()["global_serviceability"]
    env.get_global_serviceability(force=False)
    env.get_global_serviceability(force=False)
    after = get_analysis_call_counts()["global_serviceability"]
    assert after == before


def test_balanced_class_sampler_uniform_classes():
    from src.data.mixed_class_split import BalancedClassSampler
    import random

    train = {
        "C": [f"c{i}_21" for i in range(3)],
        "R": [f"r{i}_21" for i in range(30)],
        "RC": [f"rc{i}_21" for i in range(10)],
    }
    rng = random.Random(0)
    sampler = BalancedClassSampler(train, rng=rng)
    draws = [sampler.sample_name() for _ in range(3000)]

    def cls(n: str) -> str:
        if n.startswith("rc"):
            return "RC"
        if n.startswith("r"):
            return "R"
        return "C"

    counts = {"C": 0, "R": 0, "RC": 0}
    for n in draws:
        counts[cls(n)] += 1
    for k, v in counts.items():
        assert 700 <= v <= 1300, counts
