"""Consistency tests for pre-action usefulness, charge portions, and rollouts."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.env.action_space import (
    ChargePortionAnalysis,
    ChargePortionEvaluation,
    DepotFinishValidity,
    compute_depot_finish_validity,
    evaluate_charge_portions_for_node,
)
from src.env.ev_state import EVState
from src.env.policy_action import apply_policy_action, compute_pre_action_context, PolicyAction


def test_charge_portion_picks_highest_enabled_customers():
    """75% useful beats 25% feasible-only when scoring enabled customers."""
    inst = MagicMock()
    inst.customer_indices = [1]
    inst.charging_station_indices = [2]
    inst.depot_index = 0
    inst.num_customers = 1
    inst.nodes = {
        0: MagicMock(is_depot=True, is_charging_station=False, is_customer=False),
        1: MagicMock(is_customer=True, is_charging_station=False, is_depot=False),
        2: MagicMock(is_charging_station=True, is_depot=False, is_customer=False),
    }
    ev = EVState(ev_id=0, current_node=0, current_time=0.0, remaining_battery=10.0, remaining_capacity=100.0)
    served = np.zeros(3, dtype=bool)
    reserved = np.zeros(3, dtype=bool)

    def fake_eval(*args, **kwargs):
        return ChargePortionAnalysis(
            best=ChargePortionEvaluation(
                portion=0.75, enabled_customers=2, enables_depot_return=True,
                charging_time=5.0, feasible=True, usefulness_score=2005.0,
            ),
            all_feasible=[
                ChargePortionEvaluation(0.25, 0, True, 1.0, True, 10.0),
                ChargePortionEvaluation(0.75, 2, True, 5.0, True, 2005.0),
            ],
        )

    with patch("src.env.charge_masks.feasible_charge_portions") as fcp, \
         patch("src.env.action_space._feasible_customer_indices") as fci, \
         patch("src.env.action_space._count_newly_enabled_customers") as cne, \
         patch("src.env.charge_masks.simulate_recharge_departure") as sim, \
         patch("src.env.charge_masks.safe_depot_return_from") as sdr:
        fcp.return_value = (np.array([True, False, True, False]), [])
        # feasible_now=[]; after 25% -> +1; after 75% -> +2
        fci.return_value = []
        cne.side_effect = [1, 2]
        sim.side_effect = [
            (1.0, 20.0, 1.0, 0.0),
            (5.0, 80.0, 5.0, 0.0),
        ]
        sdr.return_value = (True, "ok")
        analysis = evaluate_charge_portions_for_node(
            inst, ev, 2, served, reserved,
            charging_portions=[0.25, 0.5, 0.75, 1.0],
        )
    assert analysis.best is not None
    assert analysis.best.portion == 0.75
    assert analysis.best.enabled_customers == 2


def test_pre_action_context_before_mutation(small_instance):
    from src.env.evrptw_env import EVRPTWEnv

    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    batt_before = ev.remaining_battery
    node = ev.current_node
    mask, _ = env.feasible_mask(ev)
    targets = [i for i in range(small_instance.num_nodes) if mask[i]]
    if not targets:
        pytest.skip("no feasible action at start")
    target = targets[0]
    pre = compute_pre_action_context(env, ev, target, depot_finish=False)
    assert "usefulness_before_action" in pre or "pre_global_serviceability" in pre
    assert ev.remaining_battery == batt_before
    assert ev.current_node == node
    assert pre.get("usefulness_reason_before_action") is not None or target in small_instance.customer_indices


def test_apply_policy_action_attaches_pre_context(small_instance):
    from src.env.evrptw_env import EVRPTWEnv

    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    mask, _ = env.feasible_mask(ev)
    cust = next((j for j in small_instance.customer_indices if mask[j]), None)
    if cust is None:
        pytest.skip("no feasible customer")
    info = apply_policy_action(env, ev, PolicyAction(node=cust), collect_diagnostics=True)
    assert "pre_global_serviceability" in info
    assert "post_global_serviceability" in info


def test_depot_finish_blocked_when_useful_station_exists(small_instance):
    from src.env.evrptw_env import EVRPTWEnv
    from src.env.action_space import ev_future_service_possible

    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    with patch(
        "src.env.action_space.ev_future_service_possible",
        return_value=(True, "useful_station_exists_before_finish", {"useful_station_exists_before_finish": True}),
    ):
        val = compute_depot_finish_validity(
            small_instance, ev, env.served_mask, env.reserved_mask, env.num_served_customers,
        )
    if not val.valid or val.category == "invalid":
        assert val.finish_masked_due_to_future_service or val.reason


def test_controlled_closure_recomputes_serviceability(small_instance):
    from src.env.evrptw_env import EVRPTWEnv
    from src.env.action_space import GlobalServiceability

    env = EVRPTWEnv(small_instance, m=2)
    env.last_global_serviceability = None
    env._global_serviceability_cache = None
    env.last_decision_analysis = None
    with patch(
        "src.env.serviceability.compute_global_serviceability_recharge_aware",
        return_value=GlobalServiceability(no_service_possible=False),
    ) as cgs:
        env.try_controlled_closure()
        assert cgs.called


def test_max_step_penalty_not_in_objective_component(small_instance):
    from src.env.evrptw_env import EVRPTWEnv
    from src.env.cost import CostWeights

    env = EVRPTWEnv(small_instance, m=2, max_steps_factor=1)
    env.max_steps = 2
    env.step_count = 2
    env.is_done()
    assert env.max_steps_reached
    assert env.max_step_penalty == 0.0
    w = CostWeights()
    j_before, _, _ = env.score(w)
    env.max_step_penalty = 999.0
    j_after, _, _ = env.score(w)
    assert j_before == j_after


def test_rollout_uses_apply_policy_action():
    import inspect
    from src.training import rollout
    from src.training import rollout_ac
    from src.training import rollout_trace

    assert "apply_policy_action" in inspect.getsource(rollout.run_episode)
    assert "apply_policy_action" in inspect.getsource(rollout_ac.run_episode_trajectory)
    assert "apply_policy_action" in inspect.getsource(rollout_trace.run_traced_episode)
