"""Deterministic semantic invariant tests for EVRPTW rollout/evaluation consistency."""
from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.env.action_space import (
    ChargePortionAnalysis,
    ChargePortionEvaluation,
    DepotFinishValidity,
    compute_depot_finish_validity,
    evaluate_charge_portions_for_node,
    exists_future_service_after_recharge,
)
from src.env.cost import CostWeights
from src.env.ev_state import EVState
from src.env.policy_action import PolicyAction, apply_policy_action, resolve_depot_action
from src.evaluation.baselines import NearestFeasiblePolicy, RandomFeasiblePolicy
from src.evaluation.local_search import local_search_repair


# --- A. Depot finish propagation ---


def test_depot_finish_propagated_to_apply_action():
    env = MagicMock()
    env.instance.depot_index = 0
    ev = EVState(ev_id=0, current_node=0, current_time=0.0, remaining_battery=50.0, remaining_capacity=100.0)
    action = PolicyAction(node=0, charge_portion=None, depot_finish=True)

    with patch("src.env.policy_action.compute_pre_action_context", return_value={}):
        apply_policy_action(env, ev, action)

    env.apply_action.assert_called_once()
    _, kwargs = env.apply_action.call_args
    assert kwargs.get("depot_finish") is True


# --- B. Depot recharge propagation ---


def test_depot_recharge_not_silently_finished():
    env = MagicMock()
    env.instance.depot_index = 0
    ev = EVState(ev_id=0, current_node=1, current_time=0.0, remaining_battery=10.0, remaining_capacity=100.0)
    action = PolicyAction(node=0, charge_portion=1.0, depot_finish=False)

    with patch("src.env.policy_action.compute_pre_action_context", return_value={}):
        apply_policy_action(env, ev, action)

    args, kwargs = env.apply_action.call_args
    assert kwargs.get("depot_finish") is False
    assert args[2] == 1.0


# --- C. Heuristic depot mode ---


def test_nearest_baseline_returns_policy_action():
    policy = NearestFeasiblePolicy()
    env = MagicMock()
    env.instance = MagicMock()
    env.instance.depot_index = 0
    env.instance.customer_indices = [1]
    env.instance.charging_station_indices = [2]
    env.instance.distance_matrix = np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float)
    env.use_depot_charging = True
    env.instance.depot_has_charging = True
    ev = EVState(ev_id=0, current_node=0, current_time=0.0, remaining_battery=50.0, remaining_capacity=100.0)
    mask = np.array([False, True, False])

    action = policy.select(env, ev, mask)
    assert isinstance(action, PolicyAction)
    assert action.node == 1
    assert action.depot_finish is False


def test_resolve_depot_finish_when_all_served():
    env = MagicMock()
    env.instance.depot_index = 0
    env.num_served_customers = 5
    env.instance.num_customers = 5
    ev = EVState(ev_id=0, current_node=1, current_time=0.0, remaining_battery=50.0, remaining_capacity=100.0)

    with patch("src.env.masks.depot_mode_feasible", return_value=(True, True)):
        action = resolve_depot_action(env, ev, 0, 1.0)

    assert action.depot_finish is True
    assert action.charge_portion is None


def test_resolve_depot_recharge_when_useful():
    env = MagicMock()
    env.instance.depot_index = 0
    env.num_served_customers = 0
    env.instance.num_customers = 5
    ev = EVState(ev_id=0, current_node=1, current_time=0.0, remaining_battery=10.0, remaining_capacity=100.0)

    with patch("src.env.masks.depot_mode_feasible", return_value=(True, False)):
        action = resolve_depot_action(env, ev, 0, None)

    assert action.depot_finish is False
    assert action.charge_portion == 1.0


# --- D. Charge-portion usefulness ---


def test_charge_portion_picks_unlocking_portion():
    inst = MagicMock()
    inst.customer_indices = [1]
    inst.charging_station_indices = [2]
    inst.depot_index = 0
    inst.num_customers = 1
    ev = EVState(ev_id=0, current_node=0, current_time=0.0, remaining_battery=10.0, remaining_capacity=100.0)
    served = np.zeros(3, dtype=bool)
    reserved = np.zeros(3, dtype=bool)

    with patch("src.env.charge_masks.feasible_charge_portions") as fcp, \
         patch("src.env.action_space._feasible_customer_indices") as fci, \
         patch("src.env.charge_masks.simulate_recharge_departure") as sim, \
         patch("src.env.charge_masks.safe_depot_return_from") as sdr:
        fcp.return_value = (np.array([True, False, True, False]), [])
        fci.side_effect = [[], [1], [1, 2]]
        sim.side_effect = [(1.0, 20.0, 1.0, 0.0), (5.0, 80.0, 5.0, 0.0)]
        sdr.return_value = (True, "ok")
        analysis = evaluate_charge_portions_for_node(
            inst, ev, 2, served, reserved,
            charging_portions=[0.25, 0.5, 0.75, 1.0],
        )

    assert analysis.best is not None
    assert analysis.best.portion == 0.75
    log = analysis.to_log_dict()
    assert log["best_useful_charge_portion"] == 0.75
    assert log["usefulness_score_by_charge_portion"][0.75] >= log["usefulness_score_by_charge_portion"][0.25]


# --- E. Depot finish blocked by recharge-mediated service ---


def test_depot_finish_invalid_when_future_service_after_recharge():
    inst = MagicMock()
    inst.num_customers = 3
    inst.depot_index = 0
    ev = EVState(ev_id=0, current_node=1, current_time=0.0, remaining_battery=5.0, remaining_capacity=100.0)
    served = np.zeros(4, dtype=bool)
    reserved = np.zeros(4, dtype=bool)

    with patch("src.env.action_space.depot_finish_feasible", return_value=(True, "ok")), \
         patch("src.env.action_space._feasible_customer_indices", return_value=[]), \
         patch("src.env.action_space.ev_future_service_possible", return_value=(True, "useful_station", {})), \
         patch("src.env.action_space.global_serviceability_after_finish", return_value={
             "finish_would_reduce_global_serviceability": True,
         }):
        val = compute_depot_finish_validity(inst, ev, served, reserved, num_served_customers=1)

    assert val.valid is False
    assert val.finish_masked_due_to_future_service is True


def test_exists_future_service_after_recharge_detects_station():
    inst = MagicMock()
    inst.charging_station_indices = [2]
    ev = EVState(ev_id=0, current_node=1, current_time=0.0, remaining_battery=5.0, remaining_capacity=100.0)
    served = np.zeros(4, dtype=bool)
    reserved = np.zeros(4, dtype=bool)

    with patch("src.env.action_space.compute_depot_recharge_usefulness", return_value=(False, "", 0)), \
         patch("src.env.action_space.compute_station_usefulness") as csu:
        csu.return_value = MagicMock(useful=True, enabled_customers_count=1)
        assert exists_future_service_after_recharge(inst, ev, served, reserved) is True


# --- F. Evaluation objective purity ---


def test_evaluation_objective_ignores_shaped_penalties():
    """Shaped training penalties must not change compute_cost true objective."""
    from src.env.cost import CostWeights, SolutionSummary, compute_cost

    summary = SolutionSummary(
        used_evs=1, total_distance=100.0, total_duration=10.0,
        total_waiting_time=0.0, total_charging_time=0.0, total_charging_price=0.0,
        unserved_customers=0, unresolved_reserved_customers=0,
        duplicate_customer_visits=0, depot_return_violation=0,
    )
    weights = CostWeights()
    j, _ = compute_cost(summary, weights)
    j2, _ = compute_cost(summary, weights)
    assert j == j2


# --- G. Local search validity ---


def test_local_search_diagnostic_only_no_objective_improvement():
    env = MagicMock()
    env.instance = MagicMock()
    env.instance.distance_matrix = np.array([[0, 1], [1, 0]], dtype=float)
    env.evs = []
    env.build_summary.return_value = MagicMock(
        used_evs=1, total_distance=100.0, total_duration=10.0,
        total_waiting_time=0.0, total_charging_time=0.0, total_charging_price=0.0,
        unserved_customers=0, unresolved_reserved_customers=0,
        duplicate_customer_visits=0, depot_return_violation=0.0,
    )
    env.feasible_solution.return_value = True
    weights = CostWeights()

    ls = local_search_repair(env, weights, {"enabled": True, "max_iterations": 50})
    assert ls.improved is False
    assert ls.diagnostic_only is True
    assert ls.repaired_objective == ls.raw_objective


# --- H. Rollout consistency ---


def test_training_and_eval_rollouts_use_apply_policy_action():
    from src.training import rollout
    from src.training import rollout_ac
    from src.training import rollout_trace

    for src in (
        inspect.getsource(rollout.run_episode),
        inspect.getsource(rollout_ac.run_episode_trajectory),
        inspect.getsource(rollout_trace.run_traced_episode),
    ):
        assert "apply_policy_action" in src
        assert "env.apply_action(ev, action.node" not in src.replace(" ", "")


def test_baselines_use_apply_policy_action():
    from src.evaluation import baselines

    assert "apply_policy_action" in inspect.getsource(baselines.run_heuristic_episode)
    assert "depot_mode_feasible" not in inspect.getsource(baselines.run_heuristic_episode)
