"""Tests for the cost / objective function."""
from __future__ import annotations

from src.env.cost import CostWeights, SolutionSummary, compute_cost, is_feasible


def test_feasible_solution_has_no_penalties():
    summary = SolutionSummary(
        used_evs=3,
        total_distance=100.0,
        total_duration=500.0,
        total_charging_time=20.0,
        total_waiting_time=10.0,
    )
    weights = CostWeights()
    j, breakdown = compute_cost(summary, weights)
    assert is_feasible(summary)
    assert breakdown["pen_unserved"] == 0.0
    # J equals weighted objective terms only.
    expected = (
        weights.lambda_vehicle * 3
        + weights.lambda_distance * 100.0
        + weights.lambda_duration * 500.0
        + weights.lambda_charging * 20.0
        + weights.lambda_waiting * 10.0
    )
    assert abs(j - expected) < 1e-6


def test_unserved_customers_penalized():
    summary = SolutionSummary(used_evs=1, unserved_customers=2)
    weights = CostWeights()
    j, breakdown = compute_cost(summary, weights)
    assert not is_feasible(summary)
    assert breakdown["pen_unserved"] == weights.penalty_unserved * 2


def test_battery_and_capacity_violations_penalized():
    summary = SolutionSummary(
        used_evs=1,
        total_battery_violation=5.0,
        total_capacity_violation=3.0,
    )
    weights = CostWeights()
    _, breakdown = compute_cost(summary, weights)
    assert breakdown["pen_battery"] == weights.penalty_battery * 5.0
    assert breakdown["pen_capacity"] == weights.penalty_capacity * 3.0
    assert not is_feasible(summary)


def test_reward_is_negative_cost():
    summary = SolutionSummary(used_evs=2, total_distance=50.0)
    weights = CostWeights()
    j, _ = compute_cost(summary, weights)
    reward = -j
    assert reward < 0


def test_weights_from_config():
    cfg = {"lambda_vehicle": 5.0, "penalty_unserved": 42.0, "unknown_key": 1.0}
    weights = CostWeights.from_config(cfg)
    assert weights.lambda_vehicle == 5.0
    assert weights.penalty_unserved == 42.0
