"""Tests for dense training reward shaping."""
from __future__ import annotations

from src.env.cost import CostWeights, SolutionSummary
from src.env.reward import CostSnapshot, StepInfo
from src.training.reward_shaping import (
    StepContext,
    TerminalPenaltySchedule,
    TrainingRewardCalculator,
    TrainingRewardConfig,
)


def _weights() -> CostWeights:
    return CostWeights(
        lambda_vehicle=1000.0,
        lambda_distance=1.0,
        lambda_duration=0.1,
        lambda_charging=0.5,
        lambda_charging_price=1.0,
        lambda_waiting=0.1,
        penalty_unserved=10000.0,
        penalty_duplicate=10000.0,
        penalty_late=5000.0,
        penalty_battery=10000.0,
        penalty_capacity=10000.0,
        penalty_depot=10000.0,
        penalty_infeasible=100000.0,
        penalty_reserved=10000.0,
    )


def test_served_bonus_on_new_customer():
    w = _weights()
    cfg = TrainingRewardConfig(reward_scale=1000.0, served_bonus_coef=0.5)
    calc = TrainingRewardCalculator(w, cfg, num_customers=100, cycle_id=1)
    before = CostSnapshot(num_served_customers=10)
    after = CostSnapshot(num_served_customers=11)
    step = StepInfo(
        ev_id=0, selected_node=5, node_type="customer", previous_node=0,
        reward=-0.01, raw_reward=-0.01,
    )
    out = calc.shape_step(step, before=before, after=after, node_type="customer", customer_served=True)
    expected_bonus = 0.5 * 10000.0 / 1000.0
    assert abs(out.reward - (-0.01 + expected_bonus)) < 1e-6
    assert calc.stats.new_customers_served == 1


def test_terminal_penalty_schedule():
    w = _weights()
    sched = TerminalPenaltySchedule(enabled=True, start_coef=0.25, end_coef=1.0, warmup_cycles=20)
    cfg = TrainingRewardConfig(reward_scale=1000.0, terminal_schedule=sched)
    calc = TrainingRewardCalculator(w, cfg, num_customers=100, cycle_id=1)
    summary = SolutionSummary(
        used_evs=1, total_distance=100.0, total_duration=100.0,
        total_charging_time=0.0, total_charging_price=0.0, total_waiting_time=0.0,
        unserved_customers=10, duplicate_customer_visits=0, total_lateness=0.0,
        total_battery_violation=0.0, total_capacity_violation=0.0,
        depot_return_violation=0, unresolved_reserved_customers=0, infeasible_flag=1,
    )
    info = calc.terminal_adjustment(CostSnapshot(), summary)
    assert calc.stats.terminal_unserved_penalty_raw == 100000.0
    assert calc.stats.terminal_unserved_penalty_scaled == 25000.0
    assert info.extra["terminal_penalty_coef"] == 0.25


def test_early_depot_penalty():
    w = _weights()
    cfg = TrainingRewardConfig(reward_scale=1000.0, early_depot_penalty_coef=1.0)
    calc = TrainingRewardCalculator(w, cfg, num_customers=100, cycle_id=1)
    before = CostSnapshot(num_served_customers=50)
    step = StepInfo(
        ev_id=0, selected_node=0, node_type="route_close", previous_node=1,
        reward=0.0, raw_reward=0.0,
    )
    after = CostSnapshot(num_served_customers=50)
    out = calc.shape_step(step, before=before, after=after, node_type="route_close", customer_served=False)
    assert calc.stats.early_depot_returns == 1
    assert out.reward < 0.0


def test_newly_expired_customer_penalty():
    w = _weights()
    cfg = TrainingRewardConfig(
        reward_scale=1000.0,
        use_newly_expired_customer_penalty=True,
        newly_expired_customer_penalty=1.0,
    )
    calc = TrainingRewardCalculator(w, cfg, num_customers=100, cycle_id=1)
    ctx = StepContext(expired_before=2, expired_after=5)
    step = StepInfo(
        ev_id=0, selected_node=1, node_type="station", previous_node=0,
        reward=-0.01, raw_reward=-0.01,
    )
    out = calc.shape_step(
        step, before=CostSnapshot(), after=CostSnapshot(),
        node_type="station", customer_served=False, ctx=ctx,
    )
    assert calc.stats.newly_expired_customers == 3
    assert out.reward < -0.01
    assert out.extra["newly_expired_customers"] == 3.0
