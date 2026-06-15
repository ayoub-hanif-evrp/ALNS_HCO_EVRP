"""Tests for fleet-size search: range, hierarchical/scalar selection, fallback."""
from __future__ import annotations

from src.env.cost import CostWeights, SolutionSummary
from src.evaluation.baselines import NearestFeasiblePolicy, evaluate_baseline
from src.evaluation.fleet_search import (
    FleetCandidate,
    fleet_search,
    fleet_size_range,
    select_best,
)


def _cand(m, feasible, used_evs, dist, obj):
    summary = SolutionSummary(used_evs=used_evs, total_distance=dist)
    return FleetCandidate(m=m, feasible=feasible, objective=obj, summary=summary)


def test_fleet_range(small_instance):
    m_min, m_max = fleet_size_range(small_instance, extra_margin=5)
    assert m_min == small_instance.m_min
    assert m_max >= m_min
    assert m_max <= small_instance.num_customers


def test_hierarchical_prefers_fewer_evs():
    cands = [
        _cand(1, False, 1, 10.0, 99999.0),
        _cand(2, True, 3, 50.0, 3050.0),
        _cand(3, True, 2, 80.0, 2080.0),  # fewer EVs wins despite larger distance
        _cand(4, True, 2, 60.0, 2060.0),
    ]
    best = select_best(cands, "hierarchical")
    assert best.m == 4  # 2 EVs, and smaller distance than m=3


def test_scalar_prefers_min_objective():
    cands = [
        _cand(2, True, 3, 50.0, 3050.0),
        _cand(3, True, 2, 80.0, 2080.0),
    ]
    best = select_best(cands, "scalar")
    assert best.m == 3


def test_fallback_when_no_feasible():
    cands = [
        _cand(1, False, 1, 10.0, 50000.0),
        _cand(2, False, 2, 20.0, 30000.0),
    ]
    best = select_best(cands, "hierarchical")
    assert best.m == 2  # least-infeasible (min objective)


def test_fleet_search_integration_with_baseline(small_instance):
    weights = CostWeights()

    def runner(instance, m):
        j, summary, env = evaluate_baseline(
            instance, m, NearestFeasiblePolicy(), weights, use_horizon_check=True
        )
        return FleetCandidate(m=m, feasible=env.feasible_solution(), objective=j, summary=summary)

    best, candidates = fleet_search(small_instance, runner, extra_margin=6, selection_mode="hierarchical")
    assert len(candidates) >= 1
    # With enough vehicles the nearest baseline should find a feasible solution.
    assert any(c.feasible for c in candidates)
    assert best.feasible
