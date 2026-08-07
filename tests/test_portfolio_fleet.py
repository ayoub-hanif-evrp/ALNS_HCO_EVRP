"""Portfolio + fleet + state-feature tests for TG-FPS."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.search.fleet import descending_fleet_search
from src.search.portfolio import best_portfolio_solution, run_expert_portfolio
from src.search.state_features import compute_construction_pressure

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"


@pytest.fixture(scope="module")
def small_instance():
    if not SMALL.is_dir():
        pytest.skip("c101C5 instance missing")
    return load_instance(SMALL)


def test_construction_pressure_no_name_branch(small_instance):
    env = EVRPTWEnv(small_instance, m=4, charging_portions=[1.0], hard_time_windows=True)
    p = compute_construction_pressure(env)
    assert p.n_unserved == small_instance.num_customers
    assert 0.0 <= p.fleet_pressure <= 1.0
    # Properties are bools derived from state
    assert isinstance(p.prefer_earliest_due, bool)
    assert isinstance(p.prefer_fresh_ev_release, bool)


def test_portfolio_returns_sorted_scores(small_instance):
    results = run_expert_portfolio(
        small_instance,
        m=4,
        variants=["edf", "hybrid"],
        seeds=[0],
        weights=CostWeights(),
    )
    assert len(results) >= 2
    tuples = [r.score.as_tuple() for r in results]
    assert tuples == sorted(tuples)


def test_descending_fleet_search_smoke(small_instance):
    def construct(inst, m):
        best = best_portfolio_solution(
            inst, m, variants=["hybrid"], seeds=[0], weights=CostWeights(),
        )
        return best.routes

    res = descending_fleet_search(
        small_instance,
        construct,
        weights=CostWeights(),
        diversified_retries=1,
        m_hi=5,
        m_lo=3,
    )
    assert res.m_star >= 3
    assert res.best_routes
    assert res.attempts
