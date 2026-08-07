"""Core search validity tests (fixtures skip when Schneider assets absent)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.search.charging_dp import optimize_charging_plan
from src.search.fleet_search import capacity_lower_bound, compute_fleet_bounds, solve_with_fleet_search
from src.search.incremental import capacity_precheck, energy_precheck
from src.search.score import BenchmarkScore, SearchScore
from src.search.route_pool import RoutePool
from src.search.set_partitioning import solve_set_partitioning

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"
FIX_BASELINE = ROOT / "outputs" / "baselines" / "fixed_fleet_expert_repair_alns"


@pytest.fixture(scope="module")
def small_instance():
    if not SMALL.is_dir():
        pytest.skip("external Schneider fixture c101C5 absent")
    from src.data.instance_loader import load_instance
    return load_instance(SMALL)


def test_benchmark_score_lex_order():
    # distance before charging before duration before J
    a = BenchmarkScore(0, 0, 5, 100.0, 10.0, 1000.0, 5000.0)
    b = BenchmarkScore(0, 0, 5, 101.0, 1.0, 1.0, 1.0)
    assert a.dominates(b)
    s = SearchScore.from_benchmark(a)
    assert s.as_tuple()[3] == 100.0
    assert s.as_tuple()[4] == 10.0  # charging
    assert s.as_tuple()[5] == 1000.0  # duration


def test_fixed_fleet_baseline_manifest_present():
    man = FIX_BASELINE / "manifest.json"
    if not man.is_file():
        pytest.skip("fixed fleet baseline absent")
    import json
    m = json.loads(man.read_text(encoding="utf-8"))
    assert m["method_name"] == "FixedFleet-Expert+Repair+ALNS"
    assert m["fleet_minimization_performed"] is False
    assert m["fleet_upper_bound_m"] == 40


def test_capacity_and_energy_precheck(small_instance):
    custs = [i for i, n in enumerate(small_instance.nodes) if n.is_customer]
    assert capacity_precheck(small_instance, [], custs[0]) is True
    assert energy_precheck(small_instance, small_instance.depot_index, custs[0]) in (True, False)


def test_charging_dp_smoke(small_instance):
    custs = [i for i, n in enumerate(small_instance.nodes) if n.is_customer][:2]
    plan = optimize_charging_plan(small_instance, custs, config={"problem": {"charging_portions": [1.0]}})
    assert plan.nodes
    assert isinstance(plan.feasible, bool)


def test_fleet_bounds_and_search_smoke(small_instance):
    lo, hi = compute_fleet_bounds(small_instance, {"fleet_search": {"lexicographic": True}})
    assert 1 <= lo <= hi
    assert capacity_lower_bound(small_instance) >= 1
    cfg = {
        "fleet_search": {
            "lexicographic": True,
            "upper_bound_retries": 1,
            "retries_per_fleet": 1,
            "failures_before_stop": 1,
            "seeds": [0],
        },
        "hybrid_search": {
            "construction": {"expert_variants_list": ["hybrid"]},
            "alns": {"iterations": 2, "time_limit_seconds": 1},
        },
        "problem": {"charging_portions": [1.0], "use_horizon_check": True, "use_depot_charging": True},
    }
    res = solve_with_fleet_search(small_instance, config=cfg, budget_seconds=15.0)
    assert res.fleet_minimization_performed is True
    assert res.m_star >= lo
    assert res.attempts


def test_route_pool_and_partition(small_instance):
    from src.env.cost import CostWeights
    from src.search.portfolio import best_portfolio_solution

    best = best_portfolio_solution(
        small_instance, 3, variants=["hybrid"], seeds=[0], weights=CostWeights(),
        config={"problem": {"charging_portions": [1.0]}},
    )
    pool = RoutePool()
    pool.add_from_solution(
        small_instance, best.routes, score=best.score,
        source_method="test", source_fleet_size=3,
    )
    assert len(pool) >= 1
    routes, score, meta = solve_set_partitioning(
        small_instance, pool, config={"problem": {"charging_portions": [1.0]}}, m=3,
    )
    assert "backend" in meta


def test_no_target_leakage_in_candidate_ranker_source():
    """Candidate ranker must not feed score_vectors as model inputs at inference guide path."""
    src = (ROOT / "src" / "search" / "neural_guide.py").read_text(encoding="utf-8")
    assert "score_vectors" not in src
    assert "score_candidates" in src
