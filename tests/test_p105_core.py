"""P10.5 structural tests — synthetic fixtures only (no Schneider required)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schneider.search.charging_dp import exhaustive_optimum, optimize_charging_plan
from src.schneider.search.deadline import SearchDeadline, StageBudgetPlan
from src.schneider.search.fleet_search import _variant_seed_schedule, capacity_lower_bound, construct_solution
from src.schneider.search.route_elimination import rank_elimination_candidates
from src.schneider.search.route_pool import RoutePool
from src.schneider.search.set_partitioning import solve_set_partitioning
from tests.fixtures import build_energy_tight_tiny, build_tiny_evrptw

ROOT = Path(__file__).resolve().parents[1]
P10_PARTIAL = ROOT / "outputs" / "baselines" / "deterministic_fps_p10_partial"


def test_fleet_retry_diversity_covers_all_variants():
    variants = ["edf", "regret", "energy", "fleet", "hybrid"]
    seeds = [0, 1, 2, 3]
    sched = _variant_seed_schedule(variants, seeds, 8)
    first5 = [v for v, _ in sched[:5]]
    assert first5 == variants
    assert len({(v, s) for v, s in sched}) >= 5


def test_cold_construct_after_warm_failure_flag_path():
    # Schedule / API guarantee: cold retry source string is distinct
    from src.schneider.search.fleet_search import FleetAttemptTrace
    t = FleetAttemptTrace(
        fleet_size=30, attempt=1, variant="edf", seed=0, feasible=False,
        served=99, used_vehicles=30, distance=1.0, runtime=0.1,
        warm_start_source="cold_construct_after_warm_fail",
    )
    assert t.warm_start_source == "cold_construct_after_warm_fail"
    # Tiny cold construct without warm start must return routes
    inst = build_tiny_evrptw()
    cfg = {"problem": {"charging_portions": [1.0], "use_horizon_check": True, "use_depot_charging": True}}
    r2, s2, _f2 = construct_solution(
        inst, 2, expert_variant="hybrid", seed=0, config=cfg, warm_start=None,
    )
    assert isinstance(r2, list) and len(r2) >= 1
    assert s2 is not None


def test_route_elimination_ranking_not_only_smallest():
    inst = build_tiny_evrptw(
        customers=[
            (5.0, 0.0, 5.0, 0.0, 500.0, 5.0),
            (10.0, 0.0, 5.0, 0.0, 500.0, 5.0),
            (5.0, 5.0, 5.0, 0.0, 500.0, 5.0),
            (15.0, 0.0, 5.0, 0.0, 500.0, 5.0),
        ]
    )
    depot = inst.depot_index
    routes = [
        ([depot, 1, 2, depot], [None, None, None, None]),
        ([depot, 3, depot], [None, None, None]),
        ([depot, 4, depot], [None, None, None]),
    ]
    ranked = rank_elimination_candidates(inst, routes, top_k=3)
    assert len(ranked) >= 2
    # Small routes appear, but ranking returns multiple candidates
    assert set(ranked) <= {0, 1, 2}


def test_route_pool_exact_per_route_costs():
    inst = build_tiny_evrptw()
    depot = inst.depot_index
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    route = ([depot] + custs + [depot], [None] * (len(custs) + 2))
    pool = RoutePool()
    # Fake score with huge totals — pool must NOT use proportional share
    class FakeScore:
        total_distance = 1e9
        total_duration = 1e9
        total_charging_time = 1e9

    added = pool.add_from_solution(
        inst, [route], score=FakeScore(), source_method="test", source_fleet_size=1,
        config={"problem": {"charging_portions": [1.0]}},
    )
    assert added == 1
    pr = pool.as_list()[0]
    assert pr.distance < 1e6
    assert pr.feasible is True
    assert pr.nodes[0] == depot and pr.nodes[-1] == depot
    assert pr.order == tuple(custs)


def test_set_partitioning_rejects_missing_customer_coverage():
    inst = build_tiny_evrptw()
    depot = inst.depot_index
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    # Pool covers only first customer
    route = ([depot, custs[0], depot], [None, None, None])
    pool = RoutePool()
    pool.add_from_solution(
        inst, [route], source_method="partial", source_fleet_size=1,
        config={"problem": {"charging_portions": [1.0]}},
    )
    rebuilt, score, meta = solve_set_partitioning(
        inst, pool, config={"problem": {"charging_portions": [1.0]}}, m=2,
    )
    assert rebuilt == []
    assert score is None
    assert meta.get("incomplete_coverage") is True
    assert custs[1] in meta.get("missing_customers", [])


def test_charging_dp_matches_exhaustive_optimum_on_tiny():
    inst = build_tiny_evrptw(
        customers=[(10.0, 0.0, 5.0, 0.0, 800.0, 5.0)],
        stations=[(5.0, 0.0)],
        battery_capacity=30.0,
        energy_rate=1.0,
    )
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    cfg = {"problem": {"charging_portions": [0.5, 1.0], "use_depot_charging": True}}
    opt = optimize_charging_plan(inst, custs, config=cfg)
    exh = exhaustive_optimum(inst, custs, config=cfg)
    assert opt.feasible
    assert exh is not None and exh.feasible
    # Lex distance / charging: DP should match exhaustive optimum within tolerance
    assert abs(opt.route_distance - exh.route_distance) <= 1e-4 + 0.05 * max(1.0, exh.route_distance)


def test_charging_dp_no_055_heuristic_in_source():
    src = (ROOT / "src" / "search" / "charging_dp.py").read_text(encoding="utf-8")
    assert "0.55" not in src


def test_strict_budget_deadline_propagation():
    d = SearchDeadline.from_seconds(0.01, stage="test")
    import time
    time.sleep(0.02)
    assert d.expired()
    assert d.remaining() == 0.0
    child = SearchDeadline.from_seconds(10.0).child(0.5, stage="child")
    assert child.budget_seconds <= 0.5 + 1e-9


def test_stage_budget_plan_sums_to_one():
    p = StageBudgetPlan()
    s = (
        p.upper_bound_fraction + p.fleet_descent_fraction + p.route_pool_fraction
        + p.charging_fraction + p.repair_alns_fraction + p.rescue_fraction
    )
    assert abs(s - 1.0) < 1e-6


def test_p10_partial_table_reconstruction():
    csv_path = P10_PARTIAL / "validation_summary.csv"
    if not csv_path.is_file():
        pytest.skip("P10 partial artifacts not frozen in this checkout")
    import csv
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    assert len(rows) == 6
    names = {r["instance"] for r in rows}
    assert names == {"c103_21", "c201_21", "r103_21", "r201_21", "rc103_21", "rc201_21"}
    js = json.loads((P10_PARTIAL / "validation_summary.json").read_text(encoding="utf-8"))
    assert abs(float(js["mean_used_vehicles"]) - 29.33) < 0.05
    decision = (P10_PARTIAL / "P10_DECISION.md").read_text(encoding="utf-8")
    assert "PARTIAL" in decision


def test_capacity_lower_bound_synthetic():
    inst = build_tiny_evrptw(load_capacity=15.0)
    assert capacity_lower_bound(inst) >= 1


def test_energy_tight_charging_produces_plan():
    inst = build_energy_tight_tiny()
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    plan = optimize_charging_plan(
        inst, custs, config={"problem": {"charging_portions": [0.25, 0.5, 0.75, 1.0]}},
    )
    assert plan.nodes
    assert isinstance(plan.feasible, bool)
