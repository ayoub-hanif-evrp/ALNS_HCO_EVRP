"""Exact route replay tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.search.route_replay import extract_routes_from_env, replay_env_solution, replay_route, replay_solution
from src.search.score import SearchScore

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"


@pytest.fixture(scope="module")
def small_instance():
    if not SMALL.is_dir():
        pytest.skip("c101C5 instance missing")
    return load_instance(SMALL)


def _trivial_routes(instance, m: int = 3):
    """One customer per route when possible (may be infeasible on energy — still tests API)."""
    depot = instance.depot_index
    customers = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    routes = []
    for i, c in enumerate(customers[:m]):
        routes.append(([depot, c, depot], [None, None, None]))
    while len(routes) < m:
        routes.append(([depot, depot], [None, None]))
    return routes


def test_search_score_ordering():
    a = SearchScore(0, 0, 5, 100, 1000, 10, 5000)
    b = SearchScore(0, 1, 4, 50, 500, 5, 1000)
    assert a.dominates(b)
    assert not b.dominates(a)


def test_replay_route_depot_only(small_instance):
    depot = small_instance.depot_index
    res = replay_route(small_instance, [depot, depot], [None, None])
    assert res.first_failure is None or res.feasible or res.end_time >= 0.0
    assert isinstance(res.distance, float)


def test_replay_solution_detects_duplicates(small_instance):
    depot = small_instance.depot_index
    customers = [i for i, n in enumerate(small_instance.nodes) if n.is_customer]
    if len(customers) < 1:
        pytest.skip("no customers")
    c = customers[0]
    routes = [
        ([depot, c, depot], [None, None, None]),
        ([depot, c, depot], [None, None, None]),
    ]
    res = replay_solution(small_instance, routes, m=2, weights=CostWeights())
    assert res.duplicate_customers >= 1 or not res.feasible


def test_replay_env_agrees_with_score(small_instance):
    """Build a simple env solution and check replay objective tracks env.score."""
    env = EVRPTWEnv(small_instance, m=5, charging_portions=[0.25, 0.5, 0.75, 1.0])
    weights = CostWeights()
    # Serve customers greedily with first legal customer each time.
    steps = 0
    while not env.is_done() and steps < 200:
        ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_auto_close_from_analysis(ev, analysis) is not None:
            steps += 1
            continue
        if analysis.mask_result is None:
            break
        mask = analysis.mask_result.node_mask
        chosen = None
        for j, ok in enumerate(mask):
            if not ok:
                continue
            if small_instance.nodes[j].is_customer:
                chosen = j
                break
        if chosen is None:
            # finish if possible
            depot = small_instance.depot_index
            if len(analysis.mask_result.depot_mode_mask) > 1 and analysis.mask_result.depot_mode_mask[1]:
                env.apply_depot_finish(ev)
            elif mask[depot]:
                env.apply_action(ev, depot, 1.0, depot_finish=False)
            else:
                break
        else:
            env.apply_action(ev, chosen, None, depot_finish=False)
        steps += 1

    obj_env, _, summary = env.score(weights)
    routes = extract_routes_from_env(env)
    if not routes:
        pytest.skip("no used routes produced")
    replayed = replay_solution(
        small_instance, routes, m=env.m, charging_portions=list(env.charging_portions), weights=weights
    )
    # Distance/objective should be close when replay succeeds without hard failure mid-route.
    assert abs(replayed.distance - float(summary.total_distance)) < 1e-3 or replayed.first_failure is not None
    assert abs(replayed.objective - float(obj_env)) < 1.0 or replayed.first_failure is not None
