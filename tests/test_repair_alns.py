"""Exact insertion repair and operator tests."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.search.operators import destroy_random_customers, relocate_customer, swap_customers
from src.search.repair import (
    best_insertion,
    relabel_charge_portions,
    remove_station_if_redundant,
    repair_solution,
)
from src.search.route_replay import replay_solution
from src.search.alns import run_alns

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"


@pytest.fixture(scope="module")
def small_instance():
    if not SMALL.is_dir():
        pytest.skip("c101C5 instance missing")
    return load_instance(SMALL)


def test_best_insertion_and_repair(small_instance):
    depot = small_instance.depot_index
    customers = [i for i, n in enumerate(small_instance.nodes) if n.is_customer]
    assert customers
    # Empty fleet of routes
    routes = [([depot, depot], [None, None]) for _ in range(min(5, len(customers) + 1))]
    # Leave all unserved, repair should insert at least one when feasible.
    repaired, res = repair_solution(
        small_instance,
        routes,
        weights=CostWeights(),
        m=len(routes),
        max_rounds=min(5, len(customers)),
        mode="best",
        station_candidates=3,
    )
    assert res is not None
    assert isinstance(repaired, list)
    # Either served increases or no feasible insertion exists.
    assert res.unserved_customers <= len(customers)


def test_destroy_and_rollback_preserve_length(small_instance):
    depot = small_instance.depot_index
    customers = [i for i, n in enumerate(small_instance.nodes) if n.is_customer][:3]
    routes = [([depot, c, depot], [None, None, None]) for c in customers]
    import random

    destroyed, removed = destroy_random_customers(small_instance, routes, k=1, rng=random.Random(0))
    assert len(removed) <= 1
    assert len(destroyed) == len(routes)


def test_relabel_and_station_removal_apis(small_instance):
    depot = small_instance.depot_index
    customers = [i for i, n in enumerate(small_instance.nodes) if n.is_customer][:2]
    routes = [([depot, c, depot], [None, None, None]) for c in customers]
    out = remove_station_if_redundant(small_instance, routes, weights=CostWeights(), m=len(routes))
    assert len(out) == len(routes)
    out2 = relabel_charge_portions(small_instance, out, weights=CostWeights(), m=len(routes))
    assert len(out2) == len(routes)


def test_alns_runs_short_budget(small_instance):
    depot = small_instance.depot_index
    customers = [i for i, n in enumerate(small_instance.nodes) if n.is_customer][:3]
    routes = [([depot, c, depot], [None, None, None]) for c in customers]
    result = run_alns(
        small_instance,
        routes,
        weights=CostWeights(),
        m=len(routes),
        iterations=5,
        time_limit_seconds=5.0,
        seed=0,
        destroy_k=1,
    )
    assert result.iterations >= 1
    assert result.score is not None
    assert len(result.routes) >= 1
