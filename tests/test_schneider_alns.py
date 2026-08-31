"""Smoke tests for the Schneider EVRPTW ALNS / solve path."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.schneider.data.instance_loader import load_instance
from src.schneider.search.schneider_alns import destroy_random, destroy_shaw, destroy_worst
from src.schneider.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
C203 = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c203_21"


@pytest.mark.skipif(not C203.is_dir(), reason="Schneider c203 not converted")
def test_destroy_removes_customers():
    inst = load_instance(C203)
    depot = inst.depot_index
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer][:8]
    routes = [([depot] + custs[:4] + [depot], [None] * 6), ([depot] + custs[4:] + [depot], [None] * 6)]
    rng = __import__("random").Random(0)
    trial, removed = destroy_random(inst, routes, 3, rng)
    assert 1 <= len(removed) <= 3
    left = [n for nodes, _ in trial for n in nodes if inst.nodes[n].is_customer]
    assert set(removed).isdisjoint(left)
    trial2, rem2 = destroy_worst(inst, routes, 2, rng)
    assert rem2
    trial3, rem3 = destroy_shaw(inst, routes, 3, rng)
    assert rem3


def test_incumbent_loader_picks_a_file():
    from src.schneider.evaluation.schneider_incumbent import load_best_incumbent

    routes, meta = load_best_incumbent("c204_21", "pr")
    if routes is None:
        pytest.skip("no archived c204 routes")
    assert meta["feasible"]
    assert meta["used_k"] <= 5
    assert routes


@pytest.mark.skipif(not C203.is_dir(), reason="Schneider c203 not converted")
def test_solve_schneider_short_budget():
    from src.schneider.env.cost import CostWeights
    from src.schneider.evaluation.schneider_incumbent import load_best_incumbent
    from src.schneider.search.schneider_solve import solve_schneider

    inst = load_instance(C203)
    cfg = load_config(str(ROOT / "configs" / "schneider_fps.yaml"))
    cfg["dataset"] = {"name": "schneider"}
    warm, _ = load_best_incumbent("c203_21", "pr")
    result = solve_schneider(
        inst, config=cfg, weights=CostWeights.from_eval_config(cfg),
        budget_seconds=20.0, warm_start_routes=warm,
    )
    assert result.feasible
    assert result.m_star >= 1
    if warm:
        assert result.m_star <= 4
    assert result.runtime_seconds < 90
