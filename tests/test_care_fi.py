"""Focused validity tests for CARE-FI fleet intensification."""
from __future__ import annotations

from types import SimpleNamespace

from src.search.deadline import SearchDeadline
from src.search.hanif_fleet_intensify import (
    _try_eliminate_route,
    rank_constraint_aware_routes,
)
from src.search.score import SearchScore


def _instance():
    nodes = [
        SimpleNamespace(
            is_customer=False, is_charging_station=False, x=0.0, y=0.0,
            due_time=100.0, demand=0.0,
        ),
        SimpleNamespace(
            is_customer=True, is_charging_station=False, x=1.0, y=0.0,
            due_time=20.0, demand=1.0,
        ),
        SimpleNamespace(
            is_customer=True, is_charging_station=False, x=2.0, y=0.0,
            due_time=40.0, demand=1.0,
        ),
        SimpleNamespace(
            is_customer=True, is_charging_station=False, x=10.0, y=0.0,
            due_time=80.0, demand=1.0,
        ),
    ]
    return SimpleNamespace(
        nodes=nodes,
        depot_index=0,
        vehicle=SimpleNamespace(load_capacity=10.0),
    )


def test_constraint_aware_ranking_is_deterministic(monkeypatch):
    instance = _instance()
    routes = [
        ([0, 1, 0], [None] * 3),
        ([0, 2, 0], [None] * 3),
        ([0, 3, 0], [None] * 3),
    ]

    def fake_replay(_instance, nodes, _labels, config=None):
        events = [
            SimpleNamespace(node=n, service_start=5.0)
            for n in nodes
            if _instance.nodes[n].is_customer
        ]
        return SimpleNamespace(route_events=events)

    monkeypatch.setattr(
        "src.search.hanif_fleet_intensify.replay_route", fake_replay,
    )
    a = rank_constraint_aware_routes(instance, routes, top_k=3, seed=0)
    b = rank_constraint_aware_routes(instance, routes, top_k=3, seed=0)
    assert [x.route_index for x in a] == [x.route_index for x in b]
    assert all(x.n_customers == 1 for x in a)
    assert all(x.mean_tw_slack >= 0.0 for x in a)
    assert all(0.0 <= x.load_utilization <= 1.0 for x in a)


def test_elimination_accepts_only_exact_customer_complete_solution(monkeypatch):
    instance = _instance()
    routes = [
        ([0, 1, 0], [None] * 3),
        ([0, 2, 3, 0], [None] * 4),
    ]

    def direct_recover(_instance, base, ejected, **_kwargs):
        nodes, labels = base[0]
        base[0] = (nodes[:-1] + list(ejected) + [0], labels[:-1] + [None] * (len(ejected) + 1))
        return base, []

    score = SearchScore(0, 0, 1, 10.0, 0.0, 20.0, 12.0)
    replay = SimpleNamespace(feasible=True, score=score)
    monkeypatch.setattr(
        "src.search.fast_construct._recover_unserved_fast", direct_recover,
    )
    monkeypatch.setattr(
        "src.search.hanif_fleet_intensify.regret_insertion",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.search.hanif_fleet_intensify.replay_solution",
        lambda *args, **kwargs: replay,
    )

    reduced, reduced_score, meta = _try_eliminate_route(
        instance,
        routes,
        0,
        config={},
        weights=None,
        target_m=1,
        deadline=SearchDeadline.from_seconds(2.0),
    )
    assert reduced is not None
    assert reduced_score == score
    assert len(reduced) == 1
    assert {n for n in reduced[0][0] if instance.nodes[n].is_customer} == {1, 2, 3}
    assert meta["operator"] == "direct"


def test_elimination_rejects_failed_exact_replay(monkeypatch):
    instance = _instance()
    routes = [
        ([0, 1, 0], [None] * 3),
        ([0, 2, 3, 0], [None] * 4),
    ]

    monkeypatch.setattr(
        "src.search.fast_construct._recover_unserved_fast",
        lambda _instance, base, ejected, **kwargs: (base, []),
    )
    monkeypatch.setattr(
        "src.search.hanif_fleet_intensify.regret_insertion",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "src.search.hanif_fleet_intensify.replay_solution",
        lambda *args, **kwargs: SimpleNamespace(feasible=False, score=None),
    )
    reduced, score, _meta = _try_eliminate_route(
        instance,
        routes,
        0,
        config={},
        weights=None,
        target_m=1,
        deadline=SearchDeadline.from_seconds(2.0),
    )
    assert reduced is None
    assert score is None


def test_elimination_honors_expired_deadline():
    reduced, score, meta = _try_eliminate_route(
        _instance(),
        [],
        0,
        config={},
        weights=None,
        target_m=1,
        deadline=SearchDeadline.from_seconds(1e-9),
    )
    assert reduced is None
    assert score is None
    assert meta["operator"] == "none"
