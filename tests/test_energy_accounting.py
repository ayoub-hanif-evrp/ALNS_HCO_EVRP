"""Tests for post-hoc energy accounting (measurement-only; no algorithm changes)."""
from __future__ import annotations

import copy

import pytest

from src.common.energy_accounting import (
    BALANCE_TOL,
    account_route_energy,
    account_solution_energy,
    attach_energy_reporting,
)
from tests.fixtures.synthetic import build_tiny_evrptw


def _full_labels(instance, nodes, alpha: float = 1.0):
    labs = [None] * len(nodes)
    for i, n in enumerate(nodes):
        if instance.nodes[int(n)].is_charging_station:
            labs[i] = float(alpha)
        elif (
            instance.nodes[int(n)].is_depot
            and 0 < i < len(nodes) - 1
            and getattr(instance, "depot_has_charging", False)
        ):
            labs[i] = float(alpha)
    return labs


def test_energy_balance_identity_no_charge():
    inst = build_tiny_evrptw(battery_capacity=100.0, energy_rate=1.0, depot_has_charging=False)
    depot = int(inst.depot_index)
    # Direct depot → C1 → depot; propulsion = 2 * dist(depot, C1)
    c1 = int(inst.customer_indices[0])
    nodes = [depot, c1, depot]
    labels = [None, None, None]
    acc = account_route_energy(inst, nodes, labels)
    e01 = float(inst.energy_matrix[depot, c1])
    e10 = float(inst.energy_matrix[c1, depot])
    assert acc.propulsion_energy == pytest.approx(e01 + e10)
    assert acc.enroute_purchased_energy == pytest.approx(0.0)
    assert acc.residual_battery_at_depot == pytest.approx(100.0 - (e01 + e10))
    assert abs(acc.energy_balance_residual) <= BALANCE_TOL
    assert acc.energy_balance_ok
    assert acc.depot_replenishment_energy == pytest.approx(acc.propulsion_energy)


def test_energy_balance_with_station_charge():
    inst = build_tiny_evrptw(battery_capacity=50.0, energy_rate=1.0, depot_has_charging=False)
    depot = int(inst.depot_index)
    c1 = int(inst.customer_indices[0])
    c2 = int(inst.customer_indices[1])
    s0 = int(inst.charging_station_indices[0])
    nodes = [depot, c1, s0, c2, depot]
    labels = _full_labels(inst, nodes, alpha=1.0)
    acc = account_route_energy(inst, nodes, labels)
    lhs = acc.initial_battery + acc.enroute_purchased_energy - acc.propulsion_energy
    assert lhs == pytest.approx(acc.residual_battery_at_depot, abs=BALANCE_TOL)
    assert acc.energy_balance_ok
    assert acc.charging_visits >= 1
    assert acc.mean_energy_per_charging_visit == pytest.approx(
        acc.enroute_purchased_energy / acc.charging_visits
    )
    assert acc.depot_replenishment_energy == pytest.approx(
        acc.initial_battery - acc.residual_battery_at_depot
    )


def test_solution_aggregate_matches_sum_of_routes():
    inst = build_tiny_evrptw(battery_capacity=80.0, depot_has_charging=False)
    depot = int(inst.depot_index)
    c1, c2 = [int(i) for i in inst.customer_indices[:2]]
    s0 = int(inst.charging_station_indices[0])
    r1 = ([depot, c1, depot], [None, None, None])
    r2_nodes = [depot, c2, s0, depot]
    r2 = (r2_nodes, _full_labels(inst, r2_nodes, alpha=0.5))
    sol = account_solution_energy(inst, [r1, r2])
    a1 = account_route_energy(inst, r1[0], r1[1], route_id=0)
    a2 = account_route_energy(inst, r2[0], r2[1], route_id=1)
    assert sol.propulsion_energy == pytest.approx(a1.propulsion_energy + a2.propulsion_energy)
    assert sol.enroute_purchased_energy == pytest.approx(
        a1.enroute_purchased_energy + a2.enroute_purchased_energy
    )
    assert sol.residual_battery_at_depot == pytest.approx(
        a1.residual_battery_at_depot + a2.residual_battery_at_depot
    )
    assert sol.energy_balance_ok


def test_purchased_energy_matches_hanif_replay_charged_energy():
    from src.hanif.replay import replay_route

    inst = build_tiny_evrptw(battery_capacity=60.0, depot_has_charging=False)
    depot = int(inst.depot_index)
    c1, c2 = [int(i) for i in inst.customer_indices[:2]]
    s0 = int(inst.charging_station_indices[0])
    nodes = [depot, c1, s0, c2, depot]
    labels = _full_labels(inst, nodes, alpha=0.75)
    plan = replay_route(inst, nodes, labels)
    acc = account_route_energy(inst, nodes, labels)
    assert plan.feasible
    assert acc.enroute_purchased_energy == pytest.approx(plan.charged_energy, abs=1e-9)
    assert acc.charging_visits == plan.charging_visits


def test_attach_energy_reporting_is_pure_merge():
    inst = build_tiny_evrptw()
    depot = int(inst.depot_index)
    c1 = int(inst.customer_indices[0])
    routes = [([depot, c1, depot], [None, None, None])]
    row = {"distance": 1.0, "method": "unit"}
    before = copy.deepcopy(row)
    attach_energy_reporting(row, inst, routes)
    assert before["distance"] == row["distance"]
    assert "propulsion_energy" in row
    assert row["energy_balance_ok"] is True


def test_reporting_does_not_change_hco_or_replay_outputs():
    """Smoke: measuring energy after HCO leaves plan metrics unchanged."""
    from src.alns_hco.hco import hco
    from src.hanif.replay import replay_solution

    inst = build_tiny_evrptw(battery_capacity=100.0)
    custs = [int(i) for i in inst.customer_indices]
    plan = hco(inst, custs, mode="detour", candidates_per_criterion=3)
    if not plan.feasible:
        pytest.skip("HCO infeasible on synthetic tiny instance")
    routes = [(list(plan.nodes), list(plan.labels))]
    m1 = replay_solution(inst, routes)
    energy = account_solution_energy(inst, routes)
    m2 = replay_solution(inst, routes)
    assert m1.distance == m2.distance
    assert m1.charged_energy == m2.charged_energy
    assert m1.charging_expense == m2.charging_expense
    assert m1.used_k == m2.used_k
    assert energy.energy_balance_ok
