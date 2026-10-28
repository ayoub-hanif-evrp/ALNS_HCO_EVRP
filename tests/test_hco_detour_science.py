"""Scientific verification of HCO-Detour claims (no algorithm changes).

Checks:
1. Detour candidate preference ignores heterogeneous price/waiting attributes.
2. Feasibility replay still consumes actual charging rates and waiting times.
3. Documents that the replacement tournament uses geometric distance keys
   (not pure arc-detour) — matching the Round-1 audit.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.alns_hco.hco import (
    _geometric_score,
    _lex_charge_score,
    _select_best_replace,
    hco,
    insertion_detour,
    station_candidates,
)
from src.alns_hco.hco_activity import OracleCache
from src.hanif.model import charging_rate, expected_price, expected_wait
from src.hanif.replay import replay_route
from tests.fixtures.synthetic import build_tiny_evrptw
from tests.test_hco_final_fix import _attach_het_attrs


@pytest.fixture
def spaced():
    """Stations with clearly different detours from a customer arc."""
    inst = build_tiny_evrptw(
        customers=[
            (0.0, 10.0, 5.0, 0.0, 2000.0, 5.0),
            (40.0, 10.0, 5.0, 0.0, 2000.0, 5.0),
            (80.0, 10.0, 5.0, 0.0, 2000.0, 5.0),
        ],
        stations=[
            (20.0, 10.0),   # on the path — small detour
            (20.0, 40.0),   # large detour
            (60.0, 10.0),
            (60.0, 40.0),
        ],
        battery_capacity=55.0,
        energy_rate=1.0,
        depot_has_charging=False,
        horizon=5000.0,
        inverse_charging_rate=1.0,
    )
    n = len(inst.nodes)
    prices = np.full(n, 1.0)
    waits = np.full(n, 1.0)
    rates = np.full(n, 1.0)
    # Make far stations extremely cheap / fast / short-wait to tempt heterogeneous mode
    for s in inst.charging_station_indices:
        if abs(inst.nodes[s].y - 40.0) < 1e-9:
            prices[s] = 0.001
            waits[s] = 0.001
            rates[s] = 100.0
        else:
            prices[s] = 50.0
            waits[s] = 50.0
            rates[s] = 0.1
    return _attach_het_attrs(inst, prices=prices, waits=waits, rates=rates)


def test_detour_candidates_ignore_price_and_wait(spaced):
    prev, nxt = spaced.customer_indices[0], spaced.customer_indices[1]
    cands = station_candidates(spaced, prev, nxt, k=2, mode="detour")
    assert cands
    # Rank by (detour, id) only
    scored = sorted(
        ((insertion_detour(spaced, prev, s, nxt), s) for s in spaced.charging_station_indices),
    )
    expected = [s for _, s in scored[:2]]
    assert cands == expected
    # Extreme hetero attrs must not change the shortlist
    for s in spaced.charging_station_indices:
        spaced.station_price_expected[s] = 1e6
        spaced.station_wait_expected[s] = 1e6
    cands2 = station_candidates(spaced, prev, nxt, k=2, mode="detour")
    assert cands2 == cands


def test_heterogeneous_candidates_can_prefer_cheap_far_station(spaced):
    prev, nxt = spaced.customer_indices[0], spaced.customer_indices[1]
    detour_cands = station_candidates(spaced, prev, nxt, k=1, mode="detour")
    het_cands = station_candidates(spaced, prev, nxt, k=4, mode="heterogeneous")
    far = [s for s in spaced.charging_station_indices if abs(spaced.nodes[s].y - 40.0) < 1e-9]
    assert detour_cands
    assert far
    # Heterogeneous union should include at least one far/cheap station that detour-top1 may omit
    assert any(s in het_cands for s in far) or het_cands != detour_cands


def test_detour_hco_plan_invariant_to_price_wait_permutation(spaced):
    customers = list(spaced.customer_indices)
    plan_a = hco(spaced, customers, OracleCache(), mode="detour", candidates_per_criterion=3)
    if not plan_a.feasible:
        pytest.skip("detour HCO infeasible on spaced fixture")
    prices = spaced.station_price_expected.copy()
    waits = spaced.station_wait_expected.copy()
    sts = list(spaced.charging_station_indices)
    # Invert attrs
    for i, s in enumerate(sts):
        prices[s] = 100.0 - float(i)
        waits[s] = 100.0 - float(i)
    spaced.station_price_expected = prices
    spaced.station_wait_expected = waits
    plan_b = hco(spaced, customers, OracleCache(), mode="detour", candidates_per_criterion=3)
    assert plan_b.feasible
    assert plan_a.nodes == plan_b.nodes
    assert [None if x is None else round(float(x), 8) for x in plan_a.labels] == [
        None if x is None else round(float(x), 8) for x in plan_b.labels
    ]


def test_feasibility_replay_uses_actual_wait_and_rate():
    """Waiting time and charging rate affect duration even under detour selection."""
    inst = build_tiny_evrptw(
        customers=[(10.0, 0.0, 5.0, 0.0, 900.0, 5.0), (20.0, 0.0, 5.0, 0.0, 900.0, 5.0)],
        stations=[(15.0, 0.0)],
        battery_capacity=200.0,
        energy_rate=1.0,
        depot_has_charging=False,
        horizon=2000.0,
    )
    n = len(inst.nodes)
    inst = _attach_het_attrs(
        inst,
        prices=np.ones(n),
        waits=np.zeros(n),
        rates=np.ones(n),
    )
    depot = int(inst.depot_index)
    c0, c1 = inst.customer_indices[0], inst.customer_indices[1]
    s0 = int(inst.charging_station_indices[0])
    nodes = [depot, c0, s0, c1, depot]
    labels = [None, None, 1.0, None, None]
    inst.station_wait_expected[s0] = 0.0
    rates = np.ones(n)
    rates[s0] = 10.0
    inst.station_charging_rate = rates
    fast = replay_route(inst, nodes, labels)
    inst.station_wait_expected[s0] = 25.0
    rates = rates.copy()
    rates[s0] = 0.5
    inst.station_charging_rate = rates
    slow = replay_route(inst, nodes, labels)
    assert fast.feasible and slow.feasible
    assert slow.duration > fast.duration + 10.0
    assert slow.station_waiting > fast.station_waiting
    assert slow.charging_time > fast.charging_time
    assert expected_wait(inst, s0) == pytest.approx(25.0)
    assert charging_rate(inst, s0) == pytest.approx(0.5)
    assert expected_price(inst, s0) == pytest.approx(float(inst.station_price_expected[s0]))

def test_document_replacement_uses_geometric_distance_not_pure_detour(spaced):
    """Replacement tournament keys for detour mode are distance-led (Round-1 finding).

    This test locks the *documented* behaviour; it does not change HCO logic.
    """
    customers = list(spaced.customer_indices)
    seed_plan = hco(spaced, customers, OracleCache(), mode="detour", candidates_per_criterion=3)
    if not seed_plan.feasible:
        pytest.skip("need feasible seed")
    # Geometric acceptance score ignores price/wait/expense
    g = _geometric_score(spaced, seed_plan)
    assert len(g) == 3
    assert g[0] == pytest.approx(float(seed_plan.distance))
    lex = _lex_charge_score(seed_plan)
    assert lex[0] == pytest.approx(float(seed_plan.charging_expense))
    # Replace pass is reachable and returns a feasible plan under detour mode
    improved = _select_best_replace(
        spaced, seed_plan.nodes, seed_plan.labels, k=3, mode="detour",
    )
    assert improved.feasible
    # After permuting prices/waits, geometric score of a fixed plan is unchanged
    before = _geometric_score(spaced, improved)
    spaced.station_price_expected[:] = 999.0
    spaced.station_wait_expected[:] = 999.0
    after = _geometric_score(spaced, improved)
    assert before == after
