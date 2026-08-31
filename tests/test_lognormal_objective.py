"""Unit tests for log-normal station semantics and fleet-first Φ ranking."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from src.schneider.data.instance_loader import load_instance
from src.schneider.env.lognormal import cvar_lognormal, expected_lognormal, std_lognormal
from src.schneider.evaluation.paper_objective import (
    MODE_CLASSICAL,
    MODE_EXPECTED,
    MODE_RISK_AWARE,
    ObjectiveReference,
    effective_price_s,
    effective_wait_s,
    expected_price_s,
    expected_wait_s,
    independent_recompute_phi,
    normalize_phi,
    operational_components,
    ranking_tuple,
)
from src.schneider.search.score import SearchScore

HANIF = Path("dataset/Hanif/converted_hanif_original_tw/C1_10_1")


def test_expected_lognormal_mean():
    mu, sigma = -1.3086, 0.198
    got = expected_lognormal(mu, sigma)
    assert abs(got - math.exp(mu + 0.5 * sigma ** 2)) < 1e-12


def test_std_lognormal_formula():
    mu, sigma = 0.4, 0.3
    mean = expected_lognormal(mu, sigma)
    assert abs(std_lognormal(mu, sigma) - mean * math.sqrt(math.exp(sigma ** 2) - 1)) < 1e-12
    assert std_lognormal(mu, 0.0) == 0.0


def test_cvar_lognormal_vs_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sigma, alpha = 0.2, 0.35, 0.95
    samples = rng.lognormal(mean=mu, sigma=sigma, size=250_000)
    var = np.quantile(samples, alpha)
    tail = samples[samples >= var]
    mc = float(np.mean(tail))
    closed = cvar_lognormal(mu, sigma, alpha)
    assert abs(closed - mc) / mc < 0.03


def test_wait_mu_is_not_expected_wait():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    inst = load_instance(HANIF)
    sid = int(inst.charging_station_indices[0])
    mu = float(inst.station_wait_mu[sid])
    expected = float(inst.station_wait_expected[sid])
    assert abs(expected - mu) > 1e-6
    assert abs(expected - expected_lognormal(mu, float(inst.station_wait_sigma[sid]))) < 1e-5


def test_price_mu_is_not_expected_price():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    inst = load_instance(HANIF)
    sid = int(inst.charging_station_indices[0])
    mu = float(inst.station_price_mu[sid])
    expected = float(inst.station_price_expected[sid])
    assert abs(expected - mu) > 1e-6
    assert abs(expected - expected_price_s(inst, sid)) < 1e-6


def test_rho_zero_matches_expected_price_wait():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    inst = load_instance(HANIF)
    sid = int(inst.charging_station_indices[3])
    assert abs(effective_price_s(inst, sid, rho=0.0, alpha=0.95) - expected_price_s(inst, sid)) < 1e-9
    assert abs(effective_wait_s(inst, sid, rho=0.0, alpha=0.95) - expected_wait_s(inst, sid)) < 1e-9


def test_sigma_changes_risk_not_expected():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    inst = load_instance(HANIF)
    sid = int(inst.charging_station_indices[2])
    e0 = effective_price_s(inst, sid, rho=0.0, alpha=0.95)
    r0 = effective_price_s(inst, sid, rho=0.25, alpha=0.95)
    inst.station_price_sigma[sid] = float(inst.station_price_sigma[sid]) + 0.4
    e1 = effective_price_s(inst, sid, rho=0.0, alpha=0.95)
    r1 = effective_price_s(inst, sid, rho=0.25, alpha=0.95)
    assert abs(e1 - e0) < 1e-9
    assert r1 > r0 + 1e-6


def test_density_not_in_operational_components():
    comps = operational_components(
        type("I", (), {
            "nodes": [],
            "vehicle": type("V", (), {"inverse_charging_rate": 1.0})(),
            "station_density_score": np.array([99.0]),
        })(),
        distance=10.0,
        charging_visits=[],
        rho=0.25,
        alpha=0.95,
    )
    dumped = str(comps)
    assert "99" not in dumped
    assert comps["crisk"] == 0.0
    assert comps["wrisk"] == 0.0


def test_no_charge_zero_station_cost():
    phi, _ = normalize_phi(
        distance=100.0, tstation=0.0, crisk=0.0,
        reference=ObjectiveReference(100.0, 10.0, 10.0),
    )
    assert abs(phi - 1.0) < 1e-9


def test_lower_fleet_always_beats_higher_fleet():
    a = SearchScore(0, 0, 10, 9999, 1, 1, 1, phi=99.0, objective_mode=MODE_RISK_AWARE)
    b = SearchScore(0, 0, 11, 1, 1, 1, 1, phi=0.0, objective_mode=MODE_RISK_AWARE)
    assert a.dominates(b)


def test_same_fleet_better_phi_wins():
    a = SearchScore(0, 0, 10, 100, 1, 1, 1, phi=1.1, objective_mode=MODE_EXPECTED)
    b = SearchScore(0, 0, 10, 50, 1, 1, 1, phi=1.5, objective_mode=MODE_EXPECTED)
    assert a.dominates(b)


def test_same_fleet_and_phi_distance_tiebreak():
    a = SearchScore(0, 0, 10, 80, 1, 1, 1, phi=2.0, total_station_time=5, objective_mode=MODE_EXPECTED)
    b = SearchScore(0, 0, 10, 90, 1, 1, 1, phi=2.0, total_station_time=1, objective_mode=MODE_EXPECTED)
    assert a.dominates(b)


def test_infeasible_never_beats_feasible():
    bad = SearchScore(1, 0, 5, 1, 1, 1, 1, phi=0.0, objective_mode=MODE_CLASSICAL)
    good = SearchScore(0, 0, 80, 1e6, 1, 1, 1, phi=99.0, objective_mode=MODE_CLASSICAL)
    assert good.dominates(bad)
    assert not bad.dominates(good)


def test_classical_ignores_phi():
    t = ranking_tuple(
        hard=0, unserved=0, used_k=10, phi=0.1, distance=50,
        station_time=9, duration=20, charging_time=3, paper_j=1, mode=MODE_CLASSICAL,
    )
    assert t[3] == 50.0
    assert t[4] == 20.0
    assert len(t) == 5


def test_station_waiting_separated_from_customer_waiting():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    from src.schneider.search.route_replay import replay_solution

    inst = load_instance(HANIF)
    depot = inst.depot_index
    cust = int(inst.customer_indices[0])
    res = replay_solution(inst, [([depot, cust, depot], [None, None, None])], m=1)
    assert res.station_waiting_time == 0.0
    assert res.customer_early_waiting_time >= 0.0
    assert abs(res.waiting_time - (res.customer_early_waiting_time + res.station_waiting_time)) < 1e-6


def test_charging_time_energy_over_rate():
    class Node:
        def __init__(self):
            self.is_charging_station = True
            self.is_depot = False

    class Inst:
        nodes = [Node(), Node()]
        vehicle = type("V", (), {"inverse_charging_rate": 1.0})()
        station_charging_rate = np.array([np.nan, 2.0])
        station_price_mu = np.array([0.0, 0.0])
        station_price_sigma = np.array([0.0, 0.0])
        station_price_expected = np.array([0.0, 4.0])
        station_price_per_energy = np.array([0.0, 4.0])
        station_wait_mu = np.array([0.0, 0.0])
        station_wait_sigma = np.array([0.0, 0.0])
        station_wait_expected = np.array([0.0, 5.0])

    comps = operational_components(
        Inst(),
        distance=10.0,
        charging_visits=[{"station_id": 1, "energy_added": 8.0}],
        rho=0.0,
        alpha=0.95,
    )
    assert abs(comps["charging_time"] - 4.0) < 1e-9
    assert abs(comps["crisk"] - 32.0) < 1e-9
    assert abs(comps["wrisk"] - 5.0) < 1e-9


def test_wait_sigma_changes_risk_not_expected():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    inst = load_instance(HANIF)
    sid = int(inst.charging_station_indices[2])
    e0 = effective_wait_s(inst, sid, rho=0.0, alpha=0.95)
    r0 = effective_wait_s(inst, sid, rho=0.25, alpha=0.95)
    inst.station_wait_sigma[sid] = float(inst.station_wait_sigma[sid]) + 0.4
    e1 = effective_wait_s(inst, sid, rho=0.0, alpha=0.95)
    r1 = effective_wait_s(inst, sid, rho=0.25, alpha=0.95)
    assert abs(e1 - e0) < 1e-9
    assert r1 > r0 + 1e-6


def test_exact_replay_agrees_with_search_objective():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    from src.schneider.search.route_replay import replay_solution

    inst = load_instance(HANIF)
    depot = inst.depot_index
    cust = int(inst.customer_indices[0])
    cfg = {
        "paper_objective": {"mode": "expected_heterogeneous", "risk_blend": 0.0, "risk_alpha": 0.95},
        "objective_reference": {"D_ref": 1.0, "Tstation_ref": 1.0, "Ccharge_ref": 1.0},
    }
    res = replay_solution(
        inst, [([depot, cust, depot], [None, None, None])], m=1, config=cfg,
    )
    audit = independent_recompute_phi(inst, res, cfg)
    assert audit["agree"]
    assert res.score is not None
def test_energy_consumed_from_matrix_not_zero():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    from src.schneider.search.route_replay import replay_solution

    inst = load_instance(HANIF)
    depot = inst.depot_index
    cust = int(inst.customer_indices[0])
    e12 = float(inst.energy_matrix[depot, cust])
    e21 = float(inst.energy_matrix[cust, depot])
    h = float(inst.vehicle.energy_consumption_rate)
    d12 = float(inst.distance_matrix[depot, cust])
    assert abs(e12 - h * d12) < 1e-4 * max(1.0, abs(e12))
    res = replay_solution(inst, [([depot, cust, depot], [None, None, None])], m=1)
    expected = e12 + e21
    assert res.energy_consumed > 0.0
    assert abs(res.energy_consumed - expected) < 1e-4 * max(1.0, expected)
    assert float(res.battery_metrics["total_energy_consumed"]) > 0.0
    assert abs(float(res.battery_metrics["total_energy_consumed"]) - expected) < 1e-3 * max(1.0, expected)


def test_price_mu_is_not_used_as_raw_price_in_env_wait_field_names():
    if not (HANIF / "evrp_instance.json").is_file():
        pytest.skip("converted instance missing")
    inst = load_instance(HANIF)
    sid = int(inst.charging_station_indices[0])
    assert abs(float(inst.station_price_expected[sid]) - float(inst.station_price_mu[sid])) > 1e-6
    from src.schneider.env.lognormal import expected_lognormal
    assert abs(
        float(inst.station_price_expected[sid])
        - expected_lognormal(float(inst.station_price_mu[sid]), float(inst.station_price_sigma[sid]))
    ) < 1e-5
