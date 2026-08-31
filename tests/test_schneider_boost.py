"""Type-aware Schneider start fleet, just-enough charge, and merge helpers."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.schneider.data.instance_loader import load_instance
from src.schneider.search.fleet_search import adaptive_start_fleet, compute_fleet_bounds
from src.schneider.search.schneider_boost import (
    _tw_optimistic_ok,
    is_long_horizon,
    just_enough_charge,
    schneider_start_upper,
    try_absorb_shortest_route,
)

ROOT = Path(__file__).resolve().parents[1]
C101 = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101_21"
C203 = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c203_21"


@pytest.mark.skipif(not C101.is_dir(), reason="Schneider c101 not converted")
def test_type1_is_short_horizon():
    inst = load_instance(C101)
    assert not is_long_horizon(inst)
    lo, hi = compute_fleet_bounds(inst, {"fleet_search": {"initial_upper_m": 18, "max_upper_fleet": 40}})
    start = adaptive_start_fleet(lo, hi, inst)
    assert start >= 12
    assert start <= 20


@pytest.mark.skipif(not C203.is_dir(), reason="Schneider c203 not converted")
def test_type2_starts_near_capacity_not_24():
    inst = load_instance(C203)
    assert is_long_horizon(inst)
    lo, hi = compute_fleet_bounds(inst, {"fleet_search": {
        "initial_upper_m": 18, "type2_initial_upper_m": 8, "max_upper_fleet": 40,
    }})
    start = schneider_start_upper(lo, hi, inst)
    assert start <= 8
    assert start >= lo + 2
    # The old flat initial_upper_m=24 must not apply to C2.
    from src.schneider.search.fleet_search import solve_with_fleet_search
    del solve_with_fleet_search
    cfg = {"fleet_search": {"initial_upper_m": 18, "type2_initial_upper_m": 8, "max_upper_fleet": 40}}
    lo2, hi2 = compute_fleet_bounds(inst, cfg)
    # adaptive_start used inside solve; check helper only
    assert schneider_start_upper(lo2, min(hi2, 18), inst) <= 8


@pytest.mark.skipif(not C203.is_dir(), reason="Schneider c203 not converted")
def test_just_enough_charge_is_partial():
    inst = load_instance(C203)
    depot = inst.depot_index
    stations = list(inst.charging_station_indices)
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    assert stations and custs
    q = float(inst.vehicle.battery_capacity)
    need = just_enough_charge(inst, stations[0], 0.15 * q, custs[0])
    assert 0 < need < q
    full = just_enough_charge(inst, depot, q, custs[0])
    assert full == 0.0


@pytest.mark.skipif(not C203.is_dir(), reason="Schneider c203 not converted")
def test_tw_prune_and_absorb_do_not_raise():
    inst = load_instance(C203)
    depot = inst.depot_index
    custs = [i for i, n in enumerate(inst.nodes) if n.is_customer][:6]
    assert _tw_optimistic_ok(inst, custs[:2])
    routes = [
        ([depot, custs[0], depot], [None, None, None]),
        ([depot] + custs[1:] + [depot], [None] * (len(custs[1:]) + 2)),
    ]
    out, score, meta = try_absorb_shortest_route(inst, routes, config={"problem": {"greedy_min_charge": True}})
    assert "absorbed" in meta
    assert len(out) >= 1
    if meta["absorbed"]:
        assert score is not None
