"""Energy accounting vs authoritative Hanif replay on real feasible routes."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.alns_hco.shared_phase1 import ensure_shared_phase1, shared_phase1_path
from src.common.energy_accounting import account_solution_energy
from src.common.paths import hanif_dirs
from src.hanif.loader import list_instances, load_hanif
from src.hanif.replay import replay_solution


def _has_shared(name: str = "C1_10_1", tw: str = "original", seed: int = 0) -> bool:
    root = hanif_dirs()["shared_phase1"]
    return shared_phase1_path(root, name, tw, seed).is_file()


@pytest.mark.skipif(not list_instances(), reason="Hanif converted instances unavailable")
def test_energy_matches_replay_on_shared_phase1_hanif_routes():
    if not _has_shared():
        pytest.skip("shared Phase-I artifact missing for C1_10_1")
    shared = ensure_shared_phase1(hanif_dirs()["shared_phase1"], instance_name="C1_10_1", tw="original", seed=0)
    inst = shared["instance"]
    routes = shared["routes"]
    replay = replay_solution(inst, routes)
    assert replay.feasible
    assert replay.unserved == 0
    energy = account_solution_energy(inst, routes)
    assert energy.energy_balance_ok
    assert abs(energy.energy_balance_residual) <= 1e-6
    assert energy.enroute_purchased_energy == pytest.approx(replay.charged_energy, abs=1e-6)
    assert energy.charging_visits == int(replay.charging_visits)
    assert energy.propulsion_energy > 0.0
    assert energy.depot_replenishment_energy == pytest.approx(
        energy.initial_battery_energy - energy.residual_battery_at_depot, abs=1e-6
    )
    # Identity: B0 + purchased - propulsion = residual
    lhs = energy.initial_battery_energy + energy.enroute_purchased_energy - energy.propulsion_energy
    assert lhs == pytest.approx(energy.residual_battery_at_depot, abs=1e-6)


@pytest.mark.skipif(not list_instances(), reason="Hanif converted instances unavailable")
def test_gi_feasible_route_energy_identity():
    from src.hanif.benchmarks.greedy import solve_greedy

    inst = load_hanif("C1_10_1", "original")
    out = solve_greedy(inst, deadline=40.0)
    assert out["metrics"].feasible
    energy = account_solution_energy(inst, out["routes"])
    assert energy.energy_balance_ok
    assert energy.enroute_purchased_energy == pytest.approx(out["metrics"].charged_energy, abs=1e-6)
    assert energy.charging_visits == int(out["metrics"].charging_visits)
