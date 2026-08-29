"""Schneider EVRPTW parser, conversion, and lexicographic BKS comparison."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.schneider_txt_convert import (
    LARGE_56,
    convert_one_txt,
    parse_schneider_txt,
    schneider_family,
)
from src.evaluation.schneider_bks import BKS_FR, BKS_PR, compare_row, summarize

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "dataset" / "Schneider" / "raw" / "c101_21.txt"


@pytest.mark.skipif(not RAW.is_file(), reason="Schneider c101_21.txt not downloaded")
def test_parse_c101_vehicle_and_counts():
    parsed = parse_schneider_txt(RAW)
    assert len(parsed["customers"]) == 100
    assert len(parsed["stations"]) == 21
    assert parsed["depot"]["x"] == pytest.approx(40.0)
    assert parsed["Q"] == pytest.approx(79.69)
    assert parsed["C"] == pytest.approx(200.0)
    assert parsed["r"] == pytest.approx(1.0)
    assert parsed["g"] == pytest.approx(3.39)
    assert parsed["v"] == pytest.approx(1.0)


@pytest.mark.skipif(not RAW.is_file(), reason="Schneider c101_21.txt not downloaded")
def test_convert_c101_no_wait_price(tmp_path: Path):
    convert_one_txt(RAW, tmp_path)
    from src.data.instance_loader import load_instance

    inst = load_instance(tmp_path / "c101_21")
    assert inst.num_customers == 100
    assert len(inst.charging_station_indices) == 21
    assert inst.vehicle.battery_capacity == pytest.approx(79.69)
    assert inst.vehicle.inverse_charging_rate == pytest.approx(3.39)
    assert inst.station_wait_expected is None
    assert inst.station_price_expected is None
    depot = inst.depot_index
    c20 = next(i for i, n in enumerate(inst.nodes) if n.string_id == "C20")
    assert inst.distance_matrix[depot, c20] == pytest.approx(10.0, abs=1e-5)
    assert inst.energy_matrix[depot, c20] == pytest.approx(10.0, abs=1e-5)
    assert inst.travel_time_matrix[depot, c20] == pytest.approx(10.0, abs=1e-5)
    s0 = next(i for i, n in enumerate(inst.nodes) if n.string_id == "S0")
    assert inst.distance_matrix[depot, s0] == pytest.approx(0.0, abs=1e-6)


def test_families_and_bks_coverage():
    assert len(LARGE_56) == 56
    assert schneider_family("c101_21") == "C1"
    assert schneider_family("r211_21") == "R2"
    assert schneider_family("rc208_21") == "RC2"
    assert set(BKS_FR) == set(LARGE_56)
    assert set(BKS_PR) == set(LARGE_56)
    assert BKS_PR["c101_21"]["bks_k"] == 12
    assert BKS_PR["c101_21"]["bks_d"] == pytest.approx(1043.38)
    assert BKS_FR["c101_21"]["bks_d"] == pytest.approx(1053.83)


def test_lex_gap_rule():
    win = compare_row("fr", "c201_21", 3, 700.0)
    assert win["delta_k"] == -1
    assert win["verdict"] == "fleet_win"
    assert win["distance_gap_%"] is None
    loss = compare_row("fr", "c201_21", 5, 600.0)
    assert loss["verdict"] == "fleet_loss"
    tie_d = compare_row("pr", "c101_21", 12, 1043.38)
    assert tie_d["verdict"] == "tie"
    assert tie_d["distance_gap_%"] == pytest.approx(0.0, abs=1e-6)
    worse_d = compare_row("pr", "c101_21", 12, 1147.718)
    assert worse_d["verdict"] == "distance_loss"
    assert worse_d["distance_gap_%"] == pytest.approx(10.0, abs=1e-3)
    summary = summarize([
        {**win, "feasible": True},
        {**loss, "feasible": True},
        {**tie_d, "feasible": True},
    ])
    assert summary["fleet_wins"] == 1
    assert summary["fleet_losses"] == 1
    assert summary["fleet_ties"] == 1


def test_pr_c102_erdelic_fewer_vehicles_is_latest_bks():
    rec = BKS_PR["c102_21"]
    assert rec["literature_k"] == 11
    assert rec["erdelic_k"] == 10
    assert rec["bks_k"] == 10
    assert rec["bks_source"] == "ERDE"
