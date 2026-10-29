"""Schneider adapter validation (neutral wait/price; no invented hetero attrs)."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.hanif.data.schneider_txt_convert import (
    DEFAULT_RAW,
    convert_one_txt,
    parse_schneider_txt,
    schneider_family,
)
from src.hanif.model import expected_price, expected_wait


RAW = Path(DEFAULT_RAW)


@pytest.mark.skipif(not (RAW / "c101_21.txt").is_file(), reason="Schneider raw missing")
def test_parse_and_family():
    assert schneider_family("c101_21") == "C1"
    assert schneider_family("rc201_21") == "RC2"
    p = parse_schneider_txt(RAW / "c101_21.txt")
    assert len(p["customers"]) == 100
    assert len(p["stations"]) == 21
    assert p["Q"] > 0 and p["C"] > 0 and p["r"] > 0 and p["g"] > 0


@pytest.mark.skipif(not (RAW / "c101_21.txt").is_file(), reason="Schneider raw missing")
def test_convert_neutral_wait_and_params(tmp_path):
    from src.hanif.data.instance_loader import load_instance

    summary = convert_one_txt(RAW / "c101_21.txt", tmp_path)
    assert summary["heterogeneous_stations"] is False
    assert summary["station_wait_model"] == "zero"
    inst = load_instance(tmp_path / "c101_21")
    assert inst.num_customers == 100
    assert len(inst.charging_station_indices) == 21
    src = parse_schneider_txt(RAW / "c101_21.txt")
    assert abs(inst.vehicle.battery_capacity - src["Q"]) < 1e-9
    assert abs(inst.vehicle.load_capacity - src["C"]) < 1e-9
    assert abs(inst.vehicle.energy_consumption_rate - src["r"]) < 1e-9
    assert abs(inst.vehicle.inverse_charging_rate - src["g"]) < 1e-9
    # Wait must be zero (disabled or exact 0)
    for s in inst.charging_station_indices:
        assert expected_wait(inst, s) == 0.0
        assert abs(expected_price(inst, s) - 1.0) < 1e-6
    assert inst.station_wait_expected is None
