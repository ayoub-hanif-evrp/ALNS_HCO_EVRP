"""Tests for direct Hanif dataset loading (no pre-conversion)."""
from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from src.data.evrptw_dataset import EVRPTWDataset
from src.data.hanif_loader import load_hanif_instance
from src.data.hanif_parser import parse_hanif_instance
from src.data.instance_class import instance_class, is_small_instance

HANIF_VARIANT_B = Path("dataset/Hanif/EVRPTW_ChargingStations_Dataset/variant_B_relaxed_TW")
EV_SPECS = Path("dataset/Hanif/EVRPTW_ChargingStations_Dataset/ev_specs.json")


def test_hanif_instance_class_and_tiers():
    assert instance_class("C1_10_1") == "C"
    assert instance_class("RC2_10_5") == "RC"
    assert is_small_instance("C1_10_1")
    assert is_small_instance("RC1_10_3")
    assert not is_small_instance("C2_10_1")
    assert not is_small_instance("RC2_10_5")


@pytest.mark.skipif(not (HANIF_VARIANT_B / "C1_10_1.txt").exists(), reason="Hanif data missing")
def test_parse_hanif_instance():
    parsed = parse_hanif_instance(HANIF_VARIANT_B / "C1_10_1.txt")
    assert len(parsed.customers) == 1000
    assert len(parsed.stations) == 333
    assert parsed.load_capacity == 200.0


@pytest.mark.skipif(not (HANIF_VARIANT_B / "C1_10_1.txt").exists(), reason="Hanif data missing")
def test_load_hanif_instance_in_memory():
    inst = load_hanif_instance(HANIF_VARIANT_B / "C1_10_1.txt", ev_specs_path=EV_SPECS)
    assert inst.num_customers == 1000
    assert inst.vehicle.battery_capacity == 600.0
    assert inst.vehicle.load_capacity == 200.0


@pytest.mark.skipif(not (HANIF_VARIANT_B / "C1_10_1.txt").exists(), reason="Hanif data missing")
def test_hanif_station_attributes_from_csv():
    parsed = parse_hanif_instance(HANIF_VARIANT_B / "C1_10_1.txt")
    assert 1001 in parsed.station_attributes
    attr = parsed.station_attributes[1001]
    assert attr.charging_rate > 0
    assert attr.wait_mu > 0


@pytest.mark.skipif(not (HANIF_VARIANT_B / "C1_10_1.txt").exists(), reason="Hanif data missing")
def test_convert_hanif_one_instance(tmp_path):
    from src.data.hanif_converter import convert_hanif_txt
    from src.data.hanif_parser import load_station_attributes
    from src.data.instance_loader import load_instance

    out = convert_hanif_txt(
        HANIF_VARIANT_B / "C1_10_1.txt",
        tmp_path,
        ev_specs_path=EV_SPECS,
    )
    inst = load_instance(out)
    assert inst.num_customers == 1000
    assert (out / "matrices.npz").exists()
    assert (out / "stations.csv").exists()
    assert inst.station_charging_rate is not None
    assert inst.station_price_per_energy is not None
    src_attrs = load_station_attributes(HANIF_VARIANT_B / "C1_10_1_stations.csv")
    first_station_idx = inst.charging_station_indices[0]
    hanif_id = int(inst.nodes[first_station_idx].string_id)
    expected_rate = src_attrs[hanif_id].charging_rate
    expected_price = float(np.exp(src_attrs[hanif_id].price_mu + 0.5 * src_attrs[hanif_id].price_sigma ** 2))
    assert abs(float(inst.station_charging_rate[first_station_idx]) - expected_rate) < 1e-4
    assert abs(float(inst.station_price_per_energy[first_station_idx]) - expected_price) < 1e-4


@pytest.mark.skipif(
    not Path("dataset/converted_hanif_data/C1_10_1/summary.json").exists(),
    reason="Converted Hanif data missing",
)
def test_converted_hanif_vehicle_params_match_summary():
    from src.data.instance_loader import load_instance, read_instance_summary

    cases = {
        "C1_10_1": 200.0,
        "C2_10_1": 700.0,
        "R2_10_1": 1000.0,
        "RC2_10_1": 1000.0,
    }
    root = Path("dataset/converted_hanif_data")
    for name, expected_load in cases.items():
        summary = read_instance_summary(root / name)
        inst = load_instance(root / name)
        assert summary is not None
        assert summary["battery_capacity"] == 600.0
        assert summary["load_capacity"] == expected_load
        assert inst.vehicle.battery_capacity == summary["battery_capacity"]
        assert inst.vehicle.load_capacity == summary["load_capacity"]
