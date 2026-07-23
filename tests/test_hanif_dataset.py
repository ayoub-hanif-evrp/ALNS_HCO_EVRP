"""Hanif dataset / split / attribute smoke tests."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.data.hanif_convert import SOURCE_VARIANT
from src.data.hanif_dataset import (
    deterministic_class_split,
    resolve_hanif_split,
    save_split_into_config,
)
from src.data.instance_loader import load_instance
from src.models.dynamic_candidate import compute_dynamic_candidate_features
from src.models.feature_specs import DYNAMIC_CANDIDATE_FEAT_DIM

ROOT = Path(__file__).resolve().parents[1]
HANIF_OUT = ROOT / "dataset" / "Hanif" / "converted_hanif_original_tw"
HANIF_SRC = ROOT / "dataset" / "Hanif" / "EVRPTW_ChargingStations_Dataset" / "variant_A_original_TW"


@pytest.mark.skipif(not (HANIF_OUT / "C1_10_1" / "evrp_instance.json").is_file(), reason="Hanif converted data missing")
def test_hanif_loader_wait_and_density():
    inst = load_instance(HANIF_OUT / "C1_10_1")
    assert inst.num_customers == 1000
    assert inst.num_charging_stations == 333
    assert inst.station_wait_mu is not None
    assert inst.station_density_score is not None
    assert inst.station_charging_rate is not None
    meta = __import__("json").loads(
        (HANIF_OUT / "C1_10_1" / "evrp_instance.json").read_text(encoding="utf-8")
    )["metadata"]
    assert meta["source_variant"] == SOURCE_VARIANT
    assert meta.get("time_windows_modified") is False


@pytest.mark.skipif(not HANIF_OUT.is_dir(), reason="Hanif converted root missing")
def test_hanif_split_no_leakage():
    config = {
        "dataset": {
            "name": "hanif",
            "path": str(HANIF_OUT),
            "root": str(HANIF_OUT),
            "source_txt": str(HANIF_SRC),
            "classes": ["C", "R", "RC"],
            "use_only_this_dataset": True,
            "pilot": {"enabled": True, "train_per_class": 2, "val_per_class": 1, "test_per_class": 1},
        },
        "problem": {"load_capacity": None},
    }
    train, val, test, meta = resolve_hanif_split(config)
    save_split_into_config(config, train, val, test, meta)
    assert len(train) == 6
    assert len(val) == 3
    assert len(test) == 3
    assert not (set(train) & set(val) | set(train) & set(test) | set(val) & set(test))
    for cls in ("C", "R", "RC"):
        part = meta["split_by_class"][cls]
        assert len(part["train"]) == 2
        assert len(part["validation"]) == 1
        assert len(part["test"]) == 1


def test_deterministic_class_split_sorted():
    names = [f"C1_10_{i}" for i in (1, 10, 2, 3, 4)]
    sorted_names = sorted(names)
    split = deterministic_class_split(sorted_names, train_n=2, val_n=1, test_n=1)
    assert split["train"] == sorted_names[:2]
    assert split["validation"] == [sorted_names[2]]
    assert split["test"] == [sorted_names[3]]


@pytest.mark.skipif(not (HANIF_OUT / "C1_10_1" / "evrp_instance.json").is_file(), reason="Hanif converted data missing")
def test_dynamic_features_include_wait_density():
    """Stub mask result — avoid full 1000-node exact analysis in unit tests."""
    inst = load_instance(HANIF_OUT / "C1_10_1")
    n = inst.num_nodes
    env = SimpleNamespace(
        instance=inst,
        served_mask=np.zeros(n, dtype=bool),
        reserved_mask=np.zeros(n, dtype=bool),
        use_horizon_check=True,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
    )
    ev = SimpleNamespace(
        current_node=0,
        current_time=0.0,
        remaining_battery=float(inst.vehicle.battery_capacity),
        remaining_capacity=float(inst.vehicle.load_capacity),
    )
    mask_result = SimpleNamespace(
        node_mask=np.ones(n, dtype=bool),
        hard_node_mask=np.ones(n, dtype=bool),
        charge_masks={},
        charge_portion_details={},
        depot_mode_mask=np.array([True, True], dtype=bool),
    )
    feats = compute_dynamic_candidate_features(env, ev, mask_result)
    assert feats.shape == (n, DYNAMIC_CANDIDATE_FEAT_DIM)
    station_idx = inst.charging_station_indices[0]
    assert feats[station_idx, 17] >= 0.0
    assert float(feats[station_idx, 18]) == pytest.approx(
        float(inst.station_density_score[station_idx]), rel=1e-5, abs=1e-5,
    )


@pytest.mark.skipif(not HANIF_SRC.is_dir(), reason="Hanif source missing")
def test_refuse_relaxed_source(tmp_path: Path):
    from src.data.hanif_convert import ensure_hanif_original_tw_converted

    with pytest.raises(ValueError, match="relaxed"):
        ensure_hanif_original_tw_converted(
            source_dir=tmp_path / "variant_B_relaxed_TW",
            output_dir=tmp_path / "out",
        )
