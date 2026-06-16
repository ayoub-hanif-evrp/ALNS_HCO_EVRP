"""Tests for the instance loader and dataset batching."""
from __future__ import annotations

import numpy as np
import torch

from src.data.evrptw_dataset import EVRPTWDataset, collate_instances
from src.data.instance_loader import apply_load_capacity, load_instance
from tests.conftest import DATASET_PATH, SMALL_INSTANCE


def test_load_small_instance_structure(small_instance):
    inst = small_instance
    assert inst.instance_name == SMALL_INSTANCE
    # Exactly one depot.
    assert sum(n.is_depot for n in inst.nodes) == 1
    assert inst.depot_index == 0
    # Index sets are disjoint and cover the right node types.
    assert all(inst.nodes[c].is_customer for c in inst.customer_indices)
    assert all(inst.nodes[s].is_charging_station for s in inst.charging_station_indices)
    assert inst.num_customers >= 1
    assert inst.num_charging_stations >= 1


def test_matrices_shape_and_consistency(small_instance):
    inst = small_instance
    n = inst.num_nodes
    for mat in (inst.distance_matrix, inst.travel_time_matrix, inst.energy_matrix):
        assert mat.shape == (n, n)
        assert np.allclose(np.diag(mat), 0.0)
        assert np.allclose(mat, mat.T, atol=1e-4)
    # travel_time == distance / speed, energy == distance * rate.
    assert np.allclose(inst.travel_time_matrix, inst.distance_matrix / inst.vehicle.speed, atol=1e-4)
    assert np.allclose(
        inst.energy_matrix, inst.distance_matrix * inst.vehicle.energy_consumption_rate, atol=1e-4
    )


def test_vehicle_parameters_positive(small_instance):
    vp = small_instance.vehicle
    assert vp.battery_capacity > 0
    assert vp.load_capacity > 0
    assert vp.energy_consumption_rate > 0
    assert vp.inverse_charging_rate > 0
    assert vp.speed > 0


def test_rechargeable_includes_depot_and_stations(small_instance):
    inst = small_instance
    expected = sorted([inst.depot_index] + inst.charging_station_indices)
    assert sorted(inst.rechargeable_indices) == expected


def test_m_min_positive(small_instance):
    assert small_instance.m_min >= 1


def test_apply_load_capacity_rescales_demand_norm(small_instance):
    inst = apply_load_capacity(small_instance, 100.0)
    assert inst.vehicle.load_capacity == 100.0
    for n in inst.nodes:
        if n.demand > 0:
            assert abs(n.demand_norm - n.demand / 100.0) < 1e-6
    assert inst.m_min >= small_instance.m_min


def test_to_tensors_types(small_instance):
    tensors = small_instance.to_tensors("cpu")
    n = small_instance.num_nodes
    assert tensors["node_features"].shape == (n, len(small_instance.node_feature_matrix()[0]))
    assert tensors["distance_matrix"].shape == (n, n)
    assert tensors["is_customer"].dtype == torch.bool
    assert tensors["node_type_ids"].dtype == torch.int64


def test_dataset_and_collate():
    ds = EVRPTWDataset(DATASET_PATH, size_filter=["C5"])
    assert len(ds) >= 2
    batch_instances = [ds[0], ds[1]]
    batch = collate_instances(batch_instances, "cpu")
    max_nodes = max(i.num_nodes for i in batch_instances)
    assert batch["node_features"].shape[0] == 2
    assert batch["node_features"].shape[1] == max_nodes
    # Padding mask marks padded nodes as True.
    for b, inst in enumerate(batch_instances):
        assert not batch["node_padding_mask"][b, : inst.num_nodes].any()
        assert batch["node_padding_mask"][b, inst.num_nodes :].all()
