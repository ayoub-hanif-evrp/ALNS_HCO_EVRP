"""Loader for converted E-VRPTW instances.

Reads only from ``dataset/evrptw_instances/<name>/`` which contains:
    - evrp_instance.json
    - nodes.csv
    - matrices.npz
    - summary.json

The raw ``dataset/vrptw_instances/`` folder is never read here.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Order of per-node features fed to the encoder.
NODE_FEATURE_COLUMNS = [
    "x_norm",
    "y_norm",
    "demand_norm",
    "ready_time_norm",
    "due_time_norm",
    "service_time_norm",
    "is_depot",
    "is_customer",
    "is_charging_station",
]

# Integer ids for node types (used for embeddings / debugging).
NODE_TYPE_TO_ID = {"depot": 0, "customer": 1, "charging_station": 2}


@dataclass(frozen=True)
class Node:
    id: int
    string_id: str
    node_type: str
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float
    is_depot: bool
    is_customer: bool
    is_charging_station: bool
    x_norm: float
    y_norm: float
    demand_norm: float
    ready_time_norm: float
    due_time_norm: float
    service_time_norm: float


@dataclass(frozen=True)
class VehicleParams:
    battery_capacity: float
    load_capacity: float
    energy_consumption_rate: float
    inverse_charging_rate: float
    speed: float


@dataclass
class EVRPTWInstance:
    instance_name: str
    nodes: list[Node]
    depot_index: int
    customer_indices: list[int]
    charging_station_indices: list[int]
    rechargeable_indices: list[int]
    distance_matrix: np.ndarray
    travel_time_matrix: np.ndarray
    energy_matrix: np.ndarray
    vehicle: VehicleParams
    planning_horizon: float
    depot_has_charging: bool = False
    max_abs_coordinate: float = 1.0

    # --- convenience properties ---
    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_customers(self) -> int:
        return len(self.customer_indices)

    @property
    def num_charging_stations(self) -> int:
        return len(self.charging_station_indices)

    @property
    def total_demand(self) -> float:
        return float(sum(self.nodes[c].demand for c in self.customer_indices))

    @property
    def m_min(self) -> int:
        """Minimum vehicles required to satisfy total demand."""
        if self.vehicle.load_capacity <= 0:
            return 1
        return max(1, math.ceil(self.total_demand / self.vehicle.load_capacity))

    def demand_array(self) -> np.ndarray:
        return np.array([n.demand for n in self.nodes], dtype=np.float32)

    def ready_time_array(self) -> np.ndarray:
        return np.array([n.ready_time for n in self.nodes], dtype=np.float32)

    def due_time_array(self) -> np.ndarray:
        return np.array([n.due_time for n in self.nodes], dtype=np.float32)

    def service_time_array(self) -> np.ndarray:
        return np.array([n.service_time for n in self.nodes], dtype=np.float32)

    def node_type_ids(self) -> np.ndarray:
        return np.array([NODE_TYPE_TO_ID[n.node_type] for n in self.nodes], dtype=np.int64)

    def node_feature_matrix(self) -> np.ndarray:
        """[N, F] float matrix of per-node features in NODE_FEATURE_COLUMNS order."""
        rows = []
        for n in self.nodes:
            rows.append([
                n.x_norm,
                n.y_norm,
                n.demand_norm,
                n.ready_time_norm,
                n.due_time_norm,
                n.service_time_norm,
                float(n.is_depot),
                float(n.is_customer),
                float(n.is_charging_station),
            ])
        return np.array(rows, dtype=np.float32)

    # --- tensors ---
    def to_tensors(self, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
        """Return tensors describing the static graph for the encoder/env."""
        device = torch.device(device)
        return {
            "node_features": torch.tensor(self.node_feature_matrix(), device=device),
            "distance_matrix": torch.tensor(self.distance_matrix, dtype=torch.float32, device=device),
            "travel_time_matrix": torch.tensor(self.travel_time_matrix, dtype=torch.float32, device=device),
            "energy_matrix": torch.tensor(self.energy_matrix, dtype=torch.float32, device=device),
            "node_type_ids": torch.tensor(self.node_type_ids(), device=device),
            "demand": torch.tensor(self.demand_array(), device=device),
            "ready_time": torch.tensor(self.ready_time_array(), device=device),
            "due_time": torch.tensor(self.due_time_array(), device=device),
            "service_time": torch.tensor(self.service_time_array(), device=device),
            "is_depot": torch.tensor([n.is_depot for n in self.nodes], dtype=torch.bool, device=device),
            "is_customer": torch.tensor([n.is_customer for n in self.nodes], dtype=torch.bool, device=device),
            "is_charging_station": torch.tensor(
                [n.is_charging_station for n in self.nodes], dtype=torch.bool, device=device
            ),
        }


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def load_instance(instance_dir: str | Path) -> EVRPTWInstance:
    """Load a single converted instance from its folder."""
    instance_dir = Path(instance_dir)
    json_path = instance_dir / "evrp_instance.json"
    csv_path = instance_dir / "nodes.csv"
    npz_path = instance_dir / "matrices.npz"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing evrp_instance.json in {instance_dir}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    vp = data["vehicle_parameters"]
    index_sets = data["index_sets"]

    # Prefer nodes.csv for normalized features; fall back to JSON nodes.
    nodes: list[Node] = []
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df.sort_values("id").reset_index(drop=True)
        for _, row in df.iterrows():
            nodes.append(_node_from_row(row, vp, meta))
    else:
        planning_horizon = float(meta["planning_horizon"])
        max_abs = float(meta.get("max_abs_coordinate", 1.0)) or 1.0
        for nd in data["nodes"]:
            nodes.append(_node_from_json(nd, vp["load_capacity"], planning_horizon, max_abs))

    matrices = np.load(npz_path)
    distance_matrix = matrices["distance_matrix"].astype(np.float32)
    travel_time_matrix = matrices["travel_time_matrix"].astype(np.float32)
    energy_matrix = matrices["energy_matrix"].astype(np.float32)

    vehicle = VehicleParams(
        battery_capacity=float(vp["battery_capacity"]),
        load_capacity=float(vp["load_capacity"]),
        energy_consumption_rate=float(vp["energy_consumption_rate"]),
        inverse_charging_rate=float(vp["inverse_charging_rate"]),
        speed=float(vp["speed"]),
    )

    return EVRPTWInstance(
        instance_name=meta["instance_name"],
        nodes=nodes,
        depot_index=int(index_sets["depot_index"]),
        customer_indices=[int(i) for i in index_sets["customer_indices"]],
        charging_station_indices=[int(i) for i in index_sets["charging_station_indices"]],
        rechargeable_indices=[int(i) for i in index_sets["rechargeable_indices"]],
        distance_matrix=distance_matrix,
        travel_time_matrix=travel_time_matrix,
        energy_matrix=energy_matrix,
        vehicle=vehicle,
        planning_horizon=float(meta["planning_horizon"]),
        depot_has_charging=bool(meta.get("depot_has_charging", False)),
        max_abs_coordinate=float(meta.get("max_abs_coordinate", 1.0)) or 1.0,
    )


def _node_from_row(row: pd.Series, vp: dict, meta: dict) -> Node:
    return Node(
        id=int(row["id"]),
        string_id=str(row["string_id"]),
        node_type=str(row["node_type"]),
        x=float(row["x"]),
        y=float(row["y"]),
        demand=float(row["demand"]),
        ready_time=float(row["ready_time"]),
        due_time=float(row["due_time"]),
        service_time=float(row["service_time"]),
        is_depot=_coerce_bool(row["is_depot"]),
        is_customer=_coerce_bool(row["is_customer"]),
        is_charging_station=_coerce_bool(row["is_charging_station"]),
        x_norm=float(row["x_norm"]),
        y_norm=float(row["y_norm"]),
        demand_norm=float(row["demand_norm"]),
        ready_time_norm=float(row["ready_time_norm"]),
        due_time_norm=float(row["due_time_norm"]),
        service_time_norm=float(row["service_time_norm"]),
    )


def _node_from_json(nd: dict, load_capacity: float, planning_horizon: float, max_abs: float) -> Node:
    horizon = planning_horizon if planning_horizon > 0 else 1.0
    cap = load_capacity if load_capacity > 0 else 1.0
    return Node(
        id=int(nd["id"]),
        string_id=str(nd["string_id"]),
        node_type=str(nd["node_type"]),
        x=float(nd["x"]),
        y=float(nd["y"]),
        demand=float(nd["demand"]),
        ready_time=float(nd["ready_time"]),
        due_time=float(nd["due_time"]),
        service_time=float(nd["service_time"]),
        is_depot=bool(nd["is_depot"]),
        is_customer=bool(nd["is_customer"]),
        is_charging_station=bool(nd["is_charging_station"]),
        x_norm=float(nd["x"]) / max_abs,
        y_norm=float(nd["y"]) / max_abs,
        demand_norm=float(nd["demand"]) / cap,
        ready_time_norm=float(nd["ready_time"]) / horizon,
        due_time_norm=float(nd["due_time"]) / horizon,
        service_time_norm=float(nd["service_time"]) / horizon,
    )


def list_instance_dirs(dataset_path: str | Path) -> list[Path]:
    """List converted-instance directories under the dataset path, sorted by name."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    return sorted(d for d in dataset_path.iterdir() if d.is_dir() and (d / "evrp_instance.json").exists())
