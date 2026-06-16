"""Convert Hanif ``.txt`` instances to precomputed Schneider-style folders for fast loading."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.hanif_parser import HanifNode, HanifParsedInstance, parse_hanif_instance
from src.data.instance_loader import EVRPTWInstance, Node, VehicleParams, apply_load_capacity, expected_lognormal_mean


def _load_ev_specs(ev_specs_path: Path | None) -> tuple[float, float, float]:
    battery_capacity = 600.0
    energy_consumption_rate = 1.0
    speed = 1.0
    if ev_specs_path and ev_specs_path.exists():
        specs = json.loads(ev_specs_path.read_text(encoding="utf-8"))
        battery_capacity = float(specs.get("battery_capacity", battery_capacity))
        energy_consumption_rate = float(specs.get("energy_consumption_rate", energy_consumption_rate))
    return battery_capacity, energy_consumption_rate, speed


def _median_charging_rate(station_attributes: dict, station_nodes: list[HanifNode]) -> float:
    values = [
        float(station_attributes[s.node_id].charging_rate)
        for s in station_nodes
        if s.node_id in station_attributes and station_attributes[s.node_id].charging_rate > 1e-9
    ]
    if not values:
        return 1.0
    return float(np.median(values))


def build_hanif_instance(
    parsed: HanifParsedInstance,
    *,
    ev_specs_path: str | Path | None = None,
    variant: str = "variant_B_relaxed_TW",
    load_capacity_override: float | None = None,
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None, dict[str, np.ndarray], dict, EVRPTWInstance, np.ndarray]:
    """Build converted artifacts and an in-memory ``EVRPTWInstance``."""
    battery_capacity, energy_consumption_rate, speed = _load_ev_specs(
        Path(ev_specs_path) if ev_specs_path else None
    )
    load_cap = float(load_capacity_override if load_capacity_override is not None else parsed.load_capacity)
    planning_horizon = float(parsed.depot.due_time)
    station_attrs = parsed.station_attributes
    median_rate = _median_charging_rate(station_attrs, parsed.stations)
    inverse_charging_rate = 1.0 / median_rate if median_rate > 1e-9 else 1.0

    ordered: list[tuple[HanifNode, str]] = [(parsed.depot, "depot")]
    ordered.extend((c, "customer") for c in parsed.customers)
    ordered.extend((s, "charging_station") for s in parsed.stations)

    num_nodes = len(ordered)
    station_charging_rate = np.full(num_nodes, np.nan, dtype=np.float32)
    station_price_per_energy = np.full(num_nodes, np.nan, dtype=np.float32)
    station_rows: list[dict] = []

    xs = np.array([n.x for n, _ in ordered], dtype=float)
    ys = np.array([n.y for n, _ in ordered], dtype=float)
    max_abs = float(max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1.0))
    hz = planning_horizon if planning_horizon > 0 else 1.0
    cap = load_cap if load_cap > 0 else 1.0

    converted_nodes: list[dict] = []
    customer_indices: list[int] = []
    charging_station_indices: list[int] = []
    nodes: list[Node] = []

    for new_id, (hanif_node, node_type) in enumerate(ordered):
        is_depot = node_type == "depot"
        is_customer = node_type == "customer"
        is_station = node_type == "charging_station"
        if is_customer:
            customer_indices.append(new_id)
        if is_station:
            charging_station_indices.append(new_id)
            attr = station_attrs.get(hanif_node.node_id)
            rate = float(attr.charging_rate) if attr is not None else median_rate
            price = expected_lognormal_mean(attr.price_mu, attr.price_sigma) if attr is not None else np.nan
            station_charging_rate[new_id] = rate
            station_price_per_energy[new_id] = price
            if attr is not None:
                station_rows.append({
                    "node_id": new_id,
                    "hanif_station_id": attr.station_id,
                    "charging_rate": attr.charging_rate,
                    "price_mu": attr.price_mu,
                    "price_sigma": attr.price_sigma,
                    "expected_price_per_energy": price,
                    "wait_mu": attr.wait_mu,
                    "wait_sigma": attr.wait_sigma,
                    "density_score": attr.density_score,
                })

        row = {
            "id": new_id,
            "string_id": str(hanif_node.node_id),
            "node_type": node_type,
            "x": float(hanif_node.x),
            "y": float(hanif_node.y),
            "demand": float(hanif_node.demand),
            "ready_time": float(hanif_node.ready_time),
            "due_time": float(hanif_node.due_time),
            "service_time": float(hanif_node.service_time),
            "is_depot": is_depot,
            "is_customer": is_customer,
            "is_charging_station": is_station,
            "x_norm": float(hanif_node.x) / max_abs,
            "y_norm": float(hanif_node.y) / max_abs,
            "demand_norm": float(hanif_node.demand) / cap,
            "ready_time_norm": float(hanif_node.ready_time) / hz,
            "due_time_norm": float(hanif_node.due_time) / hz,
            "service_time_norm": float(hanif_node.service_time) / hz,
        }
        converted_nodes.append(row)
        nodes.append(Node(**row))

    if median_rate > 1e-9:
        station_charging_rate[0] = median_rate
    median_price = float(np.nanmedian(station_price_per_energy))
    if not np.isnan(median_price):
        station_price_per_energy[0] = median_price

    stations_df = pd.DataFrame(station_rows) if station_rows else None
    rate_values = station_charging_rate[~np.isnan(station_charging_rate)]
    price_values = station_price_per_energy[~np.isnan(station_price_per_energy)]

    coords = np.stack([xs, ys], axis=1)
    diff = coords[:, None, :] - coords[None, :, :]
    distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2)).astype(np.float32)
    travel_time_matrix = (distance_matrix / speed).astype(np.float32)
    energy_matrix = (distance_matrix * energy_consumption_rate).astype(np.float32)

    vehicle = VehicleParams(
        battery_capacity=battery_capacity,
        load_capacity=load_cap,
        energy_consumption_rate=energy_consumption_rate,
        inverse_charging_rate=inverse_charging_rate,
        speed=speed,
    )

    evrp = {
        "metadata": {
            "instance_name": parsed.instance_name,
            "problem_type": "E-VRPTW",
            "source_dataset": "Hanif EVRPTW Charging Stations Dataset",
            "source_variant": variant,
            "source_doi": "10.3390/data11040083",
            "num_depots": 1,
            "num_customers": len(customer_indices),
            "num_charging_stations": len(charging_station_indices),
            "depot_has_charging": True,
            "planning_horizon": planning_horizon,
            "max_abs_coordinate": max_abs,
            "hanif_fleet_size": parsed.fleet_size,
        },
        "vehicle_parameters": {
            "battery_capacity": battery_capacity,
            "load_capacity": load_cap,
            "energy_consumption_rate": energy_consumption_rate,
            "inverse_charging_rate": inverse_charging_rate,
            "speed": speed,
        },
        "index_sets": {
            "depot_index": 0,
            "customer_indices": customer_indices,
            "charging_station_indices": charging_station_indices,
            "rechargeable_indices": [0, *charging_station_indices],
        },
        "nodes": converted_nodes,
    }

    instance = EVRPTWInstance(
        instance_name=parsed.instance_name,
        nodes=nodes,
        depot_index=0,
        customer_indices=customer_indices,
        charging_station_indices=charging_station_indices,
        rechargeable_indices=[0, *charging_station_indices],
        distance_matrix=distance_matrix,
        travel_time_matrix=travel_time_matrix,
        energy_matrix=energy_matrix,
        vehicle=vehicle,
        planning_horizon=planning_horizon,
        depot_has_charging=True,
        max_abs_coordinate=max_abs,
        station_charging_rate=station_charging_rate,
        station_price_per_energy=station_price_per_energy,
    )

    summary = {
        "instance_name": parsed.instance_name,
        "variant": variant,
        "num_customers": len(customer_indices),
        "num_stations": len(charging_station_indices),
        "num_nodes": len(converted_nodes),
        "planning_horizon": planning_horizon,
        "load_capacity": load_cap,
        "battery_capacity": battery_capacity,
        "energy_consumption_rate": energy_consumption_rate,
        "charging_rate_min": float(np.min(rate_values)) if len(rate_values) else None,
        "charging_rate_max": float(np.max(rate_values)) if len(rate_values) else None,
        "charging_rate_median": float(np.median(rate_values)) if len(rate_values) else None,
        "price_per_energy_min": float(np.min(price_values)) if len(price_values) else None,
        "price_per_energy_max": float(np.max(price_values)) if len(price_values) else None,
        "price_per_energy_median": float(np.median(price_values)) if len(price_values) else None,
        "source_ev_specs": str(ev_specs_path) if ev_specs_path else None,
    }
    matrices = {
        "distance_matrix": distance_matrix,
        "travel_time_matrix": travel_time_matrix,
        "energy_matrix": energy_matrix,
    }
    return evrp, pd.DataFrame(converted_nodes), stations_df, matrices, summary, instance, station_charging_rate


def save_converted_hanif_instance(
    out_dir: str | Path,
    evrp: dict,
    nodes_df: pd.DataFrame,
    stations_df: pd.DataFrame | None,
    matrices: dict,
    summary: dict,
) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "evrp_instance.json").write_text(json.dumps(evrp, indent=2), encoding="utf-8")
    nodes_df.to_csv(out_dir / "nodes.csv", index=False)
    if stations_df is not None and not stations_df.empty:
        stations_df.to_csv(out_dir / "stations.csv", index=False)
    np.savez_compressed(
        out_dir / "matrices.npz",
        distance_matrix=matrices["distance_matrix"],
        travel_time_matrix=matrices["travel_time_matrix"],
        energy_matrix=matrices["energy_matrix"],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def convert_hanif_txt(
    txt_path: str | Path,
    out_root: str | Path,
    *,
    ev_specs_path: str | Path | None = None,
    variant: str = "variant_B_relaxed_TW",
) -> Path:
    parsed = parse_hanif_instance(txt_path)
    evrp, nodes_df, stations_df, matrices, summary, _, _ = build_hanif_instance(
        parsed, ev_specs_path=ev_specs_path, variant=variant,
    )
    return save_converted_hanif_instance(
        Path(out_root) / parsed.instance_name, evrp, nodes_df, stations_df, matrices, summary,
    )


def convert_hanif_directory(
    source_dir: str | Path,
    out_root: str | Path,
    *,
    ev_specs_path: str | Path | None = None,
    variant: str = "variant_B_relaxed_TW",
) -> list[Path]:
    from src.data.hanif_loader import list_hanif_txt_files

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    converted: list[Path] = []
    for txt_path in list_hanif_txt_files(source_dir):
        converted.append(convert_hanif_txt(txt_path, out_root, ev_specs_path=ev_specs_path, variant=variant))
    return converted
