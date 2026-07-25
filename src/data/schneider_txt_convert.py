"""Convert Schneider-style StringID Solomon EVRPTW text files to runtime folders.

Supports Schneider and EVRP-TW-D layouts (optional ``gv`` + metadata footer).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

LOGGER = get_logger()

NODE_COLUMNS = [
    "StringID", "Type", "x", "y", "demand",
    "ReadyTime", "DueDate", "ServiceTime",
]

TYPE_TO_NODE_TYPE = {
    "d": "depot",
    "f": "charging_station",
    "c": "customer",
}

RAW_PARAMETER_MAPPING = {
    "Q": "Vehicle fuel tank capacity -> battery_capacity",
    "C": "Vehicle load capacity -> load_capacity",
    "r": "fuel consumption rate -> energy_consumption_rate",
    "g": "inverse refueling rate -> inverse_charging_rate",
    "v": "average Velocity -> speed",
    "gv": "charging speed (optional; inverse of g)",
}

PARAM_PATTERNS = {
    "Q": re.compile(r"Q\s+Vehicle fuel tank capacity\s+/([\d.+-eE]+)/"),
    "C": re.compile(r"C\s+Vehicle load capacity\s+/([\d.+-eE]+)/"),
    "r": re.compile(r"r\s+fuel consumption rate\s+/([\d.+-eE]+)/"),
    "g": re.compile(r"g\s+inverse refueling rate\s+/([\d.+-eE]+)/"),
    "v": re.compile(r"v\s+average Velocity\s+/([\d.+-eE]+)/"),
    "gv": re.compile(r"gv\s+charging speed\s+/([\d.+-eE]+)/"),
}

COORD_EPS = 1e-6


def parse_raw_instance(filepath: Path) -> tuple[pd.DataFrame, dict[str, float], dict[str, str]]:
    """Parse StringID Solomon file → nodes, vehicle params, optional footer meta."""
    lines = filepath.read_text(encoding="utf-8").strip().splitlines()
    node_rows: list[dict] = []
    raw_params: dict[str, float] = {}
    meta: dict[str, str] = {}
    in_footer = False

    for line in lines[1:]:  # skip header
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("-"):
            in_footer = True
            continue

        matched_param = False
        for key, pattern in PARAM_PATTERNS.items():
            m = pattern.search(stripped)
            if m:
                raw_params[key] = float(m.group(1))
                matched_param = True
                break
        if matched_param:
            continue

        if in_footer:
            # instance_id /0/  style metadata
            if "/" in stripped:
                key, _, rest = stripped.partition("/")
                val = rest.rsplit("/", 1)[0].strip() if "/" in rest else rest.strip()
                meta[key.strip()] = val
            continue

        parts = stripped.split()
        if len(parts) < 8:
            continue
        if parts[1] not in TYPE_TO_NODE_TYPE:
            continue

        node_rows.append({
            "StringID": parts[0],
            "Type": parts[1],
            "x": float(parts[2]),
            "y": float(parts[3]),
            "demand": float(parts[4]),
            "ReadyTime": float(parts[5]),
            "DueDate": float(parts[6]),
            "ServiceTime": float(parts[7]),
        })

    nodes_df = pd.DataFrame(node_rows, columns=NODE_COLUMNS)
    required = ["Q", "C", "r", "g", "v"]
    missing = [k for k in required if k not in raw_params]
    if missing:
        raise ValueError(f"Missing parameters {missing} in {filepath.name}")
    if "gv" not in raw_params and raw_params["g"] > 0:
        raw_params["gv"] = 1.0 / raw_params["g"]
    return nodes_df, raw_params, meta


def _coords_equal(x1: float, y1: float, x2: float, y2: float, eps: float = COORD_EPS) -> bool:
    return abs(x1 - x2) <= eps and abs(y1 - y2) <= eps


def convert_to_evrp(
    nodes_df: pd.DataFrame,
    raw_params: dict[str, float],
    instance_name: str,
    *,
    source_folder: str = "",
    extra_meta: dict | None = None,
) -> tuple[dict, pd.DataFrame, dict[str, np.ndarray]]:
    """Convert parsed raw data into EVRP JSON, nodes CSV rows, and matrices."""
    battery_capacity = raw_params["Q"]
    load_capacity = raw_params["C"]
    energy_consumption_rate = raw_params["r"]
    inverse_charging_rate = raw_params["g"]
    speed = raw_params["v"]

    vehicle_parameters = {
        "battery_capacity": battery_capacity,
        "load_capacity": load_capacity,
        "energy_consumption_rate": energy_consumption_rate,
        "inverse_charging_rate": inverse_charging_rate,
        "speed": speed,
    }
    if "gv" in raw_params:
        vehicle_parameters["charging_speed_gv"] = float(raw_params["gv"])

    depot_rows = nodes_df[nodes_df["Type"] == "d"]
    if depot_rows.empty:
        raise ValueError(f"No depot found in {instance_name}")
    depot_row = depot_rows.iloc[0]
    depot_x, depot_y = float(depot_row["x"]), float(depot_row["y"])

    s0_row = nodes_df[nodes_df["StringID"] == "S0"]
    s0_co_located = False
    if not s0_row.empty:
        s0 = s0_row.iloc[0]
        s0_co_located = _coords_equal(depot_x, depot_y, float(s0["x"]), float(s0["y"]))

    converted_nodes: list[dict] = []
    for idx, row in nodes_df.iterrows():
        original_type = row["Type"]
        node_type = TYPE_TO_NODE_TYPE[original_type]
        is_depot = original_type == "d"
        is_customer = original_type == "c"
        is_charging_station = original_type == "f"
        demand = float(row["demand"])
        if is_charging_station or is_depot:
            demand = 0.0
        node = {
            "id": int(idx),
            "string_id": row["StringID"],
            "original_type": original_type,
            "node_type": node_type,
            "x": float(row["x"]),
            "y": float(row["y"]),
            "demand": demand,
            "ready_time": float(row["ReadyTime"]),
            "due_time": float(row["DueDate"]),
            "service_time": float(row["ServiceTime"]),
            "is_depot": is_depot,
            "is_customer": is_customer,
            "is_charging_station": is_charging_station,
        }
        if row["StringID"] == "S0" and s0_co_located:
            node["co_located_with_depot"] = True
        converted_nodes.append(node)

    depot_indices = [n["id"] for n in converted_nodes if n["is_depot"]]
    customer_indices = [n["id"] for n in converted_nodes if n["is_customer"]]
    charging_station_indices = [n["id"] for n in converted_nodes if n["is_charging_station"]]
    depot_index = depot_indices[0]
    rechargeable_indices = [depot_index] + charging_station_indices

    depot_due = next((n["due_time"] for n in converted_nodes if n["is_depot"]), None)
    if depot_due is None or depot_due <= 0:
        planning_horizon = max(n["due_time"] for n in converted_nodes)
    else:
        planning_horizon = depot_due

    xs = np.array([n["x"] for n in converted_nodes], dtype=float)
    ys = np.array([n["y"] for n in converted_nodes], dtype=float)
    max_abs_coordinate = max(float(np.max(np.abs(xs))), float(np.max(np.abs(ys))), 1.0)

    distance_matrix = np.sqrt((xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2)
    travel_time_matrix = distance_matrix / speed
    energy_matrix = distance_matrix * energy_consumption_rate

    csv_rows = []
    for node in converted_nodes:
        csv_row = dict(node)
        csv_row["x_norm"] = node["x"] / max_abs_coordinate
        csv_row["y_norm"] = node["y"] / max_abs_coordinate
        csv_row["demand_norm"] = node["demand"] / load_capacity if load_capacity > 0 else 0.0
        csv_row["ready_time_norm"] = node["ready_time"] / planning_horizon
        csv_row["due_time_norm"] = node["due_time"] / planning_horizon
        csv_row["service_time_norm"] = node["service_time"] / planning_horizon
        csv_rows.append(csv_row)
    nodes_csv_df = pd.DataFrame(csv_rows)

    metadata = {
        "instance_name": instance_name,
        "problem_type": "E-VRPTW",
        "source_folder": source_folder,
        "source_format": "Schneider-style / EVRP-TW-D StringID Solomon",
        "converted_to": "EVRP energy notation",
        "num_depots": len(depot_indices),
        "num_customers": len(customer_indices),
        "num_charging_stations": len(charging_station_indices),
        "depot_has_charging": bool(s0_co_located),
        "planning_horizon": float(planning_horizon),
        "max_abs_coordinate": float(max_abs_coordinate),
    }
    if extra_meta:
        metadata["source_meta"] = dict(extra_meta)

    evrp_instance = {
        "metadata": metadata,
        "vehicle_parameters": vehicle_parameters,
        "raw_parameter_mapping": RAW_PARAMETER_MAPPING,
        "index_sets": {
            "depot_index": depot_index,
            "customer_indices": customer_indices,
            "charging_station_indices": charging_station_indices,
            "rechargeable_indices": rechargeable_indices,
        },
        "nodes": converted_nodes,
    }
    matrices = {
        "distance_matrix": distance_matrix.astype(np.float64),
        "travel_time_matrix": travel_time_matrix.astype(np.float64),
        "energy_matrix": energy_matrix.astype(np.float64),
    }
    return evrp_instance, nodes_csv_df, matrices


def validate_instance(evrp_instance: dict, matrices: dict[str, np.ndarray]) -> list[str]:
    errors: list[str] = []
    name = evrp_instance["metadata"]["instance_name"]
    nodes = evrp_instance["nodes"]
    vp = evrp_instance["vehicle_parameters"]

    depots = [n for n in nodes if n["is_depot"]]
    customers = [n for n in nodes if n["is_customer"]]
    stations = [n for n in nodes if n["is_charging_station"]]
    if len(depots) != 1:
        errors.append(f"[{name}] Expected 1 depot, found {len(depots)}")
    if len(customers) < 1:
        errors.append(f"[{name}] Expected >=1 customer")
    if len(stations) < 1:
        errors.append(f"[{name}] Expected >=1 charging station")

    for c in customers:
        if c["demand"] <= 0:
            errors.append(f"[{name}] Customer {c['string_id']} demand={c['demand']}")
        if c["ready_time"] > c["due_time"] + 1e-9:
            errors.append(f"[{name}] Customer {c['string_id']}: ready > due")

    for key in ["battery_capacity", "load_capacity", "energy_consumption_rate", "inverse_charging_rate", "speed"]:
        if float(vp[key]) <= 0:
            errors.append(f"[{name}] {key} must be positive")

    n_nodes = len(nodes)
    for mname, mat in matrices.items():
        if mat.shape != (n_nodes, n_nodes):
            errors.append(f"[{name}] {mname} bad shape {mat.shape}")
    return errors


def save_converted_instance(
    output_dir: Path,
    evrp_instance: dict,
    nodes_csv_df: pd.DataFrame,
    matrices: dict[str, np.ndarray],
    validation_errors: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evrp_instance.json").write_text(
        json.dumps(evrp_instance, indent=2), encoding="utf-8",
    )
    nodes_csv_df.to_csv(output_dir / "nodes.csv", index=False)
    np.savez_compressed(
        output_dir / "matrices.npz",
        distance_matrix=matrices["distance_matrix"],
        travel_time_matrix=matrices["travel_time_matrix"],
        energy_matrix=matrices["energy_matrix"],
    )
    meta = evrp_instance["metadata"]
    vp = evrp_instance["vehicle_parameters"]
    summary = {
        "instance_name": meta["instance_name"],
        "num_nodes": len(evrp_instance["nodes"]),
        "num_customers": meta["num_customers"],
        "num_charging_stations": meta["num_charging_stations"],
        "depot_has_charging": meta["depot_has_charging"],
        "planning_horizon": meta["planning_horizon"],
        **{k: v for k, v in vp.items() if k != "charging_speed_gv"},
        "validation_passed": len(validation_errors) == 0,
        "validation_errors": validation_errors,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def convert_one_txt(src: Path, out_root: Path, *, overwrite: bool = False) -> Path:
    """Convert one Solomon txt into ``out_root/<stem>/``."""
    stem = src.stem
    out_dir = out_root / stem
    marker = out_dir / "evrp_instance.json"
    if marker.is_file() and not overwrite:
        return out_dir
    nodes_df, raw_params, meta = parse_raw_instance(src)
    evrp, nodes_csv, matrices = convert_to_evrp(
        nodes_df, raw_params, stem,
        source_folder=str(src.parent),
        extra_meta=meta,
    )
    errors = validate_instance(evrp, matrices)
    if errors:
        raise ValueError(f"Validation failed for {src.name}: {errors[:3]}")
    save_converted_instance(out_dir, evrp, nodes_csv, matrices, errors)
    return out_dir


def convert_many(sources: list[Path], out_root: Path, *, overwrite: bool = False) -> list[str]:
    out_root.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for i, src in enumerate(sources):
        out = convert_one_txt(src, out_root, overwrite=overwrite)
        names.append(out.name)
        if (i + 1) % 50 == 0 or (i + 1) == len(sources):
            LOGGER.info("Converted %d/%d → %s", i + 1, len(sources), out_root)
    return names
