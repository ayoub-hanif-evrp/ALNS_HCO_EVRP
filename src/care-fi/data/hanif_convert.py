"""Convert Hanif Solomon .txt variants (A original TW / B relaxed TW) to EVRPTW folders.

Writes a NEW output tree only. Never modifies source .txt / station CSVs.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from src.hanif.data.instance_loader import expected_lognormal_mean
from src.hanif.utils.logging_utils import get_logger

LOGGER = get_logger()

DEFAULT_SOURCE_A = Path(
    "dataset/Hanif/EVRPTW_ChargingStations_Dataset/variant_A_original_TW"
)
DEFAULT_SOURCE_B = Path(
    "dataset/Hanif/EVRPTW_ChargingStations_Dataset/variant_B_relaxed_TW"
)
DEFAULT_EV_SPECS = Path(
    "dataset/Hanif/EVRPTW_ChargingStations_Dataset/ev_specs.json"
)
DEFAULT_OUTPUT_A = Path("dataset/Hanif/converted_hanif_original_tw")
DEFAULT_OUTPUT_B = Path("dataset/Hanif/converted_hanif_relaxed_tw")

# Back-compat aliases
DEFAULT_SOURCE = DEFAULT_SOURCE_A
DEFAULT_OUTPUT = DEFAULT_OUTPUT_A
SOURCE_VARIANT = "variant_A_original_TW"

VARIANT_SPECS = {
    "A": {
        "key": "variant_A_original_TW",
        "source": DEFAULT_SOURCE_A,
        "output": DEFAULT_OUTPUT_A,
        "time_windows_modified": False,
        "label": "original-TW",
    },
    "B": {
        "key": "variant_B_relaxed_TW",
        "source": DEFAULT_SOURCE_B,
        "output": DEFAULT_OUTPUT_B,
        "time_windows_modified": True,
        "label": "relaxed-TW",
    },
}


def _parse_solomon_txt(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 10:
        raise ValueError(f"Hanif file too short: {path}")
    vehicle_line = lines[4].split()
    if len(vehicle_line) < 2:
        raise ValueError(f"Missing VEHICLE NUMBER/CAPACITY in {path}")
    fleet_size = int(vehicle_line[0])
    load_capacity = float(vehicle_line[1])

    depot = None
    customers: list[tuple] = []
    stations: list[tuple] = []
    for line in lines[9:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        node_id = int(parts[0])
        row = (
            node_id,
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
        )
        if node_id == 0:
            depot = row
        elif node_id <= 1000:
            customers.append(row)
        else:
            stations.append(row)
    if depot is None:
        raise ValueError(f"No depot in {path}")
    return {
        "fleet_size": fleet_size,
        "load_capacity": load_capacity,
        "depot": depot,
        "customers": customers,
        "stations": stations,
    }


def _load_station_attrs(path: Path) -> dict[int, dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing stations CSV: {path}")
    attrs: dict[int, dict] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            sid = int(float(row.get("station_id") or row.get("id") or row.get("node_id")))
            attrs[sid] = {
                "charging_rate": float(row["charging_rate"]),
                "price_mu": float(row["price_mu"]),
                "price_sigma": float(row["price_sigma"]),
                "wait_mu": float(row["wait_mu"]),
                "wait_sigma": float(row["wait_sigma"]),
                "density_score": float(row.get("density_score", 0.0) or 0.0),
            }
    return attrs


def _euclidean_matrices(
    xs: np.ndarray, ys: np.ndarray, *, speed: float, energy_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy).astype(np.float32)
    travel = (dist / float(speed)).astype(np.float32)
    energy = (dist * float(energy_rate)).astype(np.float32)
    return dist, travel, energy


def convert_one_instance(
    txt_path: Path,
    output_dir: Path,
    *,
    battery_capacity: float,
    energy_consumption_rate: float,
    speed: float = 1.0,
    source_variant: str = SOURCE_VARIANT,
    time_windows_modified: bool = False,
) -> dict:
    name = txt_path.stem
    stations_csv = txt_path.with_name(f"{name}_stations.csv")
    parsed = _parse_solomon_txt(txt_path)
    attrs = _load_station_attrs(stations_csv)

    depot = parsed["depot"]
    customers = parsed["customers"]
    stations = parsed["stations"]
    if len(stations) != len(attrs):
        missing = sorted(set(s[0] for s in stations) - set(attrs))
        extra = sorted(set(attrs) - set(s[0] for s in stations))
        raise ValueError(
            f"{name}: station txt/csv mismatch "
            f"(txt={len(stations)} csv={len(attrs)} missing={missing[:5]} extra={extra[:5]})"
        )

    nodes_raw = [depot] + customers + stations
    n = len(nodes_raw)
    xs = np.array([r[1] for r in nodes_raw], dtype=np.float64)
    ys = np.array([r[2] for r in nodes_raw], dtype=np.float64)
    max_abs = float(max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1.0))
    horizon = float(depot[5])
    load_cap = float(parsed["load_capacity"])

    id_to_idx = {int(r[0]): i for i, r in enumerate(nodes_raw)}
    customer_indices = [id_to_idx[int(r[0])] for r in customers]
    station_indices = [id_to_idx[int(r[0])] for r in stations]
    rechargeable = [0] + station_indices

    rates = [float(attrs[int(r[0])]["charging_rate"]) for r in stations]
    median_rate = float(np.median(rates)) if rates else 1.0
    inverse_charging_rate = 1.0 / median_rate if median_rate > 1e-12 else 1.0

    dist, travel, energy = _euclidean_matrices(
        xs, ys, speed=speed, energy_rate=energy_consumption_rate,
    )

    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    node_rows = []
    nodes_json = []
    for i, r in enumerate(nodes_raw):
        nid, x, y, demand, ready, due, service = r
        is_depot = i == 0
        is_customer = i in set(customer_indices)
        is_station = i in set(station_indices)
        if is_depot:
            ntype = "depot"
            string_id = "D0"
        elif is_customer:
            ntype = "customer"
            string_id = f"C{nid}"
        else:
            ntype = "charging_station"
            string_id = f"S{nid}"
        row = {
            "id": i,
            "string_id": string_id,
            "node_type": ntype,
            "x": float(x),
            "y": float(y),
            "demand": float(demand),
            "ready_time": float(ready),
            "due_time": float(due),
            "service_time": float(service),
            "is_depot": is_depot,
            "is_customer": is_customer,
            "is_charging_station": is_station,
            "x_norm": float(x) / max_abs,
            "y_norm": float(y) / max_abs,
            "demand_norm": float(demand) / load_cap if load_cap > 0 else 0.0,
            "ready_time_norm": float(ready) / horizon if horizon > 0 else 0.0,
            "due_time_norm": float(due) / horizon if horizon > 0 else 0.0,
            "service_time_norm": float(service) / horizon if horizon > 0 else 0.0,
            "hanif_original_id": int(nid),
        }
        node_rows.append(row)
        nodes_json.append({k: row[k] for k in row if k != "hanif_original_id"})

    with (out_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(node_rows[0].keys()))
        w.writeheader()
        w.writerows(node_rows)

    station_rows = []
    prices = []
    for r in stations:
        sid = int(r[0])
        a = attrs[sid]
        price = expected_lognormal_mean(a["price_mu"], a["price_sigma"])
        prices.append(price)
        station_rows.append({
            "node_id": id_to_idx[sid],
            "hanif_station_id": sid,
            "charging_rate": a["charging_rate"],
            "price_mu": a["price_mu"],
            "price_sigma": a["price_sigma"],
            "expected_price_per_energy": price,
            "wait_mu": a["wait_mu"],
            "wait_sigma": a["wait_sigma"],
            "density_score": a["density_score"],
        })
    with (out_dir / "stations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(station_rows[0].keys()))
        w.writeheader()
        w.writerows(station_rows)

    np.savez_compressed(
        out_dir / "matrices.npz",
        distance_matrix=dist,
        travel_time_matrix=travel,
        energy_matrix=energy,
    )

    meta = {
        "instance_name": name,
        "problem_type": "E-VRPTW",
        "source_dataset": "Hanif EVRPTW Charging Stations Dataset",
        "source_variant": source_variant,
        "source_doi": "10.3390/data11040083",
        "time_windows_modified": bool(time_windows_modified),
        "num_depots": 1,
        "num_customers": len(customers),
        "num_charging_stations": len(stations),
        "depot_has_charging": True,
        "planning_horizon": horizon,
        "max_abs_coordinate": max_abs,
        "hanif_fleet_size": int(parsed["fleet_size"]),
        "station_wait_model": "expected_lognormal_wait",
        "station_attrs_supported": [
            "charging_rate", "price_mu", "price_sigma", "wait_mu", "density_score",
        ],
        "station_attrs_unsupported_stochastic": ["wait_sigma"],
    }
    vehicle = {
        "battery_capacity": float(battery_capacity),
        "load_capacity": load_cap,
        "energy_consumption_rate": float(energy_consumption_rate),
        "inverse_charging_rate": float(inverse_charging_rate),
        "speed": float(speed),
    }
    payload = {
        "metadata": meta,
        "vehicle_parameters": vehicle,
        "index_sets": {
            "depot_index": 0,
            "customer_indices": customer_indices,
            "charging_station_indices": station_indices,
            "rechargeable_indices": rechargeable,
        },
        "nodes": nodes_json,
    }
    (out_dir / "evrp_instance.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8",
    )

    summary = {
        "instance_name": name,
        "variant": source_variant,
        "time_windows_modified": bool(time_windows_modified),
        "num_customers": len(customers),
        "num_stations": len(stations),
        "num_nodes": n,
        "planning_horizon": horizon,
        "load_capacity": load_cap,
        "battery_capacity": float(battery_capacity),
        "energy_consumption_rate": float(energy_consumption_rate),
        "charging_rate_min": float(min(rates)) if rates else None,
        "charging_rate_max": float(max(rates)) if rates else None,
        "charging_rate_median": median_rate,
        "price_per_energy_min": float(min(prices)) if prices else None,
        "price_per_energy_max": float(max(prices)) if prices else None,
        "price_per_energy_median": float(np.median(prices)) if prices else None,
        "station_wait_model": "expected_lognormal_wait",
        "source_ev_specs": str(DEFAULT_EV_SPECS).replace("\\", "/"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def ensure_hanif_variant_converted(
    variant: str = "A",
    *,
    source_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    ev_specs_path: str | Path = DEFAULT_EV_SPECS,
    force: bool = False,
) -> Path:
    """Convert all instances for variant A (original TW) or B (relaxed TW)."""
    key = str(variant).strip().upper()
    if key in ("ORIGINAL", "ORIGINAL_TW", "VARIANT_A"):
        key = "A"
    if key in ("RELAXED", "RELAXED_TW", "VARIANT_B"):
        key = "B"
    if key not in VARIANT_SPECS:
        raise ValueError(f"Unknown Hanif variant {variant!r}; use A or B")
    spec = VARIANT_SPECS[key]
    source_dir = Path(source_dir or spec["source"])
    output_dir = Path(output_dir or spec["output"])
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Hanif {spec['key']} source not found: {source_dir}")

    specs = json.loads(Path(ev_specs_path).read_text(encoding="utf-8"))
    battery = float(specs["battery_capacity"])
    rate = float(specs["energy_consumption_rate"])

    txts = sorted(p for p in source_dir.glob("*.txt") if not p.name.startswith("."))
    if not txts:
        raise FileNotFoundError(f"No .txt instances under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    need = []
    for txt in txts:
        marker = output_dir / txt.stem / "evrp_instance.json"
        if force or not marker.is_file():
            need.append(txt)
            continue
        meta = json.loads(marker.read_text(encoding="utf-8")).get("metadata") or {}
        if str(meta.get("source_variant", "")) != spec["key"]:
            need.append(txt)

    t0 = time.time()
    for i, txt in enumerate(need):
        LOGGER.info(
            "Converting Hanif %s [%d/%d] %s",
            spec["label"], i + 1, len(need), txt.stem,
        )
        convert_one_instance(
            txt, output_dir,
            battery_capacity=battery,
            energy_consumption_rate=rate,
            source_variant=spec["key"],
            time_windows_modified=bool(spec["time_windows_modified"]),
        )

    names = sorted(p.stem for p in txts)
    missing = [n for n in names if not (output_dir / n / "evrp_instance.json").is_file()]
    if missing:
        raise RuntimeError(f"Hanif conversion incomplete; missing: {missing[:10]}")

    manifest = {
        "source": str(source_dir).replace("\\", "/"),
        "output": str(output_dir).replace("\\", "/"),
        "variant": spec["key"],
        "time_windows_modified": bool(spec["time_windows_modified"]),
        "num_instances": len(names),
        "converted_this_call": len(need),
        "elapsed_seconds": round(time.time() - t0, 3),
        "instances": names,
    }
    (output_dir / "conversion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8",
    )
    LOGGER.info(
        "Hanif %s ready | %d instances under %s (converted_now=%d)",
        spec["label"], len(names), output_dir, len(need),
    )
    return output_dir


def ensure_hanif_original_tw_converted(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_A,
    output_dir: str | Path = DEFAULT_OUTPUT_A,
    ev_specs_path: str | Path = DEFAULT_EV_SPECS,
    force: bool = False,
) -> Path:
    source_dir = Path(source_dir)
    if "relaxed" in str(source_dir).lower() or "variant_b" in str(source_dir).lower():
        raise ValueError(
            f"Refusing relaxed-TW source: {source_dir}. "
            "Use variant_A_original_TW only, or call ensure_hanif_relaxed_tw_converted."
        )
    return ensure_hanif_variant_converted(
        "A", source_dir=source_dir, output_dir=output_dir,
        ev_specs_path=ev_specs_path, force=force,
    )


def ensure_hanif_relaxed_tw_converted(
    *,
    source_dir: str | Path = DEFAULT_SOURCE_B,
    output_dir: str | Path = DEFAULT_OUTPUT_B,
    ev_specs_path: str | Path = DEFAULT_EV_SPECS,
    force: bool = False,
) -> Path:
    return ensure_hanif_variant_converted(
        "B", source_dir=source_dir, output_dir=output_dir,
        ev_specs_path=ev_specs_path, force=force,
    )


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_OUTPUT_A",
    "DEFAULT_OUTPUT_B",
    "DEFAULT_SOURCE",
    "DEFAULT_SOURCE_A",
    "DEFAULT_SOURCE_B",
    "SOURCE_VARIANT",
    "VARIANT_SPECS",
    "convert_one_instance",
    "ensure_hanif_original_tw_converted",
    "ensure_hanif_relaxed_tw_converted",
    "ensure_hanif_variant_converted",
]
