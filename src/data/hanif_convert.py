"""Convert Hanif variant_A_original_TW Solomon .txt files to EVRPTW folders.

Writes a NEW output tree only. Never modifies source .txt / station CSVs.
Rejects variant_B (relaxed TW) as a conversion source.
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from src.data.instance_loader import expected_lognormal_mean
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

DEFAULT_SOURCE = Path(
    "dataset/Hanif/EVRPTW_ChargingStations_Dataset/variant_A_original_TW"
)
DEFAULT_EV_SPECS = Path(
    "dataset/Hanif/EVRPTW_ChargingStations_Dataset/ev_specs.json"
)
DEFAULT_OUTPUT = Path("dataset/Hanif/converted_hanif_original_tw")
SOURCE_VARIANT = "variant_A_original_TW"


def _parse_solomon_txt(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 10:
        raise ValueError(f"Hanif file too short: {path}")
    vehicle_line = lines[4].split()
    if len(vehicle_line) < 2:
        raise ValueError(f"Missing VEHICLE NUMBER/CAPACITY in {path}")
    fleet_size = int(vehicle_line[0])
    load_capacity = float(vehicle_line[1])

    # Homberger/Hanif convention: depot id 0, customers 1..1000, stations >= 1001.
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
    customers.sort(key=lambda r: r[0])
    stations.sort(key=lambda r: r[0])
    return {
        "fleet_size": fleet_size,
        "load_capacity": load_capacity,
        "depot": depot,
        "customers": customers,
        "stations": stations,
    }


def _load_station_attrs(csv_path: Path) -> dict[int, dict]:
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Hanif station attributes CSV missing: {csv_path}. "
            "Station charging_rate/price/wait/density are required."
        )
    out: dict[int, dict] = {}
    with csv_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sid = int(row["station_id"])
            out[sid] = {
                "charging_rate": float(row["charging_rate"]),
                "price_mu": float(row["price_mu"]),
                "price_sigma": float(row["price_sigma"]),
                "wait_mu": float(row["wait_mu"]),
                "wait_sigma": float(row["wait_sigma"]),
                "density_score": float(row["density_score"]),
            }
    return out


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

    # Contiguous indices: depot=0, customers=1..Nc, stations=Nc+1..
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
        "source_variant": SOURCE_VARIANT,
        "source_doi": "10.3390/data11040083",
        "time_windows_modified": False,
        "num_depots": 1,
        "num_customers": len(customers),
        "num_charging_stations": len(stations),
        "depot_has_charging": True,
        "planning_horizon": horizon,
        "max_abs_coordinate": max_abs,
        "hanif_fleet_size": int(parsed["fleet_size"]),
        "station_wait_model": "expected_wait_mu",
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
        "variant": SOURCE_VARIANT,
        "time_windows_modified": False,
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
        "station_wait_model": "expected_wait_mu",
        "source_ev_specs": str(DEFAULT_EV_SPECS).replace("\\", "/"),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def ensure_hanif_original_tw_converted(
    *,
    source_dir: str | Path = DEFAULT_SOURCE,
    output_dir: str | Path = DEFAULT_OUTPUT,
    ev_specs_path: str | Path = DEFAULT_EV_SPECS,
    force: bool = False,
) -> Path:
    """Convert all variant_A instances if the output tree is missing or incomplete."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if "relaxed" in str(source_dir).lower() or "variant_b" in str(source_dir).lower():
        raise ValueError(
            f"Refusing relaxed-TW source: {source_dir}. "
            "Use variant_A_original_TW only (Decision A1)."
        )
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Hanif variant_A source not found: {source_dir}")

    specs = json.loads(Path(ev_specs_path).read_text(encoding="utf-8"))
    battery = float(specs["battery_capacity"])
    rate = float(specs["energy_consumption_rate"])

    txts = sorted(source_dir.glob("*.txt"))
    txts = [p for p in txts if not p.name.startswith(".")]
    if not txts:
        raise FileNotFoundError(f"No .txt instances under {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "conversion_manifest.json"
    need = []
    for txt in txts:
        inst_dir = output_dir / txt.stem
        marker = inst_dir / "evrp_instance.json"
        if force or not marker.is_file():
            need.append(txt)
        else:
            meta = json.loads(marker.read_text(encoding="utf-8")).get("metadata") or {}
            if str(meta.get("source_variant", "")).lower().find("relaxed") >= 0:
                need.append(txt)
            elif bool(meta.get("time_windows_modified", False)):
                need.append(txt)

    t0 = time.time()
    summaries = []
    for i, txt in enumerate(need):
        LOGGER.info("Converting Hanif original-TW [%d/%d] %s", i + 1, len(need), txt.stem)
        summaries.append(
            convert_one_instance(
                txt, output_dir,
                battery_capacity=battery,
                energy_consumption_rate=rate,
            )
        )

    # Always refresh manifest for full set presence.
    names = sorted(p.stem for p in txts)
    missing = [n for n in names if not (output_dir / n / "evrp_instance.json").is_file()]
    if missing:
        raise RuntimeError(f"Hanif conversion incomplete; missing: {missing[:10]}")

    manifest = {
        "source": str(source_dir).replace("\\", "/"),
        "output": str(output_dir).replace("\\", "/"),
        "variant": SOURCE_VARIANT,
        "time_windows_modified": False,
        "num_instances": len(names),
        "converted_this_call": len(need),
        "elapsed_seconds": round(time.time() - t0, 3),
        "instances": names,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Hanif original-TW ready | %d instances under %s (converted_now=%d)",
        len(names), output_dir, len(need),
    )
    return output_dir


__all__ = [
    "DEFAULT_OUTPUT",
    "DEFAULT_SOURCE",
    "SOURCE_VARIANT",
    "convert_one_instance",
    "ensure_hanif_original_tw_converted",
]
