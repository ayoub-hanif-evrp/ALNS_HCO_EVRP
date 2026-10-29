"""Convert official Schneider E-VRPTW .txt instances to CARE-FI folders.

Source format: Schneider, Stenger, Goeke (2014). Files are the 56 large
``*_21`` instances (100 customers, 21 charging stations) plus optional small
C5/C10/C15 instances. Euclidean distance; travel time = distance / v;
energy = distance * r. Homogeneous charging rate 1/g.

Neutral schema attrs only (not heterogeneous evidence):
- charging_rate = 1/g (homogeneous)
- price_mu=0, price_sigma=0 → E[price]=1 (uniform neutral)
- wait columns omitted so expected wait stays disabled (=0)
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np

# Source .txt files live beside the converter output under evrptw_instances/;
# converted folders are written to dataset/Schneider/converted/.
DEFAULT_RAW = Path("dataset/Schneider/evrptw_instances")
DEFAULT_CONVERTED = Path("dataset/Schneider/converted")

LARGE_56 = tuple(
    [f"c10{i}_21" for i in range(1, 10)]
    + [f"c20{i}_21" for i in range(1, 9)]
    + [f"r10{i}_21" for i in range(1, 10)]
    + [f"r1{i}_21" for i in range(10, 13)]
    + [f"r20{i}_21" for i in range(1, 10)]
    + [f"r2{i}_21" for i in range(10, 12)]
    + [f"rc10{i}_21" for i in range(1, 9)]
    + [f"rc20{i}_21" for i in range(1, 9)]
)

_PARAM = re.compile(
    r"^(?P<key>[QCrgv])\s+.*/(?P<val>[-+]?\d+(?:\.\d+)?)/",
    re.IGNORECASE,
)
_FAMILY = re.compile(r"^(?P<cls>rc|r|c)(?P<kind>[12])", re.IGNORECASE)


def schneider_family(name: str) -> str:
    """Map ``c101_21`` → ``C1``, ``rc208`` → ``RC2``."""
    stem = Path(name).stem.lower().replace("_21", "")
    m = _FAMILY.match(stem)
    if not m:
        return "UNK"
    cls = {"c": "C", "r": "R", "rc": "RC"}[m.group("cls")]
    return f"{cls}{m.group('kind')}"


def parse_schneider_txt(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8").splitlines()
    depot = None
    customers: list[dict] = []
    stations: list[dict] = []
    params: dict[str, float] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        m = _PARAM.match(line)
        if m:
            params[m.group("key").lower()] = float(m.group("val"))
            continue
        parts = line.split()
        if len(parts) < 8 or parts[0].lower() in {"stringid", "stringid"}:
            continue
        sid, ntype = parts[0], parts[1].lower()
        if ntype not in {"d", "f", "c"}:
            continue
        row = {
            "string_id": sid,
            "x": float(parts[2]),
            "y": float(parts[3]),
            "demand": float(parts[4]),
            "ready": float(parts[5]),
            "due": float(parts[6]),
            "service": float(parts[7]),
        }
        if ntype == "d":
            depot = row
        elif ntype == "c":
            customers.append(row)
        else:
            stations.append(row)
    missing = [k for k in ("q", "c", "r", "g", "v") if k not in params]
    if depot is None:
        raise ValueError(f"No depot in {path}")
    if missing:
        raise ValueError(f"Missing vehicle params {missing} in {path}")
    if not customers:
        raise ValueError(f"No customers in {path}")
    return {
        "depot": depot,
        "customers": customers,
        "stations": stations,
        "Q": params["q"],
        "C": params["c"],
        "r": params["r"],
        "g": params["g"],
        "v": params["v"],
    }


def _euclidean_matrices(
    xs: np.ndarray, ys: np.ndarray, *, speed: float, energy_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    travel = dist / float(speed)
    energy = dist * float(energy_rate)
    return (
        dist.astype(np.float32),
        travel.astype(np.float32),
        energy.astype(np.float32),
    )


def convert_one_txt(txt_path: Path, output_dir: Path) -> dict:
    parsed = parse_schneider_txt(txt_path)
    name = txt_path.stem
    depot = parsed["depot"]
    customers = parsed["customers"]
    stations = parsed["stations"]
    nodes_raw = [depot] + customers + stations
    n = len(nodes_raw)
    xs = np.array([r["x"] for r in nodes_raw], dtype=np.float64)
    ys = np.array([r["y"] for r in nodes_raw], dtype=np.float64)
    max_abs = float(max(np.max(np.abs(xs)), np.max(np.abs(ys)), 1.0))
    horizon = float(depot["due"])
    load_cap = float(parsed["C"])
    speed = float(parsed["v"])
    energy_rate = float(parsed["r"])
    g = float(parsed["g"])
    customer_indices = list(range(1, 1 + len(customers)))
    station_indices = list(range(1 + len(customers), n))
    rechargeable = [0] + station_indices
    dist, travel, energy = _euclidean_matrices(
        xs, ys, speed=speed, energy_rate=energy_rate,
    )

    out_dir = output_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    node_rows = []
    nodes_json = []
    for i, r in enumerate(nodes_raw):
        is_depot = i == 0
        is_customer = i in set(customer_indices)
        is_station = i in set(station_indices)
        if is_depot:
            ntype = "depot"
        elif is_customer:
            ntype = "customer"
        else:
            ntype = "charging_station"
        row = {
            "id": i,
            "string_id": r["string_id"],
            "node_type": ntype,
            "x": float(r["x"]),
            "y": float(r["y"]),
            "demand": float(r["demand"]),
            "ready_time": float(r["ready"]),
            "due_time": float(r["due"]),
            "service_time": float(r["service"]),
            "is_depot": is_depot,
            "is_customer": is_customer,
            "is_charging_station": is_station,
            "x_norm": float(r["x"]) / max_abs,
            "y_norm": float(r["y"]) / max_abs,
            "demand_norm": float(r["demand"]) / load_cap if load_cap > 0 else 0.0,
            "ready_time_norm": float(r["ready"]) / horizon if horizon > 0 else 0.0,
            "due_time_norm": float(r["due"]) / horizon if horizon > 0 else 0.0,
            "service_time_norm": float(r["service"]) / horizon if horizon > 0 else 0.0,
        }
        node_rows.append(row)
        nodes_json.append(row)

    with (out_dir / "nodes.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(node_rows[0].keys()))
        w.writeheader()
        w.writerows(node_rows)

    np.savez_compressed(
        out_dir / "matrices.npz",
        distance_matrix=dist,
        travel_time_matrix=travel,
        energy_matrix=energy,
    )

    # Neutral station attributes for schema compatibility only.
    # Wait columns omitted: loader leaves station_wait_expected=None → wait=0.
    # Do NOT write wait_mu=0 (lognormal E[W]=exp(0)=1 would invent wait=1).
    # price_mu=0, sigma=0 → E[price]=1 uniform. Homogeneous rate=1/g.
    homog_rate = (1.0 / g) if g > 0 else 1.0
    station_rows = []
    for sid in station_indices:
        station_rows.append({
            "node_id": int(sid),
            "hanif_station_id": int(sid),
            "charging_rate": float(homog_rate),
            "price_mu": 0.0,
            "price_sigma": 0.0,
            "expected_price_per_energy": 1.0,
            "density_score": 0.0,
        })
    with (out_dir / "stations.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(station_rows[0].keys()))
        w.writeheader()
        w.writerows(station_rows)

    meta = {
        "instance_name": name,
        "problem_type": "E-VRPTW",
        "source_dataset": "Schneider et al. (2014) E-VRPTW",
        "source_doi": "10.1287/trsc.2013.0490",
        "source_mendeley": "10.17632/h3mrm5dhxw.1",
        "family": schneider_family(name),
        "num_depots": 1,
        "num_customers": len(customers),
        "num_charging_stations": len(stations),
        "depot_has_charging": True,
        "planning_horizon": horizon,
        "max_abs_coordinate": max_abs,
        "station_wait_model": "zero",
        "station_price_model": "uniform_neutral_1",
        "heterogeneous_stations": False,
        "neutral_attrs_note": (
            "wait=0; expected_price_per_energy=1.0 uniform; rate=1/g homogeneous. "
            "Not used to claim heterogeneous price/waiting benefits."
        ),
        "distance_convention": "euclidean_unrounded; travel=distance/v; energy=distance*r",
    }
    vehicle = {
        "battery_capacity": float(parsed["Q"]),
        "load_capacity": load_cap,
        "energy_consumption_rate": energy_rate,
        "inverse_charging_rate": g,
        "speed": speed,
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
        "family": schneider_family(name),
        "num_customers": len(customers),
        "num_stations": len(stations),
        "num_nodes": n,
        "planning_horizon": horizon,
        "load_capacity": load_cap,
        "battery_capacity": float(parsed["Q"]),
        "energy_consumption_rate": energy_rate,
        "inverse_charging_rate": g,
        "speed": speed,
        "station_wait_model": "zero",
        "station_price_model": "uniform_neutral_1",
        "heterogeneous_stations": False,
        "validation_passed": True,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def ensure_schneider_converted(
    *,
    names: list[str] | None = None,
    source_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    force: bool = False,
) -> Path:
    src = Path(source_dir or DEFAULT_RAW)
    out = Path(output_dir or DEFAULT_CONVERTED)
    out.mkdir(parents=True, exist_ok=True)
    wanted = list(names) if names else list(LARGE_56)
    for stem in wanted:
        txt = src / f"{stem}.txt"
        if not txt.is_file():
            raise FileNotFoundError(f"Missing Schneider instance: {txt}")
        marker = out / stem / "evrp_instance.json"
        if marker.is_file() and not force:
            continue
        convert_one_txt(txt, out)
    return out
