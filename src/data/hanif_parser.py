"""Parse Hanif et al. EVRPTW benchmark files (Homberger 1000 + charging stations).

Reference: Hanif, M.; et al. Electric Vehicle Routing with Time Windows and
Heterogeneous Charging-Station Attribute Dataset. *Data* 2026, 11, 83.
https://doi.org/10.3390/data11040083
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HanifNode:
    node_id: int
    x: float
    y: float
    demand: float
    ready_time: float
    due_time: float
    service_time: float


@dataclass(frozen=True)
class HanifStationAttributes:
    station_id: int
    charging_rate: float
    price_mu: float
    price_sigma: float
    wait_mu: float
    wait_sigma: float
    density_score: float


@dataclass(frozen=True)
class HanifParsedInstance:
    instance_name: str
    fleet_size: int
    load_capacity: float
    depot: HanifNode
    customers: list[HanifNode]
    stations: list[HanifNode]
    station_attributes: dict[int, HanifStationAttributes]


def parse_hanif_txt(txt_path: str | Path) -> tuple[int, float, HanifNode, list[HanifNode], list[HanifNode]]:
    """Parse a Hanif augmented Homberger ``.txt`` instance file."""
    lines = Path(txt_path).read_text(encoding="utf-8").splitlines()
    vehicle_line = lines[4].split()
    fleet_size = int(vehicle_line[0])
    load_capacity = float(vehicle_line[1])

    depot: HanifNode | None = None
    customers: list[HanifNode] = []
    stations: list[HanifNode] = []

    for line in lines[9:]:
        parts = line.split()
        if len(parts) < 7:
            continue
        node_id = int(parts[0])
        node = HanifNode(
            node_id=node_id,
            x=float(parts[1]),
            y=float(parts[2]),
            demand=float(parts[3]),
            ready_time=float(parts[4]),
            due_time=float(parts[5]),
            service_time=float(parts[6]),
        )
        if node_id == 0:
            depot = node
        elif node_id <= 1000:
            customers.append(node)
        else:
            stations.append(node)

    if depot is None:
        raise ValueError(f"No depot (node 0) found in {txt_path}")
    if not customers:
        raise ValueError(f"No customers found in {txt_path}")
    if not stations:
        raise ValueError(f"No charging stations found in {txt_path}")

    return fleet_size, load_capacity, depot, customers, stations


def load_station_attributes(csv_path: str | Path) -> dict[int, HanifStationAttributes]:
    """Load full Hanif station attributes from a ``*_stations.csv`` file."""
    attrs: dict[int, HanifStationAttributes] = {}
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["station_id"])
            attrs[sid] = HanifStationAttributes(
                station_id=sid,
                charging_rate=float(row["charging_rate"]),
                price_mu=float(row["price_mu"]),
                price_sigma=float(row["price_sigma"]),
                wait_mu=float(row["wait_mu"]),
                wait_sigma=float(row["wait_sigma"]),
                density_score=float(row["density_score"]),
            )
    return attrs


def load_station_rates(csv_path: str | Path) -> dict[int, float]:
    """Load ``station_id -> charging_rate`` from a ``*_stations.csv`` file."""
    return {sid: a.charging_rate for sid, a in load_station_attributes(csv_path).items()}


def parse_hanif_instance(txt_path: str | Path, stations_csv: str | Path | None = None) -> HanifParsedInstance:
    """Parse ``.txt`` plus optional ``*_stations.csv`` into a structured instance."""
    txt_path = Path(txt_path)
    instance_name = txt_path.stem
    fleet_size, load_capacity, depot, customers, stations = parse_hanif_txt(txt_path)

    csv_path = Path(stations_csv) if stations_csv else txt_path.with_name(f"{instance_name}_stations.csv")
    station_attributes = load_station_attributes(csv_path) if csv_path.exists() else {}

    return HanifParsedInstance(
        instance_name=instance_name,
        fleet_size=fleet_size,
        load_capacity=load_capacity,
        depot=depot,
        customers=customers,
        stations=stations,
        station_attributes=station_attributes,
    )
