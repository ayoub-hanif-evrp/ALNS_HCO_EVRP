"""Synthetic fixtures for Deterministic-FPS unit tests (no Schneider dependency)."""
from __future__ import annotations

import numpy as np

from src.schneider.data.instance_loader import EVRPTWInstance, Node, VehicleParams


def _node(
    i: int,
    sid: str,
    typ: str,
    x: float,
    y: float,
    demand: float,
    ready: float,
    due: float,
    service: float,
    *,
    load_cap: float,
    horizon: float,
    max_abs: float,
) -> Node:
    return Node(
        id=i,
        string_id=sid,
        node_type=typ,
        x=x,
        y=y,
        demand=demand,
        ready_time=ready,
        due_time=due,
        service_time=service,
        is_depot=(typ == "depot"),
        is_customer=(typ == "customer"),
        is_charging_station=(typ == "charging_station"),
        x_norm=x / max_abs,
        y_norm=y / max_abs,
        demand_norm=demand / max(load_cap, 1e-9),
        ready_time_norm=ready / max(horizon, 1e-9),
        due_time_norm=due / max(horizon, 1e-9),
        service_time_norm=service / max(horizon, 1e-9),
    )


def build_tiny_evrptw(
    *,
    name: str = "tiny_synth",
    customers: list[tuple[float, float, float, float, float, float]] | None = None,
    stations: list[tuple[float, float]] | None = None,
    battery_capacity: float = 100.0,
    load_capacity: float = 50.0,
    energy_rate: float = 1.0,
    speed: float = 1.0,
    inverse_charging_rate: float = 1.0,
    horizon: float = 1000.0,
    depot_has_charging: bool = True,
) -> EVRPTWInstance:
    """Build a tiny in-memory instance.

    customers: list of (x, y, demand, ready, due, service)
    stations: list of (x, y)
    """
    if customers is None:
        customers = [
            (10.0, 0.0, 10.0, 0.0, 500.0, 10.0),
            (20.0, 0.0, 10.0, 0.0, 500.0, 10.0),
        ]
    if stations is None:
        stations = [(15.0, 5.0)]

    coords = [(0.0, 0.0)] + [(c[0], c[1]) for c in customers] + list(stations)
    max_abs = max(max(abs(x), abs(y)) for x, y in coords) or 1.0

    nodes: list[Node] = [
        _node(
            0, "D0", "depot", 0.0, 0.0, 0.0, 0.0, horizon, 0.0,
            load_cap=load_capacity, horizon=horizon, max_abs=max_abs,
        )
    ]
    cust_idx: list[int] = []
    for k, (x, y, dem, ready, due, svc) in enumerate(customers, start=1):
        nodes.append(
            _node(
                k, f"C{k}", "customer", x, y, dem, ready, due, svc,
                load_cap=load_capacity, horizon=horizon, max_abs=max_abs,
            )
        )
        cust_idx.append(k)
    st_idx: list[int] = []
    for j, (x, y) in enumerate(stations):
        i = len(nodes)
        nodes.append(
            _node(
                i, f"S{j}", "charging_station", x, y, 0.0, 0.0, horizon, 0.0,
                load_cap=load_capacity, horizon=horizon, max_abs=max_abs,
            )
        )
        st_idx.append(i)

    n = len(nodes)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            dx = nodes[i].x - nodes[j].x
            dy = nodes[i].y - nodes[j].y
            dist[i, j] = float(np.hypot(dx, dy))
    travel = dist / max(speed, 1e-9)
    energy = dist * energy_rate

    veh = VehicleParams(
        battery_capacity=float(battery_capacity),
        load_capacity=float(load_capacity),
        energy_consumption_rate=float(energy_rate),
        inverse_charging_rate=float(inverse_charging_rate),
        speed=float(speed),
    )
    rechargeable = [0] + list(st_idx) if depot_has_charging else list(st_idx)
    inst = EVRPTWInstance(
        instance_name=name,
        nodes=nodes,
        depot_index=0,
        customer_indices=cust_idx,
        charging_station_indices=st_idx,
        rechargeable_indices=rechargeable,
        distance_matrix=dist,
        travel_time_matrix=travel,
        energy_matrix=energy,
        vehicle=veh,
        planning_horizon=float(horizon),
        depot_has_charging=depot_has_charging,
        max_abs_coordinate=float(max_abs),
    )
    if hasattr(inst, "ensure_hot_arrays"):
        inst.ensure_hot_arrays()
    return inst


def build_energy_tight_tiny() -> EVRPTWInstance:
    """Long hops that typically need station visits under a tight battery."""
    return build_tiny_evrptw(
        name="energy_tight_tiny",
        customers=[
            (40.0, 0.0, 5.0, 0.0, 800.0, 5.0),
            (80.0, 0.0, 5.0, 0.0, 800.0, 5.0),
        ],
        stations=[(40.0, 0.0), (60.0, 5.0)],
        battery_capacity=50.0,
        load_capacity=30.0,
        energy_rate=1.0,
        speed=1.0,
        inverse_charging_rate=0.5,
        horizon=1000.0,
        depot_has_charging=True,
    )
