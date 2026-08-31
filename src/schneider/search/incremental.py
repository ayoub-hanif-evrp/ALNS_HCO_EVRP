"""Cheap necessary-condition filters + counters (never declares feasible alone)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EvalCounters:
    moves_generated: int = 0
    capacity_precheck_rejections: int = 0
    time_precheck_rejections: int = 0
    energy_precheck_rejections: int = 0
    charging_dp_calls: int = 0
    exact_route_replays: int = 0
    solution_replays: int = 0
    accepted_moves: int = 0

    def as_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass
class RouteLabels:
    cumulative_load: float = 0.0
    earliest_service: float = 0.0
    latest_service: float = 1e18
    time_slack: float = 1e18
    distance: float = 0.0
    min_required_battery: float = 0.0
    depot_return_energy: float = 0.0


def capacity_precheck(instance, route_nodes: list[int], customer: int) -> bool:
    """True if insertion is NOT provably capacity-infeasible."""
    load = sum(float(instance.nodes[n].demand) for n in route_nodes if instance.nodes[n].is_customer)
    veh = instance.vehicle
    cap = float(getattr(veh, "capacity", None) or getattr(veh, "load_capacity", None) or getattr(veh, "max_load", 0) or 0)
    return load + float(instance.nodes[customer].demand) <= cap + 1e-9


def energy_precheck(instance, from_node: int, to_node: int) -> bool:
    """True if hop is not provably battery-impossible from a full battery."""
    need = float(instance.energy_matrix[from_node, to_node])
    return need <= float(instance.vehicle.battery_capacity) + 1e-9


def time_precheck(instance, from_node: int, to_node: int, depart_time: float) -> bool:
    """True if arriving before due is not impossible ignoring waits."""
    travel = float(instance.travel_time_matrix[from_node, to_node])
    arrive = depart_time + travel
    return arrive <= float(instance.nodes[to_node].due_time) + 1e-6
