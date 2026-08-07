"""Unique exact-feasible route pool for set-partitioning recombination."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.search.route_replay import replay_route


@dataclass(frozen=True)
class PooledRoute:
    """Exact per-route column (never proportional solution totals)."""

    nodes: tuple[int, ...]
    charge_labels: tuple[float | None, ...]
    customers: tuple[int, ...]
    order: tuple[int, ...]
    stations: tuple[int, ...]
    distance: float
    duration: float
    charging_time: float
    end_time: float
    feasible: bool
    source_method: str
    source_fleet_size: int

    @property
    def key(self) -> tuple:
        """Full customer order + station sequence + charge-label sequence."""
        return (self.order, self.stations, self.charge_labels)


@dataclass
class RoutePool:
    routes: dict[tuple, PooledRoute] = field(default_factory=dict)

    def add_route(
        self,
        instance,
        nodes: list[int],
        labels: list[float | None] | None,
        *,
        source_method: str,
        source_fleet_size: int,
        config: dict | None = None,
    ) -> bool:
        """Replay one route exactly and insert if feasible and new."""
        labs = list(labels) if labels is not None else [None] * len(nodes)
        res = replay_route(instance, list(nodes), labs, config=config)
        custs = tuple(int(n) for n in res.served_customers)
        if not custs:
            return False
        if not res.feasible:
            return False
        stations = tuple(int(n) for n in nodes if instance.nodes[n].is_charging_station)
        # Align charge labels to station visits in node order
        node_t = tuple(int(n) for n in (res.route_events and [e.node for e in res.route_events] or nodes))
        # Prefer normalized nodes from replay events when available
        full_nodes = []
        full_labs: list[float | None] = []
        if res.route_events:
            for ev in res.route_events:
                full_nodes.append(int(ev.node))
                full_labs.append(ev.charge_portion)
            # Ensure depot bookends
            depot = int(instance.depot_index)
            if not full_nodes or full_nodes[0] != depot:
                full_nodes = [depot] + full_nodes
                full_labs = [None] + full_labs
            if full_nodes[-1] != depot:
                full_nodes = full_nodes + [depot]
                full_labs = full_labs + [None]
        else:
            full_nodes = [int(n) for n in nodes]
            full_labs = labs

        order = tuple(n for n in full_nodes if instance.nodes[n].is_customer)
        stations = tuple(n for n in full_nodes if instance.nodes[n].is_charging_station)
        pr = PooledRoute(
            nodes=tuple(full_nodes),
            charge_labels=tuple(full_labs),
            customers=tuple(sorted(order)),
            order=order,
            stations=stations,
            distance=float(res.distance),
            duration=float(res.duration),
            charging_time=float(res.charging_time),
            end_time=float(res.end_time),
            feasible=True,
            source_method=str(source_method),
            source_fleet_size=int(source_fleet_size),
        )
        if pr.key in self.routes:
            return False
        self.routes[pr.key] = pr
        return True

    def add_from_solution(
        self,
        instance,
        solution_routes: list,
        *,
        score=None,
        source_method: str,
        source_fleet_size: int,
        config: dict | None = None,
    ) -> int:
        del score  # unused; exact per-route replay replaces proportional metrics
        added = 0
        for item in solution_routes:
            if isinstance(item, (list, tuple)) and len(item) == 2 and not isinstance(item[0], int):
                nodes, labels = item
            else:
                nodes, labels = item, None
            if self.add_route(
                instance,
                list(nodes),
                list(labels) if labels is not None else None,
                source_method=source_method,
                source_fleet_size=source_fleet_size,
                config=config,
            ):
                added += 1
        return added

    def __len__(self) -> int:
        return len(self.routes)

    def as_list(self) -> list[PooledRoute]:
        return list(self.routes.values())

    def covers_all_customers(self, customers: set[int]) -> bool:
        covered: set[int] = set()
        for r in self.routes.values():
            covered.update(r.customers)
        return customers.issubset(covered)

    def uncovered_customers(self, customers: set[int]) -> set[int]:
        covered: set[int] = set()
        for r in self.routes.values():
            covered.update(r.customers)
        return set(customers) - covered
