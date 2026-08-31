"""Authoritative sequential EVRPTW replay for the Hanif CARE-FI path.

Implements the paper equations (battery, travel, time windows, expected
station wait, charge amount). This is the only feasibility function used
to accept Stage-1 and Stage-2 moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.hanif.model import charging_rate, expected_price, expected_wait

EPS = 1e-9


@dataclass
class RouteMetrics:
    feasible: bool
    failure: str | None
    customers: list[int]
    distance: float
    duration: float
    charging_time: float
    station_waiting: float
    customer_waiting: float
    charged_energy: float
    charging_expense: float
    charging_visits: int
    hard_violations: int = 0
    nodes: list[int] = field(default_factory=list)
    labels: list[float | None] = field(default_factory=list)


@dataclass
class SolutionMetrics:
    feasible: bool
    customers: list[int]
    unserved: int
    duplicates: int
    used_k: int
    distance: float
    duration: float
    charging_time: float
    station_waiting: float
    charged_energy: float
    charging_expense: float
    charging_visits: int
    customer_waiting: float
    hard_violations: int
    routes: list[RouteMetrics] = field(default_factory=list)
    failure: str | None = None


def _normalize(instance, nodes: list[int], labels: list[float | None] | None):
    depot = int(instance.depot_index)
    seq = [int(n) for n in nodes]
    if not seq:
        seq = [depot, depot]
    if seq[0] != depot:
        seq = [depot] + seq
    if seq[-1] != depot:
        seq = seq + [depot]
    labs = list(labels) if labels is not None else [None] * len(seq)
    if len(labs) < len(seq):
        labs = labs + [None] * (len(seq) - len(labs))
    return seq, labs[: len(seq)]


def replay_route(
    instance,
    nodes: list[int],
    labels: list[float | None] | None = None,
    *,
    default_portion: float = 1.0,
) -> RouteMetrics:
    seq, labs = _normalize(instance, nodes, labels)
    depot = int(instance.depot_index)
    batt_cap = float(instance.vehicle.battery_capacity)
    load_cap = float(instance.vehicle.load_capacity)
    horizon = float(getattr(instance, "planning_horizon", 1e18) or 1e18)
    dist_m = instance.distance_matrix
    time_m = instance.travel_time_matrix
    energy_m = instance.energy_matrix

    t = 0.0
    batt = batt_cap
    load = 0.0
    distance = 0.0
    chg_time = 0.0
    st_wait = 0.0
    cust_wait = 0.0
    energy = 0.0
    expense = 0.0
    visits = 0
    customers: list[int] = []
    hard = 0
    failure = None

    for i in range(1, len(seq)):
        prev, cur = seq[i - 1], seq[i]
        node = instance.nodes[cur]
        d = float(dist_m[prev, cur])
        tau = float(time_m[prev, cur])
        e_arc = float(energy_m[prev, cur])
        distance += d
        t += tau
        batt -= e_arc
        if batt < -EPS:
            failure = f"battery_arrival node={cur}"
            hard += 1
            break

        if node.is_customer:
            start = max(t, float(node.ready_time))
            if start > float(node.due_time) + EPS:
                failure = f"time_window node={cur}"
                hard += 1
                break
            cust_wait += start - t
            t = start + float(node.service_time)
            load += float(node.demand)
            if load > load_cap + EPS:
                failure = f"capacity node={cur}"
                hard += 1
                break
            customers.append(cur)
        elif node.is_charging_station or (node.is_depot and i < len(seq) - 1 and getattr(instance, "depot_has_charging", False)):
            wait = expected_wait(instance, cur) if node.is_charging_station else 0.0
            portion = labs[i]
            alpha = float(default_portion if portion is None else portion)
            alpha = min(1.0, max(0.0, alpha))
            missing = max(0.0, batt_cap - batt)
            charged = alpha * missing
            rate = charging_rate(instance, cur)
            ctime = charged / max(rate, EPS)
            t = t + wait + ctime
            batt = min(batt_cap, batt + charged)
            if charged > EPS:
                chg_time += ctime
                st_wait += wait
                energy += charged
                expense += charged * expected_price(instance, cur)
                visits += 1
        elif node.is_depot and i == len(seq) - 1:
            if t > horizon + EPS:
                failure = "depot_horizon"
                hard += 1
                break
        else:
            failure = f"illegal_node node={cur}"
            hard += 1
            break

    feasible = failure is None and hard == 0
    return RouteMetrics(
        feasible=feasible,
        failure=failure,
        customers=customers,
        distance=distance,
        duration=t,
        charging_time=chg_time,
        station_waiting=st_wait,
        customer_waiting=cust_wait,
        charged_energy=energy,
        charging_expense=expense,
        charging_visits=visits,
        hard_violations=hard,
        nodes=seq,
        labels=labs,
    )


def replay_solution(instance, routes: list) -> SolutionMetrics:
    seen: set[int] = set()
    duplicates = 0
    metrics: list[RouteMetrics] = []
    used = 0
    for item in routes:
        if isinstance(item, (list, tuple)) and item and not isinstance(item[0], int):
            nodes, labels = item[0], item[1]
        else:
            nodes, labels = item, None
        rm = replay_route(instance, list(nodes), list(labels) if labels is not None else None)
        metrics.append(rm)
        if rm.customers:
            used += 1
        for c in rm.customers:
            if c in seen:
                duplicates += 1
            seen.add(c)
    want = {i for i, n in enumerate(instance.nodes) if n.is_customer}
    missing = want - seen
    hard = sum(m.hard_violations for m in metrics)
    feasible = (
        hard == 0
        and duplicates == 0
        and not missing
        and all(m.feasible or not m.customers for m in metrics)
        and all(m.feasible for m in metrics if m.customers)
    )
    failure = None
    if missing:
        failure = f"unserved={len(missing)}"
    elif duplicates:
        failure = "duplicates"
    elif hard:
        failure = next((m.failure for m in metrics if m.failure), "hard")
    return SolutionMetrics(
        feasible=bool(feasible),
        customers=sorted(seen),
        unserved=len(missing),
        duplicates=duplicates,
        used_k=used or (1 if routes else 0),
        distance=sum(m.distance for m in metrics),
        duration=sum(m.duration for m in metrics),
        charging_time=sum(m.charging_time for m in metrics),
        station_waiting=sum(m.station_waiting for m in metrics),
        charged_energy=sum(m.charged_energy for m in metrics),
        charging_expense=sum(m.charging_expense for m in metrics),
        charging_visits=sum(m.charging_visits for m in metrics),
        customer_waiting=sum(m.customer_waiting for m in metrics),
        hard_violations=hard,
        routes=metrics,
        failure=failure,
    )


def customer_sequence(instance, nodes: list[int]) -> tuple[int, ...]:
    return tuple(int(n) for n in nodes if instance.nodes[int(n)].is_customer)


def sequences_match(instance, a: list, b: list) -> bool:
    sa = [customer_sequence(instance, nodes) for nodes, _ in a if customer_sequence(instance, nodes)]
    sb = [customer_sequence(instance, nodes) for nodes, _ in b if customer_sequence(instance, nodes)]
    return sa == sb
