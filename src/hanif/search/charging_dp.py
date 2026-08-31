"""Genuine resource-label charging optimizer for a fixed customer sequence.

Replaces nearest-station / fixed-fraction battery heuristics with label-setting DP:
  state = (customers_served, physical_node, time, battery, distance,
           charging_time, station_visits, station_streak)
Dominance retains a label only when it is no worse in time, battery,
distance, charging time, and station visits.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.hanif.search.deadline import SearchDeadline


@dataclass
class ChargingPlanResult:
    feasible: bool
    station_visits: list
    charge_amounts: list
    arrival_times: list
    service_times: list
    battery_profile: list
    route_distance: float
    charging_time: float
    failure_reason: str | None = None
    nodes: list[int] = field(default_factory=list)
    labels: list[float | None] = field(default_factory=list)


def _portions(config: dict | None) -> list[float]:
    problem = (config or {}).get("problem") or {}
    return [float(p) for p in (problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0])]


@dataclass
class _Label:
    customers_served: int
    node: int
    time: float
    battery: float
    distance: float
    charging_time: float
    station_visits: int
    station_streak: int
    nodes: tuple[int, ...]
    charge_labels: tuple[float | None, ...]

    @property
    def sort_key(self) -> tuple:
        return (self.distance, self.charging_time, self.time, self.station_visits)

    def dominates(self, other: "_Label") -> bool:
        if self.customers_served != other.customers_served or self.node != other.node:
            return False
        leq = (
            self.time <= other.time + 1e-9
            and self.battery + 1e-9 >= other.battery
            and self.distance <= other.distance + 1e-9
            and self.charging_time <= other.charging_time + 1e-9
            and self.station_visits <= other.station_visits
        )
        if not leq:
            return False
        strict = (
            self.time < other.time - 1e-9
            or self.battery > other.battery + 1e-9
            or self.distance < other.distance - 1e-9
            or self.charging_time < other.charging_time - 1e-9
            or self.station_visits < other.station_visits
        )
        return strict


def _charge_rate(instance, node: int) -> float:
    rates = getattr(instance, "station_charging_rate", None)
    if rates is not None and node < len(rates) and float(rates[node]) > 0:
        return float(rates[node])
    inv = float(instance.vehicle.inverse_charging_rate)
    return 1.0 / max(inv, 1e-12)


def _travel(instance, i: int, j: int) -> tuple[float, float, float]:
    return (
        float(instance.distance_matrix[i, j]),
        float(instance.travel_time_matrix[i, j]),
        float(instance.energy_matrix[i, j]),
    )


def _serve_customer(instance, node: int, arrive: float) -> tuple[float, float] | None:
    """Return (service_start, depart) or None if TW hard-infeasible."""
    n = instance.nodes[node]
    start = max(arrive, float(n.ready_time))
    if start > float(n.due_time) + 1e-9:
        return None
    return start, start + float(n.service_time)


def _insert_nondominated(bucket: list[_Label], lab: _Label, *, max_labels: int = 40) -> bool:
    for existing in bucket:
        if existing.dominates(lab):
            return False
    bucket[:] = [e for e in bucket if not lab.dominates(e)]
    bucket.append(lab)
    if len(bucket) > max_labels:
        bucket.sort(key=lambda x: (x.time, -x.battery, x.distance, x.charging_time, x.station_visits))
        del bucket[max_labels:]
    return True


def optimize_charging_plan(
    instance,
    customer_sequence: list[int],
    *,
    config: dict | None = None,
    m: int = 1,
    deadline: SearchDeadline | None = None,
    max_station_streak: int = 2,
    battery_buckets: int = 20,
) -> ChargingPlanResult:
    """Label-setting charging optimization for a fixed customer order."""
    del m  # single-route optimizer
    depot = int(instance.depot_index)
    portions = _portions(config)
    stations = [i for i, n in enumerate(instance.nodes) if n.is_charging_station]
    # Optional depot charging as a station-like recharge node
    recharge_nodes = list(stations)
    use_depot = bool((config or {}).get("problem", {}).get("use_depot_charging", True)) and bool(
        getattr(instance, "depot_has_charging", False)
    )
    if use_depot and depot not in recharge_nodes:
        recharge_nodes = [depot] + recharge_nodes

    custs = [int(c) for c in customer_sequence if instance.nodes[int(c)].is_customer]
    batt_cap = float(instance.vehicle.battery_capacity)
    horizon = float(getattr(instance, "planning_horizon", 1e9) or 1e9)

    if not custs:
        nodes = [depot, depot]
        return ChargingPlanResult(
            feasible=True,
            station_visits=[],
            charge_amounts=[],
            arrival_times=[0.0, 0.0],
            service_times=[0.0, 0.0],
            battery_profile=[batt_cap, batt_cap],
            route_distance=0.0,
            charging_time=0.0,
            nodes=nodes,
            labels=[None, None],
        )

    # open[k][node] -> nondominated labels after serving k customers, located at node
    open_labels: dict[tuple[int, int], list[_Label]] = {}
    start = _Label(
        customers_served=0,
        node=depot,
        time=0.0,
        battery=batt_cap,
        distance=0.0,
        charging_time=0.0,
        station_visits=0,
        station_streak=0,
        nodes=(depot,),
        charge_labels=(None,),
    )
    open_labels[(0, depot)] = [start]
    queue: list[tuple[int, int]] = [(0, depot)]
    queued = {(0, depot)}
    expansions = 0
    max_expansions = 15000 if bool((config or {}).get("problem", {}).get("greedy_min_charge", False)) else 5000

    best_feasible: _Label | None = None

    def _bucket_key(lab: _Label) -> tuple[int, int]:
        return (lab.customers_served, lab.node)

    while queue and expansions < max_expansions:
        if deadline is not None and deadline.expired():
            break
        key = queue.pop(0)
        queued.discard(key)
        for lab in list(open_labels.get(key, [])):
            expansions += 1
            if deadline is not None and deadline.expired():
                break
            if expansions > max_expansions:
                break
            k, cur = lab.customers_served, lab.node

            # --- Return to depot when all customers served ---
            if k == len(custs):
                d, t, e = _travel(instance, cur, depot)
                if e <= lab.battery + 1e-9:
                    arr = lab.time + t
                    if arr <= horizon + 1e-9:
                        fin = _Label(
                            customers_served=k,
                            node=depot,
                            time=arr,
                            battery=lab.battery - e,
                            distance=lab.distance + d,
                            charging_time=lab.charging_time,
                            station_visits=lab.station_visits,
                            station_streak=0,
                            nodes=lab.nodes + (depot,),
                            charge_labels=lab.charge_labels + (None,),
                        )
                        if best_feasible is None or fin.sort_key < best_feasible.sort_key:
                            best_feasible = fin
                continue

            nxt = custs[k]

            # --- Direct travel to next customer ---
            d, t, e = _travel(instance, cur, nxt)
            if e <= lab.battery + 1e-9:
                arr = lab.time + t
                served = _serve_customer(instance, nxt, arr)
                if served is not None:
                    _start, depart = served
                    child = _Label(
                        customers_served=k + 1,
                        node=nxt,
                        time=depart,
                        battery=lab.battery - e,
                        distance=lab.distance + d,
                        charging_time=lab.charging_time,
                        station_visits=lab.station_visits,
                        station_streak=0,
                        nodes=lab.nodes + (nxt,),
                        charge_labels=lab.charge_labels + (None,),
                    )
                    bkey = _bucket_key(child)
                    bucket = open_labels.setdefault(bkey, [])
                    if _insert_nondominated(bucket, child) and bkey not in queued:
                        queue.append(bkey)
                        queued.add(bkey)

            # --- Travel to a station and charge (any legal portion) ---
            if lab.station_streak < max_station_streak:
                pool = recharge_nodes
                if len(recharge_nodes) > 10:
                    pool = sorted(
                        recharge_nodes,
                        key=lambda s: float(instance.distance_matrix[cur, s])
                        + float(instance.distance_matrix[s, nxt]),
                    )[:8]
                    if depot in recharge_nodes and depot not in pool:
                        pool = [depot] + pool
                for s in pool:
                    if deadline is not None and deadline.expired():
                        break
                    # Avoid immediate identical station loop without progress
                    d, t, e = _travel(instance, cur, s)
                    if e > lab.battery + 1e-9:
                        continue
                    arr = lab.time + t
                    if arr > horizon + 1e-9:
                        continue
                    batt_arr = lab.battery - e
                    missing = max(0.0, batt_cap - batt_arr)
                    if missing <= 1e-12:
                        # Already full: only useful if we needed the node as waypoint (skip)
                        continue
                    amounts: list[tuple[float, float]] = []
                    for p in portions:
                        amounts.append((float(p) * missing, float(p)))
                    if bool((config or {}).get("problem", {}).get("greedy_min_charge", False)):
                        from src.hanif.search.schneider_boost import just_enough_charge

                        need = just_enough_charge(instance, s, batt_arr, nxt)
                        if need > 1e-12:
                            amounts.append((need, need / missing))
                    seen_chg: set[float] = set()
                    unique_amounts: list[tuple[float, float]] = []
                    for charged, p in amounts:
                        key = round(charged, 6)
                        if key in seen_chg or charged <= 1e-12:
                            continue
                        seen_chg.add(key)
                        unique_amounts.append((charged, p))
                    for charged, p in unique_amounts:
                        rate = _charge_rate(instance, s)
                        ctime = charged / max(rate, 1e-12)
                        depart = arr + ctime
                        if depart > horizon + 1e-9:
                            continue
                        is_station_visit = instance.nodes[s].is_charging_station or (
                            s == depot and use_depot
                        )
                        child = _Label(
                            customers_served=k,
                            node=s,
                            time=depart,
                            battery=min(batt_cap, batt_arr + charged),
                            distance=lab.distance + d,
                            charging_time=lab.charging_time + ctime,
                            station_visits=lab.station_visits + (1 if is_station_visit else 0),
                            station_streak=lab.station_streak + 1,
                            nodes=lab.nodes + (s,),
                            charge_labels=lab.charge_labels + (float(p),),
                        )
                        bkey = _bucket_key(child)
                        bucket = open_labels.setdefault(bkey, [])
                        if _insert_nondominated(bucket, child) and bkey not in queued:
                            queue.append(bkey)
                            queued.add(bkey)

    if best_feasible is None:
        # Fallback: try direct depot-customers-depot via exact replay for diagnostics
        from src.hanif.search.route_replay import replay_route

        nodes0 = [depot] + custs + [depot]
        labels0 = [None] * len(nodes0)
        res = replay_route(instance, nodes0, labels0, config=config)
        return ChargingPlanResult(
            feasible=False,
            station_visits=[],
            charge_amounts=[],
            arrival_times=[],
            service_times=[],
            battery_profile=[],
            route_distance=float(getattr(res, "distance", 0.0) or 0.0),
            charging_time=0.0,
            failure_reason=getattr(res, "first_failure", None) or "no_feasible_label",
            nodes=nodes0,
            labels=labels0,
        )

    nodes = list(best_feasible.nodes)
    labs = list(best_feasible.charge_labels)
    st_visits = [n for n in nodes if instance.nodes[n].is_charging_station]
    charges = [lab for n, lab in zip(nodes, labs) if instance.nodes[n].is_charging_station]
    return ChargingPlanResult(
        feasible=True,
        station_visits=st_visits,
        charge_amounts=[float(c) if c is not None else 1.0 for c in charges],
        arrival_times=[],
        service_times=[],
        battery_profile=[],
        route_distance=float(best_feasible.distance),
        charging_time=float(best_feasible.charging_time),
        nodes=nodes,
        labels=labs,
    )


def enumerate_charging_plans_exhaustive(
    instance,
    customer_sequence: list[int],
    *,
    config: dict | None = None,
    max_station_streak: int = 1,
) -> list[ChargingPlanResult]:
    """Enumerate station/portion insertions for tiny fixtures (bounded)."""
    from itertools import product

    from src.hanif.search.route_replay import replay_route

    depot = int(instance.depot_index)
    portions = _portions(config)
    stations = [i for i, n in enumerate(instance.nodes) if n.is_charging_station]
    custs = [int(c) for c in customer_sequence if instance.nodes[int(c)].is_customer]
    results: list[ChargingPlanResult] = []

    def _eval(nodes: list[int], labels: list[float | None]) -> ChargingPlanResult:
        res = replay_route(instance, nodes, labels, config=config)
        st = [n for n in nodes if instance.nodes[n].is_charging_station]
        ch = [lab for n, lab in zip(nodes, labels) if instance.nodes[n].is_charging_station]
        return ChargingPlanResult(
            feasible=bool(res.feasible),
            station_visits=st,
            charge_amounts=[float(c) if c is not None else 1.0 for c in ch],
            arrival_times=[],
            service_times=[],
            battery_profile=[],
            route_distance=float(getattr(res, "distance", 0.0) or 0.0),
            charging_time=float(getattr(res, "charging_time", 0.0) or 0.0),
            failure_reason=None if res.feasible else (getattr(res, "first_failure", None) or "infeasible"),
            nodes=list(nodes),
            labels=list(labels),
        )

    # No stations
    results.append(_eval([depot] + custs + [depot], [None] * (len(custs) + 2)))

    # Insert at most one station in each gap with each portion (tiny enumeration)
    gaps = len(custs) + 1
    for gap in range(gaps):
        for s in stations:
            for p in portions:
                nodes = [depot]
                labels: list[float | None] = [None]
                for gi, c in enumerate(custs):
                    if gi == gap:
                        nodes += [s]
                        labels += [float(p)]
                    nodes += [c]
                    labels += [None]
                if gap == len(custs):
                    nodes += [s]
                    labels += [float(p)]
                nodes += [depot]
                labels += [None]
                results.append(_eval(nodes, labels))
    return results


def exhaustive_optimum(
    instance,
    customer_sequence: list[int],
    *,
    config: dict | None = None,
) -> ChargingPlanResult | None:
    plans = enumerate_charging_plans_exhaustive(instance, customer_sequence, config=config)
    feas = [p for p in plans if p.feasible]
    if not feas:
        return None
    feas.sort(key=lambda p: (p.route_distance, p.charging_time, len(p.station_visits)))
    return feas[0]
