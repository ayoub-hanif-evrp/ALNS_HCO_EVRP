"""Deterministic Stage-1 construction: due-time order, restricted insertion."""
from __future__ import annotations

import time

from src.hanif.replay import replay_route

DEFAULTS = {
    "nearby_routes": 8,
    "insert_positions": 3,
    "nearby_stations": 3,
}


def customer_order(instance) -> list[int]:
    custs = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    custs.sort(key=lambda i: (
        float(instance.nodes[i].due_time),
        float(instance.nodes[i].ready_time),
        -float(instance.nodes[i].demand),
        i,
    ))
    return custs


def _load(instance, nodes: list[int]) -> float:
    return sum(float(instance.nodes[n].demand) for n in nodes if instance.nodes[n].is_customer)


def _centroid(instance, nodes: list[int]) -> tuple[float, float]:
    xs, ys = [], []
    for n in nodes:
        if instance.nodes[n].is_customer:
            xs.append(float(instance.nodes[n].x))
            ys.append(float(instance.nodes[n].y))
    if not xs:
        d = instance.nodes[instance.depot_index]
        return float(d.x), float(d.y)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def _delta_d(instance, u: int, i: int, v: int) -> float:
    dist = instance.distance_matrix
    return float(dist[u, i]) + float(dist[i, v]) - float(dist[u, v])


def _nearest_stations(instance, customer: int, k: int) -> list[int]:
    dist = instance.distance_matrix
    scored = [(float(dist[customer, s]), int(s)) for s in instance.charging_station_indices]
    scored.sort()
    return [s for _, s in scored[: max(1, k)]]


def _tw_optimistic(instance, nodes: list[int]) -> bool:
    depot = int(instance.depot_index)
    t = 0.0
    prev = depot
    tt = instance.travel_time_matrix
    horizon = float(getattr(instance, "planning_horizon", 1e18) or 1e18)
    for n in nodes:
        if not instance.nodes[n].is_customer:
            continue
        t += float(tt[prev, n])
        node = instance.nodes[n]
        t = max(t, float(node.ready_time))
        if t > float(node.due_time) + 1e-9:
            return False
        t += float(node.service_time)
        prev = n
    return t + float(tt[prev, depot]) <= horizon + 1e-9


def _try_sequences(instance, sequences: list[list[int]], stations: list[int]) -> tuple[list[int], list[float | None], object] | None:
    best = None
    for seq in sequences:
        if not _tw_optimistic(instance, seq):
            continue
        plan = replay_route(instance, seq, default_portion=1.0)
        if plan.feasible:
            key = (plan.distance, plan.duration)
            if best is None or key < best[0]:
                best = (key, plan.nodes, plan.labels, plan)
            continue
        for s in stations:
            for pos in range(1, len(seq)):
                prev, nxt = seq[pos - 1], seq[pos]
                if instance.nodes[prev].is_charging_station or instance.nodes[nxt].is_charging_station:
                    continue
                trial = seq[:pos] + [s] + seq[pos:]
                labs = [None] * len(trial)
                labs[pos] = 1.0
                plan = replay_route(instance, trial, labs, default_portion=1.0)
                if plan.feasible:
                    key = (plan.distance, plan.duration)
                    if best is None or key < best[0]:
                        best = (key, plan.nodes, plan.labels, plan)
    if best is None:
        return None
    return best[1], best[2], best[3]


def insert_customer(
    instance,
    routes: list,
    customer: int,
    *,
    nearby_routes: int = 8,
    insert_positions: int = 3,
    nearby_stations: int = 3,
) -> bool:
    """Insert ``customer`` into a nearby route, or open a new route."""
    cap = float(instance.vehicle.load_capacity)
    demand = float(instance.nodes[customer].demand)
    depot = int(instance.depot_index)
    cx, cy = float(instance.nodes[customer].x), float(instance.nodes[customer].y)
    stations = _nearest_stations(instance, customer, nearby_stations)

    scored: list[tuple[float, int]] = []
    for ri, (nodes, _) in enumerate(routes):
        if _load(instance, nodes) + demand > cap + 1e-9:
            continue
        gx, gy = _centroid(instance, nodes)
        scored.append(((gx - cx) ** 2 + (gy - cy) ** 2, ri))
    scored.sort()
    nearby = [ri for _, ri in scored[: max(1, nearby_routes)]]
    rest = [ri for _, ri in scored[max(1, nearby_routes):]]

    def _search(route_ids: list[int], n_pos: int):
        found_best = None
        for ri in route_ids:
            nodes, labels = routes[ri]
            gaps: list[tuple[float, int]] = []
            for pos in range(1, len(nodes)):
                u, v = nodes[pos - 1], nodes[pos]
                if instance.nodes[u].is_charging_station and instance.nodes[v].is_charging_station:
                    continue
                gaps.append((_delta_d(instance, u, customer, v), pos))
            gaps.sort()
            positions = sorted({p for _, p in gaps[: max(1, n_pos)]})
            if nodes:
                positions.append(len(nodes) - 1)
            positions = sorted(set(positions))
            sequences = [list(nodes[:pos]) + [customer] + list(nodes[pos:]) for pos in positions]
            found = _try_sequences(instance, sequences, stations)
            if found is None:
                continue
            new_nodes, new_labs, plan = found
            key = (plan.distance - replay_route(instance, nodes, labels).distance, plan.duration, ri)
            if found_best is None or key < found_best[0]:
                found_best = (key, ri, new_nodes, new_labs)
        return found_best

    best = _search(nearby, insert_positions)
    if best is None and rest:
        best = _search(rest, max(insert_positions, 5))

    if best is not None:
        _, ri, new_nodes, new_labs = best
        routes[ri] = (new_nodes, new_labs)
        return True

    opened = _try_sequences(instance, [[depot, customer, depot]], stations)
    if opened is None:
        return False
    routes.append((opened[0], opened[1]))
    return True


def construct_feasible(
    instance,
    *,
    nearby_routes: int = 8,
    insert_positions: int = 3,
    nearby_stations: int = 3,
    deadline: float | None = None,
) -> list:
    routes: list = []
    t0 = time.perf_counter()
    for cust in customer_order(instance):
        if deadline is not None and time.perf_counter() - t0 > deadline:
            break
        insert_customer(
            instance, routes, cust,
            nearby_routes=nearby_routes,
            insert_positions=insert_positions,
            nearby_stations=nearby_stations,
        )
    return [(list(n), list(l)) for n, l in routes if any(instance.nodes[x].is_customer for x in n)]
