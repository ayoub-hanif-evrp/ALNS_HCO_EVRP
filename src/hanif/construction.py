"""Stage 1 construction: due-time insertion, minimum feasible detour.

No weighted score, no lexicographic comparison, no random seeds.
Feasibility is a constraint. Distance is the insertion criterion.
"""
from __future__ import annotations

import time

from src.hanif.replay import replay_route

# A few nearby routes and a few insertion positions — not the full neighbourhood.
NEARBY_ROUTES = 8
INSERT_POSITIONS = 3
NEARBY_STATIONS = 3


def customer_order(instance) -> list[int]:
    """Process customers by increasing due time b_i (stable ties)."""
    custs = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    custs.sort(key=lambda i: (
        float(instance.nodes[i].due_time),
        float(instance.nodes[i].ready_time),
        i,
    ))
    return custs


def route_load(instance, nodes: list[int]) -> float:
    return sum(float(instance.nodes[n].demand) for n in nodes if instance.nodes[n].is_customer)


def route_centroid(instance, nodes: list[int]) -> tuple[float, float]:
    xs, ys = [], []
    for n in nodes:
        if instance.nodes[n].is_customer:
            xs.append(float(instance.nodes[n].x))
            ys.append(float(instance.nodes[n].y))
    if not xs:
        d = instance.nodes[instance.depot_index]
        return float(d.x), float(d.y)
    return sum(xs) / len(xs), sum(ys) / len(ys)


def delta_d(instance, u: int, i: int, v: int) -> float:
    dist = instance.distance_matrix
    return float(dist[u, i]) + float(dist[i, v]) - float(dist[u, v])


def nearest_stations(instance, node: int, k: int) -> list[int]:
    dist = instance.distance_matrix
    scored = [(float(dist[node, s]), int(s)) for s in instance.charging_station_indices]
    scored.sort()
    return [s for _, s in scored[: max(1, k)]]


def tw_optimistic(instance, nodes: list[int]) -> bool:
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


def nearby_route_ids(
    instance,
    routes: list,
    customer: int,
    *,
    limit: int,
) -> list[int]:
    cap = float(instance.vehicle.load_capacity)
    demand = float(instance.nodes[customer].demand)
    cx, cy = float(instance.nodes[customer].x), float(instance.nodes[customer].y)
    scored: list[tuple[float, int]] = []
    for ri, (nodes, _) in enumerate(routes):
        if route_load(instance, nodes) + demand > cap + 1e-9:
            continue
        gx, gy = route_centroid(instance, nodes)
        scored.append(((gx - cx) ** 2 + (gy - cy) ** 2, ri))
    scored.sort()
    return [ri for _, ri in scored[: max(1, limit)]] if scored else []


def cheapest_positions(instance, nodes: list[int], customer: int, n_pos: int) -> list[int]:
    gaps: list[tuple[float, int]] = []
    for pos in range(1, len(nodes)):
        u, v = nodes[pos - 1], nodes[pos]
        if instance.nodes[u].is_charging_station and instance.nodes[v].is_charging_station:
            continue
        gaps.append((delta_d(instance, u, customer, v), pos))
    gaps.sort()
    positions = {p for _, p in gaps[: max(1, n_pos)]}
    if nodes:
        positions.add(len(nodes) - 1)
    return sorted(positions)


def _accept(plan) -> bool:
    return bool(plan.feasible)


def _try_plain(instance, seq: list[int]):
    if not tw_optimistic(instance, seq):
        return None
    plan = replay_route(instance, seq, default_portion=1.0)
    return plan if _accept(plan) else None


def _try_with_station(instance, seq: list[int], customer: int, stations: list[int]):
    """Insert one nearby station next to the new customer, full recharge."""
    try:
        cpos = seq.index(customer)
    except ValueError:
        return None
    best = None
    best_d = None
    for s in stations:
        for insert_at in (cpos, cpos + 1):
            prev, nxt = seq[insert_at - 1], seq[insert_at]
            if instance.nodes[prev].is_charging_station or instance.nodes[nxt].is_charging_station:
                continue
            trial = seq[:insert_at] + [s] + seq[insert_at:]
            labs = [None] * len(trial)
            labs[insert_at] = 1.0
            plan = replay_route(instance, trial, labs, default_portion=1.0)
            if _accept(plan) and (best_d is None or plan.distance < best_d):
                best, best_d = plan, plan.distance
    return best


def insert_customer(
    instance,
    routes: list,
    customer: int,
    *,
    nearby_routes: int = NEARBY_ROUTES,
    insert_positions: int = INSERT_POSITIONS,
    nearby_stations: int = NEARBY_STATIONS,
    allow_new_route: bool = True,
    allowed_routes: list[int] | None = None,
) -> bool:
    """Insert customer i into a nearby route at the feasible position of minimum Δd.

    If no existing nearby route can take i, open a new route.
    A nearby station is inserted only when the customer-only plan is infeasible.
    """
    depot = int(instance.depot_index)
    stations = nearest_stations(instance, customer, nearby_stations)
    candidates = nearby_route_ids(instance, routes, customer, limit=nearby_routes)
    if allowed_routes is not None:
        allow = set(allowed_routes)
        candidates = [ri for ri in candidates if ri in allow]

    best = None  # (delta_d, ri, pos, nodes, labels)
    for ri in candidates:
        nodes, _labels = routes[ri]
        for pos in cheapest_positions(instance, nodes, customer, insert_positions):
            u, v = nodes[pos - 1], nodes[pos]
            detour = delta_d(instance, u, customer, v)
            seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
            plan = _try_plain(instance, seq)
            if plan is None:
                continue
            key = (detour, ri, pos)
            if best is None or key < best[0]:
                best = (key, ri, plan.nodes, plan.labels)

    if best is None:
        for ri in candidates:
            nodes, _labels = routes[ri]
            for pos in cheapest_positions(instance, nodes, customer, insert_positions):
                u, v = nodes[pos - 1], nodes[pos]
                seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
                plan = _try_with_station(instance, seq, customer, stations)
                if plan is None:
                    continue
                detour = plan.distance  # station detour included
                key = (detour, ri, pos)
                if best is None or key < best[0]:
                    best = (key, ri, plan.nodes, plan.labels)

    if best is not None:
        _, ri, new_nodes, new_labs = best
        routes[ri] = (list(new_nodes), list(new_labs))
        return True

    if not allow_new_route:
        return False

    opened = _try_plain(instance, [depot, customer, depot])
    if opened is None:
        opened = _try_with_station(instance, [depot, customer, depot], customer, stations)
    if opened is None:
        return False
    routes.append((list(opened.nodes), list(opened.labels)))
    return True


def construct_feasible(
    instance,
    *,
    nearby_routes: int = NEARBY_ROUTES,
    insert_positions: int = INSERT_POSITIONS,
    nearby_stations: int = NEARBY_STATIONS,
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
