"""Deterministic route elimination with difficulty-ordered direct reinsertion."""
from __future__ import annotations

import time

from src.hanif.current.construction import _centroid, _delta_d, _load, _nearest_stations, _tw_optimistic
from src.hanif.replay import customer_sequence, replay_route, replay_solution


def _route_customers(instance, nodes: list[int]) -> list[int]:
    return [n for n in nodes if instance.nodes[n].is_customer]


def _compat_count(instance, customer: int, routes: list, *, nearby: int = 8) -> int:
    cap = float(instance.vehicle.load_capacity)
    demand = float(instance.nodes[customer].demand)
    cx, cy = float(instance.nodes[customer].x), float(instance.nodes[customer].y)
    scored = []
    for ri, (nodes, _) in enumerate(routes):
        if _load(instance, nodes) + demand > cap + 1e-9:
            continue
        gx, gy = _centroid(instance, nodes)
        scored.append(((gx - cx) ** 2 + (gy - cy) ** 2, ri))
    scored.sort()
    count = 0
    due = float(instance.nodes[customer].due_time)
    ready = float(instance.nodes[customer].ready_time)
    for _, ri in scored[: max(1, nearby)]:
        nodes = routes[ri][0]
        # cheap TW: some customer on the host has overlapping window slack
        ok = False
        for n in _route_customers(instance, nodes):
            nd = instance.nodes[n]
            if float(nd.ready_time) <= due and ready <= float(nd.due_time):
                ok = True
                break
        if ok or not _route_customers(instance, nodes):
            count += 1
    return count


def _insert_into(
    instance,
    routes: list,
    customer: int,
    *,
    nearby_routes: int,
    insert_positions: int,
    nearby_stations: int,
) -> bool:
    cap = float(instance.vehicle.load_capacity)
    demand = float(instance.nodes[customer].demand)
    cx, cy = float(instance.nodes[customer].x), float(instance.nodes[customer].y)
    stations = _nearest_stations(instance, customer, nearby_stations)
    scored = []
    for ri, (nodes, _) in enumerate(routes):
        if _load(instance, nodes) + demand > cap + 1e-9:
            continue
        gx, gy = _centroid(instance, nodes)
        scored.append(((gx - cx) ** 2 + (gy - cy) ** 2, ri))
    scored.sort()
    best = None
    for _, ri in scored[: max(1, nearby_routes)]:
        nodes, labels = routes[ri]
        gaps = []
        for pos in range(1, len(nodes)):
            u, v = nodes[pos - 1], nodes[pos]
            gaps.append((_delta_d(instance, u, customer, v), pos))
        gaps.sort()
        positions = sorted({p for _, p in gaps[: max(1, insert_positions)]} | {len(nodes) - 1})
        for pos in positions:
            seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
            if not _tw_optimistic(instance, seq):
                continue
            plan = replay_route(instance, seq, default_portion=1.0)
            if plan.feasible:
                key = (plan.distance, plan.duration, ri)
                if best is None or key < best[0]:
                    best = (key, ri, plan.nodes, plan.labels)
                continue
            for s in stations:
                trial = list(nodes[:pos]) + [s, customer] + list(nodes[pos:])
                labs = [None] * len(trial)
                labs[pos] = 1.0
                plan = replay_route(instance, trial, labs, default_portion=1.0)
                if plan.feasible:
                    key = (plan.distance, plan.duration, ri)
                    if best is None or key < best[0]:
                        best = (key, ri, plan.nodes, plan.labels)
    if best is None:
        return False
    _, ri, nodes, labs = best
    routes[ri] = (list(nodes), list(labs))
    return True


def _two_opt(instance, nodes: list[int], labels: list[float | None]) -> tuple[list[int], list[float | None]]:
    custs = _route_customers(instance, nodes)
    if len(custs) < 4:
        return nodes, labels
    depot = int(instance.depot_index)
    best_seq = list(custs)
    base = replay_route(instance, [depot] + best_seq + [depot], default_portion=1.0)
    if not base.feasible:
        return nodes, labels
    best_d = base.distance
    improved = True
    guard = 0
    while improved and guard < 20:
        improved = False
        guard += 1
        for i in range(len(best_seq) - 1):
            for j in range(i + 1, len(best_seq)):
                trial = best_seq[:i] + list(reversed(best_seq[i : j + 1])) + best_seq[j + 1 :]
                plan = replay_route(instance, [depot] + trial + [depot], default_portion=1.0)
                if plan.feasible and plan.distance + 1e-6 < best_d:
                    best_seq, best_d, improved = trial, plan.distance, True
                    nodes, labels = plan.nodes, plan.labels
                    break
            if improved:
                break
    return nodes, labels


def try_eliminate(
    instance,
    routes: list,
    route_index: int,
    *,
    nearby_routes: int = 8,
    insert_positions: int = 3,
    nearby_stations: int = 3,
) -> list | None:
    if route_index < 0 or route_index >= len(routes):
        return None
    displaced = _route_customers(instance, routes[route_index][0])
    if not displaced:
        return None
    remaining = [r for i, r in enumerate(routes) if i != route_index]
    remaining = [(list(n), list(l)) for n, l in remaining]

    def difficulty(c: int) -> tuple:
        return (
            _compat_count(instance, c, remaining, nearby=nearby_routes),
            float(instance.nodes[c].due_time),
            -float(instance.nodes[c].demand),
            c,
        )

    for cust in sorted(displaced, key=difficulty):
        if not _insert_into(
            instance, remaining, cust,
            nearby_routes=nearby_routes,
            insert_positions=insert_positions,
            nearby_stations=nearby_stations,
        ):
            return None
    check = replay_solution(instance, remaining)
    if not check.feasible:
        return None
    cleaned = []
    for nodes, labels in remaining:
        n2, l2 = _two_opt(instance, nodes, labels)
        cleaned.append((n2, l2))
    final = replay_solution(instance, cleaned)
    return cleaned if final.feasible else remaining


def reduce_fleet(
    instance,
    routes: list,
    *,
    max_candidates: int = 10,
    nearby_routes: int = 8,
    insert_positions: int = 3,
    nearby_stations: int = 3,
    deadline: float | None = None,
) -> tuple[list, dict]:
    current = [(list(n), list(l)) for n, l in routes]
    log = {"eliminations": 0, "attempts": 0}
    t0 = time.perf_counter()
    while True:
        if deadline is not None and time.perf_counter() - t0 > deadline:
            break
        used = [(i, _route_customers(instance, n)) for i, (n, _) in enumerate(current)]
        used = [(i, c) for i, c in used if c]
        if len(used) <= 1:
            break
        centroids = {i: _centroid(instance, current[i][0]) for i, _ in used}

        def rank(item: tuple[int, list[int]]) -> tuple:
            i, custs = item
            gx, gy = centroids[i]
            nearest = min(
                ((gx - centroids[j][0]) ** 2 + (gy - centroids[j][1]) ** 2, j)
                for j, _ in used if j != i
            )[0]
            return (len(custs), nearest, i)

        ordered = sorted(used, key=rank)
        success = False
        for ri, _custs in ordered[: max(1, max_candidates)]:
            if deadline is not None and time.perf_counter() - t0 > deadline:
                break
            log["attempts"] += 1
            trial = try_eliminate(
                instance, current, ri,
                nearby_routes=nearby_routes,
                insert_positions=insert_positions,
                nearby_stations=nearby_stations,
            )
            if trial is None:
                continue
            after = replay_solution(instance, trial)
            before_k = sum(1 for n, _ in current if customer_sequence(instance, n))
            if after.feasible and after.used_k < before_k:
                current = trial
                log["eliminations"] += 1
                success = True
                break
        if not success:
            break
    return current, log
