"""Fleet compression: drop a short route, reinsert most-constrained-first.

Not regret. No route-scoring formula. No cold rebuild. No 2-opt.
"""
from __future__ import annotations

import time

from src.hanif.construction import (
    INSERT_POSITIONS,
    NEARBY_ROUTES,
    NEARBY_STATIONS,
    insert_customer,
    nearby_route_ids,
    route_load,
)
from src.hanif.replay import customer_sequence, replay_solution

Q_SMALLEST = 8


def _customers(instance, nodes: list[int]) -> list[int]:
    return [n for n in nodes if instance.nodes[n].is_customer]


def _windows_overlap(instance, customer: int, host_nodes: list[int]) -> bool:
    hosts = _customers(instance, host_nodes)
    if not hosts:
        return True
    ready = float(instance.nodes[customer].ready_time)
    due = float(instance.nodes[customer].due_time)
    for n in hosts:
        nd = instance.nodes[n]
        if float(nd.ready_time) <= due and ready <= float(nd.due_time):
            return True
    return False


def receiving_routes(
    instance,
    customer: int,
    routes: list,
    *,
    geo_limit: int,
) -> list[int]:
    """K_i: remaining routes that pass capacity, geography, and time-window filters."""
    cap = float(instance.vehicle.load_capacity)
    demand = float(instance.nodes[customer].demand)
    geo = nearby_route_ids(instance, routes, customer, limit=geo_limit)
    return [ri for ri in geo if route_load(instance, routes[ri][0]) + demand <= cap + 1e-9
            and _windows_overlap(instance, customer, routes[ri][0])]


def try_eliminate(
    instance,
    routes: list,
    route_index: int,
    *,
    nearby_routes: int = NEARBY_ROUTES,
    insert_positions: int = INSERT_POSITIONS,
    nearby_stations: int = NEARBY_STATIONS,
) -> list | None:
    if route_index < 0 or route_index >= len(routes):
        return None
    displaced = _customers(instance, routes[route_index][0])
    if not displaced:
        return None
    remaining = [(list(n), list(l)) for i, (n, l) in enumerate(routes) if i != route_index]

    def difficulty(c: int) -> tuple:
        k_i = receiving_routes(instance, c, remaining, geo_limit=max(nearby_routes, 16))
        return (len(k_i), float(instance.nodes[c].due_time), c)

    for cust in sorted(displaced, key=difficulty):
        k_i = receiving_routes(instance, cust, remaining, geo_limit=max(nearby_routes, 16))
        if not insert_customer(
            instance, remaining, cust,
            nearby_routes=nearby_routes,
            insert_positions=insert_positions,
            nearby_stations=nearby_stations,
            allow_new_route=False,
            allowed_routes=k_i,
        ):
            return None
    check = replay_solution(instance, remaining)
    if not check.feasible:
        return None
    return remaining


def reduce_fleet(
    instance,
    routes: list,
    *,
    max_candidates: int = Q_SMALLEST,
    nearby_routes: int = NEARBY_ROUTES,
    insert_positions: int = INSERT_POSITIONS,
    nearby_stations: int = NEARBY_STATIONS,
    deadline: float | None = None,
) -> tuple[list, dict]:
    current = [(list(n), list(l)) for n, l in routes]
    log = {"eliminations": 0, "attempts": 0}
    t0 = time.perf_counter()
    while True:
        if deadline is not None and time.perf_counter() - t0 > deadline:
            break
        used = [(i, _customers(instance, n)) for i, (n, _) in enumerate(current)]
        used = [(i, c) for i, c in used if c]
        if len(used) <= 1:
            break
        # q smallest routes only. Ties by route index. No spatial score.
        ordered = sorted(used, key=lambda item: (len(item[1]), item[0]))
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
