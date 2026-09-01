"""Exact ALNS destroy / repair neighborhoods for TG-FPS."""
from __future__ import annotations

import random
from typing import Callable

from src.hanif.env.cost import CostWeights
from src.hanif.search.repair import (
    _clone_routes,
    _served_in_routes,
    best_insertion,
    critical_slack_order,
    regret_insertion,
    remove_station_if_redundant,
    repair_solution,
)
from src.hanif.search.route_replay import replay_solution
from src.hanif.search.score import SearchScore


def _customers_on_route(instance, nodes: list[int]) -> list[tuple[int, int]]:
    """Return (position, customer_node) for customer visits."""
    out = []
    for i, n in enumerate(nodes):
        if instance.nodes[n].is_customer:
            out.append((i, int(n)))
    return out


def destroy_random_customers(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    k: int,
    rng: random.Random,
) -> tuple[list[tuple[list[int], list[float | None]]], list[int]]:
    current = _clone_routes(routes)
    pool: list[tuple[int, int, int]] = []  # route, pos, customer
    for ri, (nodes, _) in enumerate(current):
        for pos, cust in _customers_on_route(instance, nodes):
            pool.append((ri, pos, cust))
    rng.shuffle(pool)
    removed: list[int] = []
    # Remove from highest position first per route to keep indices stable.
    by_route: dict[int, list[tuple[int, int]]] = {}
    for ri, pos, cust in pool[: max(0, k)]:
        by_route.setdefault(ri, []).append((pos, cust))
        removed.append(cust)
    for ri, items in by_route.items():
        nodes, labels = current[ri]
        labels = list(labels) if labels else [None] * len(nodes)
        for pos, _cust in sorted(items, key=lambda x: -x[0]):
            del nodes[pos]
            if pos < len(labels):
                del labels[pos]
        current[ri] = (nodes, labels)
    return current, removed


def destroy_critical_tw(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    k: int,
    rng: random.Random,
) -> tuple[list[tuple[list[int], list[float | None]]], list[int]]:
    current = _clone_routes(routes)
    pool: list[tuple[float, int, int, int]] = []
    for ri, (nodes, _) in enumerate(current):
        for pos, cust in _customers_on_route(instance, nodes):
            pool.append((float(instance.nodes[cust].due_time), ri, pos, cust))
    pool.sort(key=lambda x: (x[0], x[3]))
    removed: list[int] = []
    by_route: dict[int, list[int]] = {}
    for _due, ri, pos, cust in pool[: max(0, k)]:
        by_route.setdefault(ri, []).append(pos)
        removed.append(cust)
    for ri, positions in by_route.items():
        nodes, labels = current[ri]
        labels = list(labels) if labels else [None] * len(nodes)
        for pos in sorted(positions, reverse=True):
            del nodes[pos]
            if pos < len(labels):
                del labels[pos]
        current[ri] = (nodes, labels)
    return current, removed


def destroy_route(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    rng: random.Random,
) -> tuple[list[tuple[list[int], list[float | None]]], list[int]]:
    current = _clone_routes(routes)
    used = [i for i, (nodes, _) in enumerate(current) if any(instance.nodes[n].is_customer for n in nodes)]
    if not used:
        return current, []
    ri = rng.choice(used)
    nodes, _ = current[ri]
    removed = [n for n in nodes if instance.nodes[n].is_customer]
    depot = instance.depot_index
    current[ri] = ([depot, depot], [None, None])
    return current, removed


def relocate_customer(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    rng: random.Random | None = None,
) -> list[tuple[list[int], list[float | None]]] | None:
    """Try relocating one random customer to a better insertion."""
    rng = rng or random.Random(0)
    current = _clone_routes(routes)
    pool: list[tuple[int, int, int]] = []
    for ri, (nodes, _) in enumerate(current):
        for pos, cust in _customers_on_route(instance, nodes):
            pool.append((ri, pos, cust))
    if not pool:
        return None
    ri, pos, cust = rng.choice(pool)
    nodes, labels = current[ri]
    labels = list(labels) if labels else [None] * len(nodes)
    del nodes[pos]
    if pos < len(labels):
        del labels[pos]
    current[ri] = (nodes, labels)
    cand = best_insertion(instance, current, cust, config=config, weights=weights, m=m)
    return None if cand is None else cand.routes


def swap_customers(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    rng: random.Random | None = None,
) -> list[tuple[list[int], list[float | None]]] | None:
    rng = rng or random.Random(0)
    current = _clone_routes(routes)
    pool: list[tuple[int, int, int]] = []
    for ri, (nodes, _) in enumerate(current):
        for pos, cust in _customers_on_route(instance, nodes):
            pool.append((ri, pos, cust))
    if len(pool) < 2:
        return None
    a, b = rng.sample(pool, 2)
    if a[0] == b[0] and a[1] == b[1]:
        return None
    ra, pa, ca = a
    rb, pb, cb = b
    na, la = current[ra]
    nb, lb = current[rb]
    na, nb = list(na), list(nb)
    la = list(la) if la else [None] * len(na)
    lb = list(lb) if lb else [None] * len(nb)
    na[pa], nb[pb] = nb[pb], na[pa]
    current[ra] = (na, la)
    current[rb] = (nb, lb)
    res = replay_solution(instance, current, config=config, weights=weights, m=m)
    return current if res.feasible or (res.score and res.unserved_customers == 0) else None


def two_opt_route(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    route_index: int,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
) -> list[tuple[list[int], list[float | None]]] | None:
    """Exact-replay 2-opt on customer subsequence of one route."""
    current = _clone_routes(routes)
    if route_index >= len(current):
        return None
    nodes, labels = current[route_index]
    cust_pos = [i for i, n in enumerate(nodes) if instance.nodes[n].is_customer]
    if len(cust_pos) < 4:
        return None
    base = replay_solution(instance, current, config=config, weights=weights, m=m)
    if base.score is None:
        return None
    best = None
    best_score = base.score
    for i in range(len(cust_pos) - 1):
        for j in range(i + 1, len(cust_pos)):
            p, q = cust_pos[i], cust_pos[j]
            trial = _clone_routes(current)
            n2 = list(trial[route_index][0])
            # Reverse segment between p and q inclusive.
            n2[p : q + 1] = reversed(n2[p : q + 1])
            # Labels for stations may misalign; keep non-station labels None and drop mid-stations in segment.
            l2 = [None] * len(n2)
            trial[route_index] = (n2, l2)
            res = replay_solution(instance, trial, config=config, weights=weights, m=m)
            if res.score is not None and res.score.dominates(best_score):
                best = trial
                best_score = res.score
    return best


DESTROY_OPS: dict[str, Callable] = {
    "random_customers": destroy_random_customers,
    "critical_tw": destroy_critical_tw,
    "route_removal": lambda inst, routes, k, rng: destroy_route(inst, routes, rng),
}
