"""Shared operators for Hanif metaheuristic baselines.

Feasibility is always decided by ``src.hanif.replay``. Lexicographic score is
(unserved, fleet size K, distance). These are matched internals, not published
algorithms from other papers.
"""
from __future__ import annotations

from src.hanif.construction import insert_customer
from src.hanif.replay import replay_route, replay_solution

NEARBY_ROUTES = 8
INSERT_POSITIONS = 3
NEARBY_STATIONS = 3


def clone(routes: list) -> list:
    return [(list(n), list(l) if l is not None else [None] * len(n)) for n, l in routes]


def customers_of(instance, nodes: list[int]) -> list[int]:
    return [n for n in nodes if instance.nodes[n].is_customer]


def nonempty(instance, routes: list) -> list:
    return [(n, l) for n, l in routes if customers_of(instance, n)]


def remove_customer(instance, routes: list, customer: int) -> bool:
    for ri, (nodes, labels) in enumerate(routes):
        if customer not in nodes:
            continue
        labs = list(labels) if labels is not None else [None] * len(nodes)
        keep_n, keep_l = [], []
        for n, lab in zip(nodes, labs):
            if n == customer:
                continue
            keep_n.append(n)
            keep_l.append(lab)
        routes[ri] = (keep_n, keep_l)
        return True
    return False


def all_customers(instance, routes: list) -> list[int]:
    out = []
    for nodes, _ in routes:
        out.extend(customers_of(instance, nodes))
    return out


def evaluate(instance, routes: list):
    routes = nonempty(instance, routes)
    m = replay_solution(instance, routes)
    # Feasible complete solutions always rank above partial or violating ones.
    return m, (
        0 if m.feasible else 1,
        int(m.unserved),
        int(m.hard_violations),
        int(m.used_k),
        float(m.distance),
    )


def better(a: tuple, b: tuple) -> bool:
    return a < b


def greedy_reinsert(instance, routes: list, customers: list[int]) -> bool:
    ok = True
    for c in customers:
        if not insert_customer(
            instance, routes, c,
            nearby_routes=NEARBY_ROUTES,
            insert_positions=INSERT_POSITIONS,
            nearby_stations=NEARBY_STATIONS,
            allow_new_route=True,
        ):
            ok = False
    return ok


def relocate_once(instance, routes: list, rng) -> list | None:
    custs = all_customers(instance, routes)
    if not custs:
        return None
    c = rng.choice(custs)
    trial = clone(routes)
    if not remove_customer(instance, trial, c):
        return None
    if not insert_customer(
        instance, trial, c,
        nearby_routes=NEARBY_ROUTES,
        insert_positions=INSERT_POSITIONS,
        nearby_stations=NEARBY_STATIONS,
        allow_new_route=True,
    ):
        return None
    return nonempty(instance, trial)


def swap_once(instance, routes: list, rng) -> list | None:
    custs = all_customers(instance, routes)
    if len(custs) < 2:
        return None
    a, b = rng.sample(custs, 2)
    trial = clone(routes)
    if not remove_customer(instance, trial, a) or not remove_customer(instance, trial, b):
        return None
    if not insert_customer(instance, trial, b, nearby_routes=NEARBY_ROUTES, insert_positions=INSERT_POSITIONS, nearby_stations=NEARBY_STATIONS, allow_new_route=True):
        return None
    if not insert_customer(instance, trial, a, nearby_routes=NEARBY_ROUTES, insert_positions=INSERT_POSITIONS, nearby_stations=NEARBY_STATIONS, allow_new_route=True):
        return None
    return nonempty(instance, trial)


def two_opt_once(instance, routes: list, rng) -> list | None:
    used = [i for i, (n, _) in enumerate(routes) if len(customers_of(instance, n)) >= 4]
    if not used:
        return None
    ri = rng.choice(used)
    nodes, labels = routes[ri]
    cust_idx = [i for i, n in enumerate(nodes) if instance.nodes[n].is_customer]
    if len(cust_idx) < 4:
        return None
    i = rng.randrange(len(cust_idx) - 1)
    j = rng.randrange(i + 1, len(cust_idx))
    lo, hi = cust_idx[i], cust_idx[j]
    new_nodes = nodes[:lo] + list(reversed(nodes[lo : hi + 1])) + nodes[hi + 1 :]
    plan = replay_route(instance, new_nodes, default_portion=1.0)
    if not plan.feasible:
        return None
    trial = clone(routes)
    trial[ri] = (list(plan.nodes), list(plan.labels))
    return nonempty(instance, trial)
