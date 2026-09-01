"""Route list helpers shared by Phase I and Phase II."""
from __future__ import annotations

from src.hanif.replay import replay_solution


def clone(routes: list) -> list:
    return [(list(n), list(l) if l is not None else [None] * len(n)) for n, l in routes]


def customers_of(instance, nodes: list[int]) -> list[int]:
    return [n for n in nodes if instance.nodes[n].is_customer]


def nonempty(instance, routes: list) -> list:
    return [(n, l) for n, l in routes if customers_of(instance, n)]


def used_k(instance, routes: list) -> int:
    return sum(1 for n, _ in routes if customers_of(instance, n))


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


def customer_key(instance, nodes: list[int]) -> tuple[int, ...]:
    return tuple(customers_of(instance, nodes))


def evaluate(instance, routes: list):
    routes = nonempty(instance, routes)
    m = replay_solution(instance, routes)
    return m, (
        0 if m.feasible else 1,
        int(m.unserved),
        int(m.hard_violations),
        int(m.used_k),
        float(m.distance),
    )
