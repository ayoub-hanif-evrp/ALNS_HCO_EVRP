"""Bounded ejection chains for fleet reduction / critical rescue."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from src.schneider.search.deadline import SearchDeadline
from src.schneider.search.repair import _clone_routes, evaluate_insertion, enumerate_insertions_for_customer
from src.schneider.search.route_replay import replay_solution
from src.schneider.search.score import SearchScore


@dataclass
class EjectionChainAttempt:
    depth: int
    customer: int
    feasible: bool
    runtime: float
    reason: str | None = None


def _customers_on_route(instance, nodes) -> list[int]:
    return [int(n) for n in nodes if instance.nodes[n].is_customer]


def _remove_customer(instance, routes, route_index: int, customer: int):
    nodes, labs = routes[route_index]
    labs = list(labs) if labs else [None] * len(nodes)
    new_nodes, new_labs = [], []
    for n, lab in zip(nodes, labs):
        if int(n) == int(customer):
            continue
        new_nodes.append(int(n))
        new_labs.append(lab)
    if len(new_nodes) < 2:
        depot = instance.depot_index
        new_nodes, new_labs = [depot, depot], [None, None]
    routes[route_index] = (new_nodes, new_labs)
    return routes


def try_ejection_chain(
    instance,
    routes: list,
    customer: int,
    *,
    config: dict | None = None,
    weights=None,
    m: int | None = None,
    depth: int = 2,
    max_candidates: int = 20,
    deadline: SearchDeadline | None = None,
) -> tuple[list | None, SearchScore | None, EjectionChainAttempt]:
    """insert u → eject v → insert v (→ eject w …), exact-replay validated."""
    t0 = time.perf_counter()
    cfg = config or {}
    inserts = enumerate_insertions_for_customer(
        instance, routes, customer, config=cfg, weights=weights, m=m, allow_new_route=False,
    )[:max_candidates]

    # Direct insert first
    for cand in inserts:
        if deadline is not None and deadline.expired():
            break
        trial = cand.routes
        final = replay_solution(instance, trial, config=cfg, weights=weights, m=m)
        if final.feasible and customer in final.served_customers:
            return trial, final.score, EjectionChainAttempt(
                depth=0, customer=customer, feasible=True, runtime=time.perf_counter() - t0,
            )

    if depth < 1:
        return None, None, EjectionChainAttempt(
            depth=depth, customer=customer, feasible=False,
            runtime=time.perf_counter() - t0, reason="no_direct_insert",
        )

    # Depth-1/2: force-insert by ejecting a customer from target route
    for cand in inserts[: max(5, max_candidates // 2)]:
        if deadline is not None and deadline.expired():
            break
        ri = cand.route_index
        base = _clone_routes(routes)
        # Apply forced insert of u (even if may be infeasible until ejection)
        # Use candidate routes if provided
        trial = _clone_routes(cand.routes)
        victims = _customers_on_route(instance, trial[ri][0])
        victims = [v for v in victims if v != customer][:8]
        for v in victims:
            if deadline is not None and deadline.expired():
                break
            t2 = _clone_routes(trial)
            t2 = _remove_customer(instance, t2, ri, v)
            # Ensure u still present; if not, re-insert
            served_u = any(customer in nodes for nodes, _ in t2 for _ in [0])
            # Fix presence check
            present = any(customer in nodes for nodes, _ in t2)
            if not present:
                rein = evaluate_insertion(
                    instance, t2, ri, min(1, max(0, len(t2[ri][0]) - 1)), customer,
                    config=cfg, weights=weights, m=m,
                )
                if rein is None:
                    continue
                t2 = rein.routes
            # Insert v elsewhere
            v_inserts = enumerate_insertions_for_customer(
                instance, t2, v, config=cfg, weights=weights, m=m, allow_new_route=True,
            )[:max_candidates]
            for vc in v_inserts:
                t3 = vc.routes
                if depth >= 2:
                    # Optional second ejection if v insert needed further ejection — skip if feasible
                    final = replay_solution(instance, t3, config=cfg, weights=weights, m=m)
                    if final.feasible and int(final.unserved_customers) == 0:
                        return t3, final.score, EjectionChainAttempt(
                            depth=1, customer=customer, feasible=True,
                            runtime=time.perf_counter() - t0,
                        )
                    # Depth 2: eject w from v's new route
                    if depth >= 3 or depth >= 2:
                        r_v = vc.route_index
                        w_list = [w for w in _customers_on_route(instance, t3[r_v][0]) if w not in (customer, v)][:5]
                        for w in w_list:
                            t4 = _clone_routes(t3)
                            t4 = _remove_customer(instance, t4, r_v, w)
                            w_ins = enumerate_insertions_for_customer(
                                instance, t4, w, config=cfg, weights=weights, m=m, allow_new_route=True,
                            )[:10]
                            for wc in w_ins:
                                final2 = replay_solution(
                                    instance, wc.routes, config=cfg, weights=weights, m=m,
                                )
                                if final2.feasible and int(final2.unserved_customers) == 0:
                                    return wc.routes, final2.score, EjectionChainAttempt(
                                        depth=2, customer=customer, feasible=True,
                                        runtime=time.perf_counter() - t0,
                                    )
                else:
                    final = replay_solution(instance, t3, config=cfg, weights=weights, m=m)
                    if final.feasible and int(final.unserved_customers) == 0:
                        return t3, final.score, EjectionChainAttempt(
                            depth=1, customer=customer, feasible=True,
                            runtime=time.perf_counter() - t0,
                        )

    return None, None, EjectionChainAttempt(
        depth=depth, customer=customer, feasible=False,
        runtime=time.perf_counter() - t0, reason="chain_exhausted",
    )


def ejection_rescue(
    instance,
    routes: list,
    unserved: list[int],
    *,
    config: dict | None = None,
    weights=None,
    m: int | None = None,
    depth: int = 2,
    deadline: SearchDeadline | None = None,
) -> tuple[list, SearchScore | None, list[EjectionChainAttempt]]:
    trial = _clone_routes(routes)
    logs: list[EjectionChainAttempt] = []
    score = None
    for u in unserved:
        if deadline is not None and deadline.expired():
            break
        new_routes, new_score, att = try_ejection_chain(
            instance, trial, u, config=config, weights=weights, m=m,
            depth=depth, deadline=deadline,
        )
        logs.append(att)
        if new_routes is not None:
            trial = new_routes
            score = new_score
    return trial, score, logs
