"""Explicit fleet compression after the routing ALNS.

Keskin & Çatay: shorter (just-enough) charges free time-window slack so more
customers fit on existing routes. Wang et al. (2026): Phase 1 should eliminate
routes, not treat a smaller fleet as a side effect of distance search.
"""
from __future__ import annotations

import time

from src.alns_hco.insert import insert_just_enough
from src.alns_hco.oracle import fast_hco, just_enough_labels
from src.alns_hco.routes import clone, customers_of, nonempty
from src.hanif.construction import nearby_route_ids, route_load
from src.hanif.replay import replay_route, replay_solution


def _overlap(instance, customer: int, host_nodes: list[int]) -> bool:
    hosts = customers_of(instance, host_nodes)
    if not hosts:
        return True
    ready = float(instance.nodes[customer].ready_time)
    due = float(instance.nodes[customer].due_time)
    for n in hosts:
        nd = instance.nodes[n]
        if float(nd.ready_time) <= due and ready <= float(nd.due_time):
            return True
    return False


def relax_charges(instance, routes: list) -> list:
    """Rebuild each route's stations/portions with just-enough charge.

    Baseline ALNS uses full recharge, so host routes have almost no time-window
    slack. Keskin & Çatay (2016): partial charge on the *existing* sequence is
    what frees slack so another customer can fit.
    """
    relaxed = []
    for nodes, labs in routes:
        custs = customers_of(instance, nodes)
        if not custs:
            continue
        plan = fast_hco(instance, custs)
        if plan.feasible:
            relaxed.append((list(plan.nodes), list(plan.labels)))
            continue
        trial_labs = just_enough_labels(instance, nodes)
        check = replay_route(instance, nodes, trial_labs)
        if check.feasible:
            relaxed.append((list(nodes), list(check.labels)))
        else:
            relaxed.append((list(nodes), list(labs)))
    check = replay_solution(instance, relaxed)
    return nonempty(instance, relaxed) if check.feasible else nonempty(instance, clone(routes))


def try_eliminate(instance, routes: list, route_index: int) -> list | None:
    displaced = customers_of(instance, routes[route_index][0])
    if not displaced:
        return None
    remaining = [(list(n), list(l)) for i, (n, l) in enumerate(routes) if i != route_index]
    cap = float(instance.vehicle.load_capacity)

    def difficulty(c: int) -> tuple:
        demand = float(instance.nodes[c].demand)
        geo = nearby_route_ids(instance, remaining, c, limit=16)
        k_i = [
            ri for ri in geo
            if route_load(instance, remaining[ri][0]) + demand <= cap + 1e-9
            and _overlap(instance, c, remaining[ri][0])
        ]
        return (len(k_i), float(instance.nodes[c].due_time), c)

    for cust in sorted(displaced, key=difficulty):
        demand = float(instance.nodes[cust].demand)
        geo = nearby_route_ids(instance, remaining, cust, limit=16)
        allow = [
            ri for ri in geo
            if route_load(instance, remaining[ri][0]) + demand <= cap + 1e-9
            and _overlap(instance, cust, remaining[ri][0])
        ]
        if not insert_just_enough(
            instance, remaining, cust,
            nearby_routes=16, insert_positions=5, nearby_stations=5,
            allow_new_route=False, allowed_routes=allow or None,
        ):
            return None
    check = replay_solution(instance, remaining)
    return nonempty(instance, remaining) if check.feasible else None


def compress_fleet(instance, routes: list, *, deadline_s: float = 8.0, max_candidates: int = 8):
    current = relax_charges(instance, nonempty(instance, clone(routes)))
    log = {"eliminations": 0, "attempts": 0, "relaxed": True}
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < float(deadline_s):
        used = [(i, customers_of(instance, n)) for i, (n, _) in enumerate(current)]
        used = [(i, c) for i, c in used if c]
        if len(used) <= 1:
            break
        ordered = sorted(used, key=lambda item: (len(item[1]), item[0]))
        success = False
        for ri, _custs in ordered[: max(1, max_candidates)]:
            if time.perf_counter() - t0 >= float(deadline_s):
                break
            log["attempts"] += 1
            trial = try_eliminate(instance, current, ri)
            if trial is None:
                continue
            after = replay_solution(instance, trial)
            before_k = sum(1 for _, c in used if c)
            if after.feasible and after.used_k < before_k:
                current = trial
                log["eliminations"] += 1
                success = True
                break
        if not success:
            break
    return current, log
