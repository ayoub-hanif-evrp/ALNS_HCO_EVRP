"""Feasible insertion with minimum required partial charge (Phase I) and HCO-guided repair (Phase II)."""
from __future__ import annotations

import time

from src.alns_hco.hco import (
    ensure_explicit_labels, hco, just_enough_labels, plan_min_charge_stations,
    replay_explicit, station_candidates,
)
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.routes import customers_of, used_k
from src.hanif.construction import (
    cheapest_positions,
    customer_order,
    delta_d,
    nearby_route_ids,
    nearest_stations,
    tw_optimistic,
)
from src.hanif.replay import replay_route

NEARBY_ROUTES = 8
INSERT_POSITIONS = 3
NEARBY_STATIONS = 3


def _try_plain(instance, seq: list[int]):
    if not tw_optimistic(instance, seq):
        return None
    plan = replay_route(instance, seq, default_portion=1.0)
    return plan if plan.feasible else None


def _try_plain_partial(instance, seq: list[int]):
    """Replay with just-enough on existing stations, then full recharge as fallback (Phase I only)."""
    if not tw_optimistic(instance, seq):
        return None
    labs = just_enough_labels(instance, seq)
    plan = replay_explicit(instance, seq, labs)
    if plan.feasible:
        return plan
    plan = replay_route(instance, seq, default_portion=1.0)
    return plan if plan.feasible else None


def _try_min_charge(instance, seq: list[int], customer: int, stations: list[int]):
    try:
        cpos = seq.index(customer)
    except ValueError:
        return None
    nxt = next((n for n in seq[cpos + 1:] if instance.nodes[n].is_customer or n == instance.depot_index), instance.depot_index)
    best = None
    for s in stations:
        for insert_at in (cpos, cpos + 1):
            prev, after = seq[insert_at - 1], seq[insert_at]
            if instance.nodes[prev].is_charging_station or instance.nodes[after].is_charging_station:
                continue
            trial = seq[:insert_at] + [s] + seq[insert_at:]
            plan = plan_min_charge_stations(instance, trial, insert_at, nxt)
            if plan.feasible and (best is None or plan.distance < best.distance):
                best = plan
    return best


def _try_just_enough(instance, seq: list[int], customer: int, stations: list[int]):
    """Explicit just-enough on the new station; preserve other stations via remap in plan_min_charge."""
    try:
        cpos = seq.index(customer)
    except ValueError:
        return None
    nxt = next(
        (n for n in seq[cpos + 1:] if instance.nodes[n].is_customer or n == instance.depot_index),
        instance.depot_index,
    )
    best = None
    for s in stations:
        for insert_at in (cpos, cpos + 1):
            prev, after = seq[insert_at - 1], seq[insert_at]
            if instance.nodes[prev].is_charging_station or instance.nodes[after].is_charging_station:
                continue
            trial = seq[:insert_at] + [s] + seq[insert_at:]
            # Explicit just-enough for *all* stations on the trial (no None→1.0).
            labs = just_enough_labels(instance, trial)
            plan = replay_explicit(instance, trial, labs)
            if not plan.feasible:
                # Explicit full charge only at the new station; recompute others just-enough.
                labs = just_enough_labels(instance, trial)
                labs[insert_at] = 1.0
                plan = replay_explicit(instance, trial, labs)
            if plan.feasible and (best is None or plan.distance < best.distance):
                best = plan
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
    """Phase I insert: min feasible detour, min required charge if a station is needed."""
    depot = int(instance.depot_index)
    stations = nearest_stations(instance, customer, nearby_stations)
    candidates = nearby_route_ids(instance, routes, customer, limit=nearby_routes)
    if allowed_routes is not None:
        allow = set(allowed_routes)
        candidates = [ri for ri in candidates if ri in allow]

    best = None
    for ri in candidates:
        nodes, _ = routes[ri]
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
            nodes, _ = routes[ri]
            for pos in cheapest_positions(instance, nodes, customer, insert_positions):
                seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
                plan = _try_min_charge(instance, seq, customer, stations)
                if plan is None:
                    continue
                key = (plan.distance, ri, pos)
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
        opened = _try_min_charge(instance, [depot, customer, depot], customer, stations)
    if opened is None:
        return False
    routes.append((list(opened.nodes), list(opened.labels)))
    return True


def construct_feasible(instance, *, deadline: float | None = None) -> list:
    routes: list = []
    t0 = time.perf_counter()
    for cust in customer_order(instance):
        if deadline is not None and time.perf_counter() - t0 > deadline:
            break
        insert_customer(instance, routes, cust)
    return [(list(n), list(l)) for n, l in routes if any(instance.nodes[x].is_customer for x in n)]


def greedy_reinsert(instance, routes: list, customers: list[int], *, allow_new_route: bool = True) -> bool:
    ok = True
    for c in customers:
        if not insert_customer(instance, routes, c, allow_new_route=allow_new_route):
            ok = False
    return ok


def insert_just_enough(
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
    """Same neighbourhood as ALNS insert, but charge only what the next hop needs."""
    depot = int(instance.depot_index)
    stations = nearest_stations(instance, customer, nearby_stations)
    candidates = nearby_route_ids(instance, routes, customer, limit=nearby_routes)
    if allowed_routes is not None:
        allow = set(allowed_routes)
        candidates = [ri for ri in candidates if ri in allow]

    best = None
    for ri in candidates:
        nodes, _ = routes[ri]
        for pos in cheapest_positions(instance, nodes, customer, insert_positions):
            detour = delta_d(instance, nodes[pos - 1], customer, nodes[pos])
            seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
            plan = _try_plain_partial(instance, seq)
            if plan is None:
                continue
            key = (detour, ri, pos)
            if best is None or key < best[0]:
                best = (key, ri, plan.nodes, plan.labels)
    if best is None:
        for ri in candidates:
            nodes, _ = routes[ri]
            for pos in cheapest_positions(instance, nodes, customer, insert_positions):
                seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
                plan = _try_just_enough(instance, seq, customer, stations)
                if plan is None:
                    continue
                key = (plan.distance, ri, pos)
                if best is None or key < best[0]:
                    best = (key, ri, plan.nodes, plan.labels)
    if best is not None:
        _, ri, new_nodes, new_labs = best
        routes[ri] = (list(new_nodes), list(new_labs))
        return True
    if not allow_new_route:
        return False
    opened = _try_plain_partial(instance, [depot, customer, depot])
    if opened is None:
        opened = _try_just_enough(instance, [depot, customer, depot], customer, stations)
    if opened is None:
        return False
    routes.append((list(opened.nodes), list(opened.labels)))
    return True


def greedy_reinsert_just_enough(
    instance, routes: list, customers: list[int], *, allow_new_route: bool = True,
) -> bool:
    ok = True
    for c in customers:
        if not insert_just_enough(instance, routes, c, allow_new_route=allow_new_route):
            ok = False
    return ok


def _shortlist_insertions(
    instance,
    routes: list,
    customer: int,
    *,
    nearby_routes: int,
    insert_positions: int,
    stations: list[int],
    shortlist_size: int,
) -> list[tuple]:
    """Generate promising feasible (ri, plan) insertions ranked before HCO."""
    candidates = nearby_route_ids(instance, routes, customer, limit=nearby_routes)
    scored: list[tuple] = []
    for ri in candidates:
        nodes, _labels = routes[ri]
        for pos in cheapest_positions(instance, nodes, customer, insert_positions):
            seq = list(nodes[:pos]) + [customer] + list(nodes[pos:])
            plan = _try_plain_partial(instance, seq) or _try_just_enough(instance, seq, customer, stations)
            if plan is None:
                continue
            scored.append((
                float(plan.charging_expense),
                float(plan.distance),
                ri,
                plan,
            ))
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[: max(1, int(shortlist_size))]


def insert_charging_aware(
    instance,
    routes: list,
    customer: int,
    *,
    cache,
    k_max: int,
    nearby_routes: int = NEARBY_ROUTES,
    insert_positions: int = INSERT_POSITIONS,
    hco_mode: str = "heterogeneous",
    candidates_per_criterion: int = 5,
    shortlist_size: int | None = None,
    use_hco: bool = True,
) -> bool:
    """Phase II insert: shortlist hosts → HCO-evaluate each → pick best (or NoHCO ranking)."""
    depot = int(instance.depot_index)
    sl = int(shortlist_size if shortlist_size is not None else HYPERPARAMETERS.get("phase2_repair_shortlist_size", 5))
    mode = "detour" if hco_mode == "detour" else "heterogeneous"
    stations = station_candidates(
        instance, customer, customer, k=candidates_per_criterion, mode=mode,
    )
    shortlist = _shortlist_insertions(
        instance, routes, customer,
        nearby_routes=nearby_routes, insert_positions=insert_positions,
        stations=stations, shortlist_size=sl,
    )

    if shortlist:
        best = None
        for _ce, _d, ri, plan in shortlist:
            if use_hco and hco_mode not in ("plain", "nohco"):
                hat = hco(
                    instance, customers_of(instance, plan.nodes), cache,
                    mode=mode, candidates_per_criterion=candidates_per_criterion,
                    input_nodes=list(plan.nodes), input_labels=list(plan.labels),
                    non_worsening=False,
                )
                if hat.feasible:
                    cand_nodes, cand_labs = list(hat.nodes), list(hat.labels)
                    score = (
                        float(hat.charging_expense),
                        float(hat.station_waiting),
                        float(hat.charging_time),
                        float(hat.distance),
                    )
                else:
                    cand_nodes, cand_labs = list(plan.nodes), list(plan.labels)
                    score = (
                        float(plan.charging_expense),
                        float(plan.station_waiting),
                        float(plan.charging_time),
                        float(plan.distance),
                    )
            else:
                # PhaseII-NoHCO: same shortlist + lex charge ranking, no oracle.
                cand_nodes, cand_labs = list(plan.nodes), list(plan.labels)
                score = (
                    float(plan.charging_expense),
                    float(plan.station_waiting),
                    float(plan.charging_time),
                    float(plan.distance),
                )
            if best is None or score < best[0]:
                best = (score, ri, cand_nodes, cand_labs)
        assert best is not None
        _, ri, new_nodes, new_labs = best
        routes[ri] = (list(new_nodes), ensure_explicit_labels(instance, new_nodes, new_labs))
        return True

    if used_k(instance, routes) >= int(k_max):
        return False
    opened = _try_plain_partial(instance, [depot, customer, depot])
    if opened is None:
        opened = _try_just_enough(instance, [depot, customer, depot], customer, stations)
    if opened is None:
        return False
    if use_hco and hco_mode not in ("plain", "nohco"):
        hat = hco(
            instance, customers_of(instance, opened.nodes), cache,
            mode=mode, candidates_per_criterion=candidates_per_criterion,
            input_nodes=list(opened.nodes), input_labels=list(opened.labels),
            non_worsening=False,
        )
        if hat.feasible:
            routes.append((list(hat.nodes), list(hat.labels)))
            return True
    routes.append((list(opened.nodes), ensure_explicit_labels(instance, opened.nodes, opened.labels)))
    return True


def charging_aware_reinsert(
    instance,
    routes: list,
    customers: list[int],
    *,
    cache,
    k_max: int,
    hco_mode: str = "heterogeneous",
    candidates_per_criterion: int = 5,
    shortlist_size: int | None = None,
    use_hco: bool = True,
) -> bool:
    ok = True
    for c in customers:
        if not insert_charging_aware(
            instance, routes, c, cache=cache, k_max=k_max,
            hco_mode=hco_mode, candidates_per_criterion=candidates_per_criterion,
            shortlist_size=shortlist_size, use_hco=use_hco,
        ):
            ok = False
    return ok
