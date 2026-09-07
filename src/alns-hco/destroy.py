"""ALNS destroy operators, including normalized charging-aware removals for Phase II."""
from __future__ import annotations

from src.alns_hco.routes import all_customers, clone, customers_of, nonempty, remove_customer
from src.hanif.model import charging_rate, expected_price, expected_wait
from src.hanif.replay import replay_route


def _detour_of(instance, routes, customer: int) -> float:
    dist = instance.distance_matrix
    for nodes, _ in routes:
        if customer not in nodes:
            continue
        i = nodes.index(customer)
        if 0 < i < len(nodes) - 1:
            u, v = nodes[i - 1], nodes[i + 1]
            return float(dist[u, customer]) + float(dist[customer, v]) - float(dist[u, v])
    return 0.0


def destroy_random(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    custs = all_customers(instance, trial)
    k = min(k, len(custs))
    removed = rng.sample(custs, k) if k else []
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def destroy_worst(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    custs = all_customers(instance, trial)
    ranked = sorted(custs, key=lambda c: -_detour_of(instance, trial, c))
    removed = ranked[: min(k, len(ranked))]
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def destroy_related(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    custs = all_customers(instance, trial)
    if not custs:
        return trial, []
    seed = rng.choice(custs)
    dist = instance.distance_matrix
    ranked = sorted(custs, key=lambda c: (float(dist[seed, c]), c))
    removed = ranked[: min(k, len(ranked))]
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def destroy_short_route(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    used = [(i, customers_of(instance, n)) for i, (n, _) in enumerate(trial)]
    used = [(i, c) for i, c in used if c]
    if not used:
        return trial, []
    used.sort(key=lambda t: (len(t[1]), t[0]))
    ri, displaced = used[0]
    removed = list(displaced[: min(k, len(displaced))])
    if len(removed) == len(displaced):
        trial.pop(ri)
    else:
        for c in removed:
            remove_customer(instance, trial, c)
        trial = nonempty(instance, trial)
    return trial, removed


def _normalize(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo <= 1e-12:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _station_region_scores(instance, routes) -> list[tuple[float, int, int]]:
    """Return (undesirability, route_index, station_node) with normalized multi-criteria score."""
    raw = []  # expense, wait, slow, detour, energy, ri, station
    for ri, (nodes, labels) in enumerate(routes):
        plan = replay_route(instance, nodes, labels)
        if not plan.feasible:
            continue
        for i, n in enumerate(plan.nodes):
            if not instance.nodes[n].is_charging_station:
                continue
            prev = plan.nodes[i - 1]
            nxt = plan.nodes[i + 1] if i + 1 < len(plan.nodes) else plan.nodes[-1]
            dist = instance.distance_matrix
            detour = float(dist[prev, n]) + float(dist[n, nxt]) - float(dist[prev, nxt])
            rate = max(charging_rate(instance, n), 1e-9)
            wait = expected_wait(instance, n)
            price = expected_price(instance, n)
            alpha = 0.0
            if plan.labels and i < len(plan.labels) and plan.labels[i] is not None:
                alpha = float(plan.labels[i])
            energy = float(plan.charged_energy) / max(plan.charging_visits, 1)
            expense = energy * price * max(alpha, 0.05)
            slow = 1.0 / rate
            raw.append((expense, wait, slow, detour, energy, ri, int(n)))
    if not raw:
        return []
    exp_n = _normalize([r[0] for r in raw])
    wait_n = _normalize([r[1] for r in raw])
    slow_n = _normalize([r[2] for r in raw])
    det_n = _normalize([r[3] for r in raw])
    eng_n = _normalize([r[4] for r in raw])
    out = []
    for i, r in enumerate(raw):
        score = exp_n[i] + wait_n[i] + slow_n[i] + det_n[i] + eng_n[i]
        out.append((score, r[5], r[6]))
    return out


def _remove_around_station(instance, routes, ri: int, station: int, k: int) -> list[int]:
    nodes, _ = routes[ri]
    if station not in nodes:
        custs = customers_of(instance, nodes)
        return custs[:k]
    pos = nodes.index(station)
    nearby = []
    for n in nodes[max(0, pos - 4): pos + 5]:
        if instance.nodes[n].is_customer and n not in nearby:
            nearby.append(n)
    if len(nearby) < k:
        for n in customers_of(instance, nodes):
            if n not in nearby:
                nearby.append(n)
            if len(nearby) >= k:
                break
    return nearby[:k]


def destroy_undesirable_charge(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    """Remove customers near the most undesirable charging region (normalized multi-criteria)."""
    trial = clone(routes)
    scored = _station_region_scores(instance, trial)
    if not scored:
        return destroy_random(instance, routes, rng, k)
    scored.sort(reverse=True)
    _score, ri, station = scored[0]
    removed = _remove_around_station(instance, trial, ri, station, k)
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def destroy_expensive_charge(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    scored = _station_region_scores(instance, trial)
    if not scored:
        return destroy_random(instance, routes, rng, k)
    # Prefer high expense component: re-score with expense-only using raw order from multi score
    scored.sort(reverse=True)
    _score, ri, station = scored[0]
    removed = _remove_around_station(instance, trial, ri, station, k)
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def destroy_slow_charge(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    return destroy_undesirable_charge(instance, routes, rng, k)


PHASE1_DESTROY = (destroy_random, destroy_worst, destroy_related, destroy_short_route)
PHASE2_DESTROY = PHASE1_DESTROY + (
    destroy_expensive_charge,
    destroy_slow_charge,
    destroy_undesirable_charge,
)
