"""Keskin-style EVRPTW ALNS for Schneider-scale instances.

Customer/station destroy–repair, adaptive weights, simulated annealing, and
per-route charging DP. Not a line-for-line Keskin 2016 reproduction (no
continuous MILP recharge variable; charge uses discrete portions plus
just-enough). Built to reduce fleet then distance on the 56 large instances.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any

from src.schneider.env.cost import CostWeights
from src.schneider.search.charging_dp import optimize_charging_plan
from src.schneider.search.repair import (
    _clone_routes,
    best_insertion,
    regret_insertion,
    repair_solution,
)
from src.schneider.search.route_replay import replay_solution
from src.schneider.search.score import SearchScore


def _custs(instance, nodes: list[int]) -> list[int]:
    return [n for n in nodes if instance.nodes[n].is_customer]


def _used(instance, routes: list) -> int:
    return sum(1 for nodes, _ in routes if _custs(instance, nodes))


def _replan_charges(instance, routes: list, *, config, deadline=None) -> list:
    out = []
    for nodes, labels in routes:
        if deadline is not None and deadline.expired():
            out.append((list(nodes), list(labels)))
            continue
        seq = _custs(instance, nodes)
        if not seq:
            depot = instance.depot_index
            out.append(([depot, depot], [None, None]))
            continue
        plan = optimize_charging_plan(instance, seq, config=config, deadline=deadline)
        if plan.feasible and plan.nodes:
            out.append((list(plan.nodes), list(plan.labels)))
        else:
            out.append((list(nodes), list(labels)))
    return out


def _eval(instance, routes, *, config, weights, m=None) -> tuple[list, SearchScore, bool]:
    used = _used(instance, routes) or 1
    target = int(m) if m is not None else used
    res = replay_solution(instance, routes, config=config, weights=weights, m=target)
    return routes, res.score or SearchScore.worst(), bool(res.feasible)


def _remove_customers(instance, routes, customers: list[int]) -> list:
    want = set(customers)
    out = []
    for nodes, labels in routes:
        labels = list(labels) if labels else [None] * len(nodes)
        keep_n, keep_l = [], []
        for n, lab in zip(nodes, labels):
            if n in want:
                continue
            keep_n.append(n)
            keep_l.append(lab)
        if len(keep_n) < 2:
            depot = instance.depot_index
            out.append(([depot, depot], [None, None]))
        else:
            out.append((keep_n, keep_l))
    return out


def _all_customers(instance, routes) -> list[int]:
    out = []
    for nodes, _ in routes:
        out.extend(_custs(instance, nodes))
    return out


def destroy_random(instance, routes, k: int, rng: random.Random) -> tuple[list, list[int]]:
    pool = _all_customers(instance, routes)
    rng.shuffle(pool)
    removed = pool[: max(1, min(k, len(pool)))]
    return _remove_customers(instance, routes, removed), removed


def destroy_worst(instance, routes, k: int, rng: random.Random) -> tuple[list, list[int]]:
    scored: list[tuple[float, int]] = []
    dist = instance.distance_matrix
    for nodes, _ in routes:
        for i, n in enumerate(nodes):
            if not instance.nodes[n].is_customer:
                continue
            prev, nxt = nodes[i - 1], nodes[i + 1]
            extra = float(dist[prev, n]) + float(dist[n, nxt]) - float(dist[prev, nxt])
            scored.append((extra, int(n)))
    scored.sort(reverse=True)
    # Slight noise so the same customers are not always stripped.
    top = scored[: max(k * 2, k)]
    rng.shuffle(top)
    removed = [c for _, c in top[:k]]
    return _remove_customers(instance, routes, removed), removed


def destroy_shaw(instance, routes, k: int, rng: random.Random) -> tuple[list, list[int]]:
    pool = _all_customers(instance, routes)
    if not pool:
        return _clone_routes(routes), []
    seed = rng.choice(pool)
    dist = instance.distance_matrix
    related = []
    for c in pool:
        d = float(dist[seed, c])
        tw = abs(float(instance.nodes[seed].ready_time) - float(instance.nodes[c].ready_time))
        related.append((d + 0.15 * tw, c))
    related.sort()
    removed = [c for _, c in related[: max(1, k)]]
    return _remove_customers(instance, routes, removed), removed


def destroy_time(instance, routes, k: int, rng: random.Random) -> tuple[list, list[int]]:
    pool = _all_customers(instance, routes)
    pool.sort(key=lambda c: (float(instance.nodes[c].due_time), c))
    start = rng.randrange(max(1, len(pool) - k + 1)) if len(pool) > k else 0
    removed = pool[start : start + k]
    return _remove_customers(instance, routes, removed), removed


def destroy_route(instance, routes, rng: random.Random) -> tuple[list, list[int]]:
    used = [i for i, (nodes, _) in enumerate(routes) if _custs(instance, nodes)]
    if not used:
        return _clone_routes(routes), []
    # Prefer a small route (fleet-reduction destroy).
    used.sort(key=lambda i: (len(_custs(instance, routes[i][0])), i))
    ri = used[0] if rng.random() < 0.55 else rng.choice(used)
    removed = _custs(instance, routes[ri][0])
    trial = _clone_routes(routes)
    depot = instance.depot_index
    trial[ri] = ([depot, depot], [None, None])
    return trial, removed


def destroy_stations(instance, routes, rng: random.Random) -> tuple[list, list[int]]:
    """Strip stations from one route; charging DP will re-insert."""
    used = [i for i, (nodes, _) in enumerate(routes) if _custs(instance, nodes)]
    if not used:
        return _clone_routes(routes), []
    ri = rng.choice(used)
    trial = _clone_routes(routes)
    nodes, labels = trial[ri]
    keep_n, keep_l = [], []
    for n, lab in zip(nodes, labels if labels else [None] * len(nodes)):
        if instance.nodes[n].is_charging_station:
            continue
        keep_n.append(n)
        keep_l.append(None if instance.nodes[n].is_customer or n == instance.depot_index else lab)
    if len(keep_n) < 2:
        depot = instance.depot_index
        trial[ri] = ([depot, depot], [None, None])
    else:
        trial[ri] = (keep_n, keep_l)
    return trial, []


def _repair(instance, routes, removed, *, config, weights, m, deadline) -> list:
    trial = _clone_routes(routes)
    leftover = list(removed)
    st_k = 2
    for _ in range(max(1, len(leftover) * 2)):
        if not leftover or (deadline is not None and deadline.expired()):
            break
        pick = regret_insertion(
            instance, trial, leftover, regret_k=3, station_candidates=st_k,
            config=config, weights=weights, m=m, allow_new_route=False,
            deadline=deadline, max_full_replays=2,
        )
        if pick is None:
            pick = best_insertion(
                instance, trial, leftover[0], station_candidates=st_k,
                config=config, weights=weights, m=m, allow_new_route=False,
                deadline=deadline, max_full_replays=2,
            )
        if pick is None:
            break
        trial = pick.routes
        if pick.customer in leftover:
            leftover.remove(pick.customer)
    if leftover:
        trial, _ = repair_solution(
            instance, trial, config=config, weights=weights, m=m,
            max_rounds=min(8, len(leftover) + 2), regret_k=2, station_candidates=1,
            mode="regret", deadline=deadline,
        )
    return _replan_charges(instance, trial, config=config, deadline=deadline)


@dataclass
class SchneiderALNSResult:
    routes: list
    score: SearchScore
    iterations: int
    accepted: int
    elapsed_seconds: float
    telemetry: dict[str, Any] = field(default_factory=dict)


def run_schneider_alns(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    time_limit_seconds: float = 60.0,
    seed: int = 0,
    destroy_k: int = 8,
    prefer_fleet_cut: bool = False,
) -> SchneiderALNSResult:
    cfg = config or {}
    w = weights or CostWeights()
    rng = random.Random(int(seed))
    current = _clone_routes(routes)
    current = _replan_charges(instance, current, config=cfg)
    _, score, feas = _eval(instance, current, config=cfg, weights=w, m=m)
    if not feas:
        current, score, feas = _eval(instance, routes, config=cfg, weights=w, m=m)
        current = _clone_routes(routes)
    best = _clone_routes(current)
    best_score = score
    ops = {
        "random": 1.0,
        "worst": 1.0,
        "shaw": 1.0,
        "time": 1.0,
        "route": 2.4 if prefer_fleet_cut else 0.8,
        "stations": 0.6,
    }
    scores_seg = {k: 0.0 for k in ops}
    uses_seg = {k: 0 for k in ops}
    accepted = 0
    it = 0
    t0 = time.perf_counter()
    temp0 = max(5.0, 0.05 * max(float(score.total_distance), 1.0))

    while time.perf_counter() - t0 < float(time_limit_seconds):
        it += 1
        rem = float(time_limit_seconds) - (time.perf_counter() - t0)
        if rem < 0.4:
            break
        names = list(ops)
        total = sum(ops[n] for n in names) or 1.0
        r = rng.random() * total
        acc = 0.0
        op = names[0]
        for n in names:
            acc += ops[n]
            if r <= acc:
                op = n
                break
        k = max(4, int(destroy_k + rng.randint(-2, 4)))
        k = min(k, max(4, instance.num_customers // 8))
        if op == "random":
            trial, removed = destroy_random(instance, current, k, rng)
        elif op == "worst":
            trial, removed = destroy_worst(instance, current, k, rng)
        elif op == "shaw":
            trial, removed = destroy_shaw(instance, current, k, rng)
        elif op == "time":
            trial, removed = destroy_time(instance, current, k, rng)
        elif op == "route":
            trial, removed = destroy_route(instance, current, rng)
        else:
            trial, removed = destroy_stations(instance, current, rng)
        trial = _repair(
            instance, trial, removed, config=cfg, weights=w, m=m,
            deadline=None,
        )
        _, tscore, tfeas = _eval(instance, trial, config=cfg, weights=w, m=m)
        uses_seg[op] += 1
        improved = tscore.dominates(score)
        better_best = tscore.dominates(best_score)
        accept = False
        if tfeas and (improved or better_best):
            accept = True
            scores_seg[op] += 9.0 if better_best else 3.0
        elif tfeas:
            temp = temp0 * (0.995 ** it)
            delta = float(tscore.total_distance) - float(score.total_distance)
            # Allow a worse-distance same-K move; never accept more vehicles or unserved.
            same_k = tscore.used_evs <= score.used_evs and tscore.unserved_customer_count <= score.unserved_customer_count
            if same_k and tscore.hard_violation_count <= score.hard_violation_count:
                if delta <= 0 or rng.random() < math.exp(-max(delta, 0.0) / max(temp, 1e-6)):
                    accept = True
                    scores_seg[op] += 1.0
        if accept:
            current, score = trial, tscore
            accepted += 1
            if tscore.dominates(best_score):
                best, best_score = _clone_routes(trial), tscore
        if it % 20 == 0:
            for n in ops:
                if uses_seg[n]:
                    ops[n] = 0.7 * ops[n] + 0.3 * (scores_seg[n] / uses_seg[n])
                    ops[n] = min(8.0, max(0.15, ops[n]))
                scores_seg[n] = 0.0
                uses_seg[n] = 0

    return SchneiderALNSResult(
        routes=best,
        score=best_score,
        iterations=it,
        accepted=accepted,
        elapsed_seconds=time.perf_counter() - t0,
        telemetry={"op_weights": dict(ops)},
    )
