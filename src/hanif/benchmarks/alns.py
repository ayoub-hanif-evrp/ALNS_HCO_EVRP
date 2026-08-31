"""ALNS-style destroy/repair with adaptive weights and SA acceptance.

Matched internal baseline. Not Keskin & Çatay (2016) ALNS.
"""
from __future__ import annotations

import math
import random
import time

from src.hanif.benchmarks.common import (
    all_customers,
    better,
    clone,
    customers_of,
    evaluate,
    greedy_reinsert,
    nonempty,
    remove_customer,
)

HYPERPARAMETERS = {
    "name": "Adaptive large neighbourhood search",
    "abbreviation": "ALNS",
    "random_seed": 0,
    "time_limit_s": 15.0,
    "destroy_size_min": 4,
    "destroy_size_max": 12,
    "destroy_operators": ["random", "worst_detour", "related_shaw", "short_route"],
    "repair": "greedy cheapest feasible insertion (nearby_routes=8, positions=3)",
    "adaptive_reaction_factor_rho": 0.2,
    "segment_length": 20,
    "score_new_best": 33,
    "score_better": 9,
    "score_accepted": 5,
    "sa_start_temperature_factor": 0.05,
    "sa_cooling": 0.9995,
    "charge_policy": "full recharge when a station is required",
    "accept_infeasible": False,
}


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


def _destroy_random(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    custs = all_customers(instance, trial)
    k = min(k, len(custs))
    removed = rng.sample(custs, k) if k else []
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def _destroy_worst(instance, routes, rng, k: int) -> tuple[list, list[int]]:
    trial = clone(routes)
    custs = all_customers(instance, trial)
    ranked = sorted(custs, key=lambda c: -_detour_of(instance, trial, c))
    removed = ranked[: min(k, len(ranked))]
    for c in removed:
        remove_customer(instance, trial, c)
    return nonempty(instance, trial), removed


def _destroy_related(instance, routes, rng, k: int) -> tuple[list, list[int]]:
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


def _destroy_short_route(instance, routes, rng, k: int) -> tuple[list, list[int]]:
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


DESTROY = (_destroy_random, _destroy_worst, _destroy_related, _destroy_short_route)


def solve_alns(instance, initial: list, *, time_limit_s: float = 15.0, seed: int = 0):
    rng = random.Random(int(seed))
    current = nonempty(instance, clone(initial))
    cur_m, cur_key = evaluate(instance, current)
    best, best_key, best_m = clone(current), cur_key, cur_m
    weights = [1.0] * len(DESTROY)
    scores = [0.0] * len(DESTROY)
    uses = [0] * len(DESTROY)
    t0 = time.perf_counter()
    temperature = max(1.0, HYPERPARAMETERS["sa_start_temperature_factor"] * max(cur_m.distance, 1.0))
    cooling = HYPERPARAMETERS["sa_cooling"]
    it = accepted = 0
    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        op = rng.choices(range(len(DESTROY)), weights=weights, k=1)[0]
        k = rng.randint(HYPERPARAMETERS["destroy_size_min"], HYPERPARAMETERS["destroy_size_max"])
        wrecked, removed = DESTROY[op](instance, current, rng, k)
        if not removed:
            continue
        order = sorted(removed, key=lambda c: (float(instance.nodes[c].due_time), c))
        if not greedy_reinsert(instance, wrecked, order):
            continue
        wrecked = nonempty(instance, wrecked)
        m, key = evaluate(instance, wrecked)
        if not m.feasible:
            continue
        uses[op] += 1
        accept = False
        gain = 0.0
        if better(key, best_key):
            best, best_key, best_m = clone(wrecked), key, m
            current, cur_key, cur_m = clone(wrecked), key, m
            accept, gain = True, HYPERPARAMETERS["score_new_best"]
        elif better(key, cur_key):
            current, cur_key, cur_m = clone(wrecked), key, m
            accept, gain = True, HYPERPARAMETERS["score_better"]
        else:
            d_cur, d_new = float(cur_key[4]), float(key[4])
            delta = (d_new - d_cur) / max(abs(d_cur), 1.0)
            if (key[0] == 0 and key[1] <= cur_key[1] and key[2] <= cur_key[2]
                    and key[3] <= cur_key[3]
                    and rng.random() < math.exp(-max(delta, 0.0) / max(temperature, 1e-9))):
                current, cur_key, cur_m = clone(wrecked), key, m
                accept, gain = True, HYPERPARAMETERS["score_accepted"]
        if accept:
            accepted += 1
            scores[op] += gain
        temperature *= cooling
        if it % HYPERPARAMETERS["segment_length"] == 0:
            rho = HYPERPARAMETERS["adaptive_reaction_factor_rho"]
            for i in range(len(weights)):
                if uses[i]:
                    weights[i] = (1.0 - rho) * weights[i] + rho * (scores[i] / uses[i])
                    scores[i] = 0.0
                    uses[i] = 0
            if min(weights) <= 1e-9:
                weights = [max(w, 0.05) for w in weights]
    return {
        "routes": best,
        "metrics": best_m,
        "key": best_key,
        "runtime_s": time.perf_counter() - t0,
        "iterations": it,
        "accepted": accepted,
    }
