"""Simulated annealing with relocate, swap, and 2-opt moves."""
from __future__ import annotations

import math
import random
import time

from src.hanif.benchmarks.common import better, clone, evaluate, nonempty, relocate_once, swap_once, two_opt_once

HYPERPARAMETERS = {
    "name": "Simulated annealing",
    "abbreviation": "SA",
    "random_seed": 0,
    "time_limit_s": 15.0,
    "initial_temperature": 200.0,
    "cooling_rate": 0.995,
    "minimum_temperature": 0.5,
    "move_probabilities": {"relocate": 0.50, "swap": 0.30, "two_opt": 0.20},
    "accept_infeasible": False,
    "charge_policy": "full recharge when a station is required",
}


def solve_sa(instance, initial: list, *, time_limit_s: float = 15.0, seed: int = 0):
    rng = random.Random(int(seed))
    current = nonempty(instance, clone(initial))
    cur_m, cur_key = evaluate(instance, current)
    best, best_key, best_m = clone(current), cur_key, cur_m
    t0 = time.perf_counter()
    temp = float(HYPERPARAMETERS["initial_temperature"])
    cool = float(HYPERPARAMETERS["cooling_rate"])
    tmin = float(HYPERPARAMETERS["minimum_temperature"])
    it = accepted = 0
    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        u = rng.random()
        if u < 0.50:
            cand = relocate_once(instance, current, rng)
        elif u < 0.80:
            cand = swap_once(instance, current, rng)
        else:
            cand = two_opt_once(instance, current, rng)
        if cand is None:
            temp = max(tmin, temp * cool)
            continue
        m, key = evaluate(instance, cand)
        if not m.feasible:
            temp = max(tmin, temp * cool)
            continue
        take = False
        if better(key, cur_key):
            take = True
        elif key[0] == cur_key[0] and key[1] <= cur_key[1] and key[2] <= cur_key[2] and key[3] <= cur_key[3]:
            delta = (float(key[4]) - float(cur_key[4])) / max(abs(float(cur_key[4])), 1.0)
            take = rng.random() < math.exp(-max(delta, 0.0) * 100.0 / max(temp, 1e-9))
        if take:
            current, cur_key, cur_m = cand, key, m
            accepted += 1
            if better(key, best_key):
                best, best_key, best_m = clone(current), key, m
        temp = max(tmin, temp * cool)
    return {
        "routes": best,
        "metrics": best_m,
        "key": best_key,
        "runtime_s": time.perf_counter() - t0,
        "iterations": it,
        "accepted": accepted,
    }
