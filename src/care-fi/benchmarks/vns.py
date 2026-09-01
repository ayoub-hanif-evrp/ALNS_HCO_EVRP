"""Variable neighbourhood search: relocate, swap, then intra-route 2-opt.

Matched internal baseline. Not Schneider et al. (2014) VNS–TS.
"""
from __future__ import annotations

import random
import time

from src.hanif.benchmarks.common import better, clone, evaluate, nonempty, relocate_once, swap_once, two_opt_once

HYPERPARAMETERS = {
    "name": "Variable neighbourhood search",
    "abbreviation": "VNS",
    "random_seed": 0,
    "time_limit_s": 15.0,
    "k_max": 3,
    "neighbourhoods": ["relocate", "swap", "intra_route_2opt"],
    "shake": "k random relocations, k = 1..k_max",
    "first_improvement": True,
    "attempts_per_neighbourhood": 12,
    "charge_policy": "full recharge when a station is required",
}


def solve_vns(instance, initial: list, *, time_limit_s: float = 15.0, seed: int = 0):
    rng = random.Random(int(seed))
    current = nonempty(instance, clone(initial))
    cur_m, cur_key = evaluate(instance, current)
    best, best_key, best_m = clone(current), cur_key, cur_m
    ops = (relocate_once, swap_once, two_opt_once)
    t0 = time.perf_counter()
    it = accepted = 0
    k = 1
    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        shaken = clone(current)
        for _ in range(k):
            trial = relocate_once(instance, shaken, rng)
            if trial is not None:
                shaken = trial
        improved = False
        for op in ops:
            for _ in range(HYPERPARAMETERS["attempts_per_neighbourhood"]):
                if time.perf_counter() - t0 >= float(time_limit_s):
                    break
                cand = op(instance, shaken, rng)
                if cand is None:
                    continue
                m, key = evaluate(instance, cand)
                if m.feasible and better(key, cur_key):
                    current, cur_key, cur_m = cand, key, m
                    shaken = clone(current)
                    accepted += 1
                    improved = True
                    if better(key, best_key):
                        best, best_key, best_m = clone(current), key, m
                    break
            if improved:
                break
        k = 1 if improved else min(k + 1, HYPERPARAMETERS["k_max"])
    return {
        "routes": best,
        "metrics": best_m,
        "key": best_key,
        "runtime_s": time.perf_counter() - t0,
        "iterations": it,
        "accepted": accepted,
    }
