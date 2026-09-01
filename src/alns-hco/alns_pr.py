"""ALNS with just-enough partial recharge (ALNS-PR).

Same destroy/repair loop as the full-recharge ALNS baseline; only the charge
policy changes. This is Phase I of ALNS-HCO and the equal-budget baseline.
"""
from __future__ import annotations

import math
import random
import time

from src.alns_hco.destroy import PHASE1_DESTROY
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.insert import greedy_reinsert_just_enough
from src.alns_hco.routes import clone, evaluate, nonempty
from src.hanif.benchmarks.greedy import solve_greedy


def solve_alns_pr(instance, initial: list | None = None, *, time_limit_s: float = 30.0, seed: int = 0):
    hp = HYPERPARAMETERS
    gi_runtime = 0.0
    if initial is None:
        gi = solve_greedy(instance, deadline=40.0)
        initial = gi["routes"]
        gi_runtime = float(gi["runtime_s"])
    rng = random.Random(int(seed))
    current = nonempty(instance, clone(initial))
    cur_m, cur_key = evaluate(instance, current)
    best, best_key, best_m = clone(current), cur_key, cur_m
    ops = PHASE1_DESTROY
    weights = [1.0] * len(ops)
    scores = [0.0] * len(ops)
    uses = [0] * len(ops)
    t0 = time.perf_counter()
    temperature = max(1.0, hp["sa_start_temperature_factor"] * max(cur_m.distance, 1.0))
    it = accepted = 0
    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        op = rng.choices(range(len(ops)), weights=weights, k=1)[0]
        k = rng.randint(hp["destroy_size_min"], hp["destroy_size_max"])
        wrecked, removed = ops[op](instance, current, rng, k)
        if not removed:
            continue
        order = sorted(removed, key=lambda c: (float(instance.nodes[c].due_time), c))
        if not greedy_reinsert_just_enough(instance, wrecked, order, allow_new_route=True):
            continue
        wrecked = nonempty(instance, wrecked)
        m, key = evaluate(instance, wrecked)
        if not m.feasible:
            continue
        uses[op] += 1
        accept = False
        gain = 0.0
        if key < best_key:
            best, best_key, best_m = clone(wrecked), key, m
            current, cur_key, cur_m = clone(wrecked), key, m
            accept, gain = True, hp["score_new_best"]
        elif key < cur_key:
            current, cur_key, cur_m = clone(wrecked), key, m
            accept, gain = True, hp["score_better"]
        else:
            d_cur, d_new = float(cur_key[4]), float(key[4])
            delta = (d_new - d_cur) / max(abs(d_cur), 1.0)
            if (key[0] == 0 and key[1] <= cur_key[1] and key[2] <= cur_key[2]
                    and key[3] <= cur_key[3]
                    and rng.random() < math.exp(-max(delta, 0.0) / max(temperature, 1e-9))):
                current, cur_key, cur_m = clone(wrecked), key, m
                accept, gain = True, hp["score_accepted"]
        if accept:
            accepted += 1
            scores[op] += gain
        temperature *= hp["sa_cooling"]
        if it % hp["segment_length"] == 0:
            rho = hp["adaptive_reaction_factor_rho"]
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
        "runtime_s": gi_runtime + (time.perf_counter() - t0),
        "search_s": time.perf_counter() - t0,
        "gi_runtime_s": gi_runtime,
        "iterations": it,
        "accepted": accepted,
    }
