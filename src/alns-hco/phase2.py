"""Phase II: charging-aware ALNS at frozen K=K*.

Customer sequences may change. The number of vehicles is exactly K*.
Each complete candidate is evaluated by the Heterogeneous Charging Oracle.
"""
from __future__ import annotations

import math
import random
import time

from src.alns_hco.destroy import PHASE2_DESTROY
from src.alns_hco.hco import OracleCache, apply_hco_to_routes
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.insert import charging_aware_reinsert
from src.alns_hco.routes import clone, nonempty, used_k
from src.hanif.replay import replay_solution


def _key(m) -> tuple:
    return (
        0 if m.feasible else 1,
        int(m.unserved),
        int(m.hard_violations),
        float(m.distance),
        float(m.charging_expense),
    )


def solve_phase2(
    instance,
    initial: list,
    *,
    k_star: int,
    d_star: float = 0.0,
    t_star: float = 0.0,
    time_limit_s: float = 15.0,
    seed: int = 0,
    use_exact: bool = True,
    cache: OracleCache | None = None,
):
    del d_star, t_star, use_exact
    hp = HYPERPARAMETERS
    rng = random.Random(int(seed) + 17)
    cache = cache or OracleCache()
    k_star = int(k_star)
    current = nonempty(instance, clone(initial))
    planned = apply_hco_to_routes(instance, current, cache)
    if planned:
        current = planned
    cur_m = replay_solution(instance, current)
    best, best_m, best_key = clone(current), cur_m, _key(cur_m)
    ops = PHASE2_DESTROY
    weights = [1.0] * len(ops)
    scores = [0.0] * len(ops)
    uses = [0] * len(ops)
    temperature = max(1.0, hp["sa_start_temperature_factor"] * max(cur_m.distance, 1.0))
    t0 = time.perf_counter()
    it = accepted = hco_evals = 0

    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        op = rng.choices(range(len(ops)), weights=weights, k=1)[0]
        k = rng.randint(hp["destroy_size_min"], hp["destroy_size_max"])
        wrecked, removed = ops[op](instance, current, rng, k)
        if not removed:
            continue
        order = sorted(removed, key=lambda c: (float(instance.nodes[c].due_time), c))
        if not charging_aware_reinsert(instance, wrecked, order, cache=cache, k_max=k_star):
            continue
        wrecked = nonempty(instance, wrecked)
        if used_k(instance, wrecked) != k_star:
            continue
        planned = apply_hco_to_routes(instance, wrecked, cache)
        hco_evals += 1
        if not planned:
            continue
        wrecked = planned
        if used_k(instance, wrecked) != k_star:
            continue
        m = replay_solution(instance, wrecked)
        if not m.feasible or int(m.used_k) != k_star:
            continue
        uses[op] += 1
        key = _key(m)
        accept = False
        gain = 0.0
        if key < best_key:
            best, best_m, best_key = clone(wrecked), m, key
            current, cur_m = clone(wrecked), m
            accept, gain = True, hp["score_new_best"]
        elif key < _key(cur_m):
            current, cur_m = clone(wrecked), m
            accept, gain = True, hp["score_better"]
        else:
            d_cur, d_new = float(cur_m.distance), float(m.distance)
            delta = (d_new - d_cur) / max(abs(d_cur), 1.0)
            if rng.random() < math.exp(-max(delta, 0.0) / max(temperature, 1e-9)):
                current, cur_m = clone(wrecked), m
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

    polished = apply_hco_to_routes(instance, best, cache)
    if polished:
        pm = replay_solution(instance, polished)
        if pm.feasible and int(pm.used_k) == k_star and _key(pm) <= best_key:
            best, best_m = polished, pm
    return {
        "routes": best,
        "metrics": best_m,
        "runtime_s": time.perf_counter() - t0,
        "iterations": it,
        "accepted": accepted,
        "hco_evals": hco_evals,
        "cache_hits": cache.hits,
        "k_star": k_star,
    }
