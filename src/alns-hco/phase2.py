"""Phase II: keep searching with lex (K, D, C). Fleet may decrease; it may not increase."""
from __future__ import annotations

import math
import random
import time

from src.alns_hco.destroy import PHASE2_DESTROY
from src.alns_hco.fleet import compress_fleet
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.insert import greedy_reinsert_just_enough
from src.alns_hco.oracle import OracleCache, apply_oracle_to_routes
from src.alns_hco.routes import clone, nonempty, used_k
from src.hanif.replay import replay_solution


def _key(m) -> tuple:
    return (
        0 if m.feasible else 1,
        int(m.unserved),
        int(m.hard_violations),
        int(m.used_k),
        float(m.distance),
        float(m.charging_expense),
    )


def solve_phase2(
    instance,
    initial: list,
    *,
    k_star: int,
    d_star: float,
    t_star: float,
    time_limit_s: float = 15.0,
    seed: int = 0,
    use_exact: bool = True,
    cache: OracleCache | None = None,
):
    del d_star, t_star
    hp = HYPERPARAMETERS
    rng = random.Random(int(seed) + 17)
    cache = cache or OracleCache()
    current = nonempty(instance, clone(initial))
    cur_m = replay_solution(instance, current)
    best, best_m, best_key = clone(current), cur_m, _key(cur_m)
    k_cap = int(k_star)
    ops = PHASE2_DESTROY
    weights = [1.0] * len(ops)
    scores = [0.0] * len(ops)
    uses = [0] * len(ops)
    temperature = max(1.0, hp["sa_start_temperature_factor"] * max(cur_m.distance, 1.0))
    t0 = time.perf_counter()
    it = accepted = exact_evals = 0

    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        if it % 25 == 0:
            packed, _ = compress_fleet(instance, current, deadline_s=0.6)
            pm = replay_solution(instance, packed)
            if pm.feasible and _key(pm) < best_key:
                current, cur_m = packed, pm
                best, best_m, best_key = clone(packed), pm, _key(pm)
                k_cap = min(k_cap, int(pm.used_k))
        op = rng.choices(range(len(ops)), weights=weights, k=1)[0]
        k = rng.randint(hp["destroy_size_min"], hp["destroy_size_max"])
        wrecked, removed = ops[op](instance, current, rng, k)
        if not removed:
            continue
        order = sorted(removed, key=lambda c: (float(instance.nodes[c].due_time), c))
        if not greedy_reinsert_just_enough(instance, wrecked, order, allow_new_route=True):
            continue
        wrecked = nonempty(instance, wrecked)
        if used_k(instance, wrecked) > k_cap:
            continue
        m = replay_solution(instance, wrecked)
        if not m.feasible:
            continue
        uses[op] += 1
        key = _key(m)
        accept = False
        gain = 0.0
        if key < best_key:
            if use_exact:
                exact_routes = apply_oracle_to_routes(instance, wrecked, exact=True, cache=cache)
                exact_evals += 1
                if exact_routes:
                    em = replay_solution(instance, exact_routes)
                    if em.feasible and em.used_k <= k_cap and _key(em) <= key:
                        wrecked, m, key = exact_routes, em, _key(em)
            best, best_m, best_key = clone(wrecked), m, key
            current, cur_m = clone(wrecked), m
            k_cap = min(k_cap, int(m.used_k))
            accept, gain = True, hp["score_new_best"]
        elif key < _key(cur_m):
            current, cur_m = clone(wrecked), m
            accept, gain = True, hp["score_better"]
        else:
            d_cur, d_new = float(cur_m.distance), float(m.distance)
            delta = (d_new - d_cur) / max(abs(d_cur), 1.0)
            if (m.used_k <= cur_m.used_k
                    and rng.random() < math.exp(-max(delta, 0.0) / max(temperature, 1e-9))):
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

    if use_exact:
        polished = apply_oracle_to_routes(instance, best, exact=True, cache=cache)
        if polished:
            pm = replay_solution(instance, polished)
            if pm.feasible and pm.used_k <= k_cap and _key(pm) <= best_key:
                best, best_m = polished, pm
    return {
        "routes": best,
        "metrics": best_m,
        "runtime_s": time.perf_counter() - t0,
        "iterations": it,
        "accepted": accepted,
        "exact_evals": exact_evals,
        "cache_fast_hits": cache.fast_hits,
        "cache_exact_hits": cache.exact_hits,
        "d_cap": None,
        "t_cap": None,
    }
