"""Phase II: fixed-fleet ALNS at K=K*.

Modes:
  - plain: general (Phase-I) destroy + just-enough repair, no HCO
  - nohco: same Phase-II destroy/repair shortlist/acceptance as HCO modes, but no oracle
  - detour / heterogeneous: charging-aware shortlist repair + HCO of that mode

Ranking (explicit, no time-cost conversion):
  Reject if infeasible, K!=K*, or D > (1+eps) D_PR.
  Among admissible candidates: lex (C_ch, T_wait, T_ch, D).
"""
from __future__ import annotations

import math
import random
import time
from typing import Literal

from src.alns_hco.destroy import PHASE1_DESTROY, PHASE2_DESTROY
from src.alns_hco.hco import apply_hco_to_routes
from src.alns_hco.hco_activity import OracleCache
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.insert import charging_aware_reinsert, greedy_reinsert_just_enough
from src.alns_hco.routes import clone, nonempty, used_k
from src.hanif.replay import replay_solution

Phase2Mode = Literal["plain", "nohco", "detour", "heterogeneous"]


def phase2_admissible(m, *, k_star: int, d_pr: float, eps: float) -> bool:
    if not m.feasible:
        return False
    if int(m.used_k) != int(k_star):
        return False
    limit = (1.0 + float(eps)) * float(d_pr)
    return float(m.distance) <= limit + 1e-9


def phase2_key(m) -> tuple:
    return (
        float(m.charging_expense),
        float(m.station_waiting),
        float(m.charging_time),
        float(m.distance),
    )


def solve_phase2(
    instance,
    initial: list,
    *,
    k_star: int,
    d_pr: float,
    time_limit_s: float = 15.0,
    seed: int = 0,
    cache: OracleCache | None = None,
    hco_mode: Phase2Mode = "heterogeneous",
    distance_tolerance: float | None = None,
    candidates_per_criterion: int | None = None,
    repair_shortlist_size: int | None = None,
    d_star: float = 0.0,
    t_star: float = 0.0,
    use_exact: bool = True,
):
    del d_star, t_star, use_exact
    hp = HYPERPARAMETERS
    eps = float(hp["phase2_distance_tolerance"] if distance_tolerance is None else distance_tolerance)
    k_cand = int(hp["hco_candidates_per_criterion"] if candidates_per_criterion is None else candidates_per_criterion)
    shortlist = int(
        hp.get("phase2_repair_shortlist_size", 5) if repair_shortlist_size is None else repair_shortlist_size
    )
    plain = hco_mode == "plain"
    nohco = hco_mode == "nohco"
    use_hco = hco_mode in ("detour", "heterogeneous")
    rng = random.Random(int(seed) + 17)
    cache = cache or OracleCache()
    k_star = int(k_star)
    d_pr = float(d_pr)

    current = nonempty(instance, clone(initial))
    if use_hco:
        planned = apply_hco_to_routes(
            instance, current, cache, mode=hco_mode, candidates_per_criterion=k_cand,
            non_worsening=True,
        )
        if planned and used_k(instance, planned) == k_star:
            current = planned
    cur_m = replay_solution(instance, current)
    if not phase2_admissible(cur_m, k_star=k_star, d_pr=d_pr, eps=eps):
        current = nonempty(instance, clone(initial))
        cur_m = replay_solution(instance, current)

    best, best_m, best_key = clone(current), cur_m, phase2_key(cur_m)
    # Plain uses Phase-I destroy; NoHCO and HCO modes share Phase-II destroy.
    ops = PHASE1_DESTROY if plain else PHASE2_DESTROY
    weights = [1.0] * len(ops)
    scores = [0.0] * len(ops)
    uses = [0] * len(ops)
    temperature = max(1.0, hp["sa_start_temperature_factor"] * max(cur_m.charging_expense, 1.0))
    t0 = time.perf_counter()
    it = accepted = hco_evals = rejected_distance = 0

    while time.perf_counter() - t0 < float(time_limit_s):
        it += 1
        op = rng.choices(range(len(ops)), weights=weights, k=1)[0]
        k = rng.randint(hp["destroy_size_min"], hp["destroy_size_max"])
        wrecked, removed = ops[op](instance, current, rng, k)
        if not removed:
            continue
        order = sorted(removed, key=lambda c: (float(instance.nodes[c].due_time), c))
        if plain:
            if not greedy_reinsert_just_enough(instance, wrecked, order, allow_new_route=False):
                continue
        else:
            # NoHCO and HCO: identical shortlist repair; only use_hco differs.
            if not charging_aware_reinsert(
                instance, wrecked, order, cache=cache, k_max=k_star,
                hco_mode="heterogeneous" if nohco else hco_mode,
                candidates_per_criterion=k_cand,
                shortlist_size=shortlist,
                use_hco=use_hco,
            ):
                continue
        wrecked = nonempty(instance, wrecked)
        if used_k(instance, wrecked) != k_star:
            continue
        if use_hco:
            planned = apply_hco_to_routes(
                instance, wrecked, cache, mode=hco_mode, candidates_per_criterion=k_cand,
                non_worsening=True,
            )
            hco_evals += 1
            if not planned:
                continue
            wrecked = planned
            if used_k(instance, wrecked) != k_star:
                continue
        m = replay_solution(instance, wrecked)
        if not phase2_admissible(m, k_star=k_star, d_pr=d_pr, eps=eps):
            rejected_distance += 1
            continue
        # Assert fleet freeze after every accepted-admissible candidate
        assert int(m.used_k) == k_star
        uses[op] += 1
        key = phase2_key(m)
        accept = False
        gain = 0.0
        if key < best_key:
            best, best_m, best_key = clone(wrecked), m, key
            current, cur_m = clone(wrecked), m
            accept, gain = True, hp["score_new_best"]
        elif key < phase2_key(cur_m):
            current, cur_m = clone(wrecked), m
            accept, gain = True, hp["score_better"]
        else:
            c_cur, c_new = float(cur_m.charging_expense), float(m.charging_expense)
            delta = (c_new - c_cur) / max(abs(c_cur), 1.0)
            if rng.random() < math.exp(-max(delta, 0.0) / max(temperature, 1e-9)):
                current, cur_m = clone(wrecked), m
                accept, gain = True, hp["score_accepted"]
        if accept:
            accepted += 1
            scores[op] += gain
            assert used_k(instance, current) == k_star
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

    if use_hco:
        polished = apply_hco_to_routes(
            instance, best, cache, mode=hco_mode, candidates_per_criterion=k_cand,
            non_worsening=True,
        )
        if polished and used_k(instance, polished) == k_star:
            pm = replay_solution(instance, polished)
            if phase2_admissible(pm, k_star=k_star, d_pr=d_pr, eps=eps) and phase2_key(pm) <= best_key:
                best, best_m = polished, pm

    assert used_k(instance, best) == k_star
    return {
        "routes": best,
        "metrics": best_m,
        "runtime_s": time.perf_counter() - t0,
        "iterations": it,
        "accepted": accepted,
        "hco_evals": hco_evals,
        "rejected_distance": rejected_distance,
        "cache_hits": cache.hits,
        "cache_calls": cache.calls,
        "hco_activity": cache.activity.as_dict(),
        "k_star": k_star,
        "d_pr": d_pr,
        "eps": eps,
        "hco_mode": hco_mode,
        "plain": plain,
        "nohco": nohco,
        "repair_shortlist_size": shortlist,
    }
