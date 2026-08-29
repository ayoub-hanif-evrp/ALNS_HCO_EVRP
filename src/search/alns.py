"""ALNS-style destroy/repair with lexicographic acceptance and exact-replay validation.

Matched baseline inspired by Keskin & Çatay neighbourhood families, not a
faithful reproduction of their 2016 ALNS (continuous recharge, adaptive weights,
full operator set).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.search.operators import (
    destroy_critical_tw,
    destroy_random_customers,
    destroy_route,
    relocate_customer,
    swap_customers,
    two_opt_route,
)
from src.search.repair import (
    _clone_routes,
    best_insertion,
    remove_station_if_redundant,
    repair_solution,
)
from src.search.route_replay import replay_solution
from src.search.score import SearchScore


@dataclass
class ALNSResult:
    routes: list[tuple[list[int], list[float | None]]]
    score: SearchScore
    iterations: int
    accepted: int
    elapsed_seconds: float
    history: list[dict] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)


def _destroy_ranked_customers(
    instance,
    routes,
    customers_ranked: list[int],
    k: int,
) -> tuple[list, list[int]]:
    """Remove up to k customers following neural/expert ranking."""
    current = _clone_routes(routes)
    want = set(customers_ranked[: max(0, k)])
    removed: list[int] = []
    by_route: dict[int, list[int]] = {}
    for ri, (nodes, _) in enumerate(current):
        for pos, n in enumerate(nodes):
            if n in want and instance.nodes[n].is_customer:
                by_route.setdefault(ri, []).append(pos)
                removed.append(int(n))
    for ri, positions in by_route.items():
        nodes, labels = current[ri]
        labels = list(labels) if labels else [None] * len(nodes)
        for pos in sorted(positions, reverse=True):
            del nodes[pos]
            if pos < len(labels):
                del labels[pos]
        current[ri] = (nodes, labels)
    return current, removed


def run_alns(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    iterations: int | None = None,
    time_limit_seconds: float | None = None,
    seed: int = 0,
    destroy_k: int = 4,
    regret_k: int = 2,
    guide=None,
    move_logger=None,
) -> ALNSResult:
    """Run exact-replay ALNS; rejected moves roll back. Optional neural guidance.

    Explicit ``iterations`` / ``time_limit_seconds`` always override config when
    provided (None → fall back to hybrid_search.alns.*).
    """
    cfg = config or {}
    hs = (cfg.get("hybrid_search") or {}).get("alns") or {}
    iterations = int(hs.get("iterations", 100) if iterations is None else iterations)
    time_limit_seconds = float(
        hs.get("time_limit_seconds", 30.0) if time_limit_seconds is None else time_limit_seconds
    )
    rng = random.Random(int(seed))

    current = _clone_routes(routes)
    cur_res = replay_solution(instance, current, config=config, weights=weights, m=m)
    assert cur_res.score is not None
    best = _clone_routes(current)
    best_score = cur_res.score
    current_score = cur_res.score
    elite_pool: list[tuple[list, SearchScore]] = [(best, best_score)]

    weights_ops = {
        "random_customers": 1.0,
        "critical_tw": 1.0,
        "route_removal": 0.5,
        "relocate": 1.0,
        "swap": 0.8,
        "two_opt": 0.5,
        "station_clean": 0.3,
    }
    accepted = 0
    history: list[dict] = []
    t0 = time.perf_counter()
    neural_accepted = 0

    for it in range(iterations):
        if time.perf_counter() - t0 > time_limit_seconds:
            break

        if guide is not None and hasattr(guide, "set_solution_context"):
            guide.set_solution_context(current, current_score)

        ops = list(weights_ops.keys())
        if guide is not None:
            if hasattr(guide, "rank_destroy_operators"):
                try:
                    ordered = guide.rank_destroy_operators(
                        ops, instance=instance, routes=current,
                    )
                except TypeError:
                    ordered = guide.rank_destroy_operators(ops)
            else:
                ordered = ops
            # Conservative mix: search-move guide mainly ranks removals/routes;
            # operator choice stays mostly adaptive-weight (avoids BC-style collapse).
            mix_p = 0.25
            if hasattr(guide, "net") and hasattr(guide, "checkpoint_path"):
                mix_p = 0.25
            if rng.random() < mix_p and ordered:
                op = ordered[0]
            else:
                wts = [weights_ops[o] for o in ops]
                total = sum(wts) or 1.0
                r = rng.random() * total
                acc = 0.0
                op = ops[0]
                for o, w in zip(ops, wts):
                    acc += w
                    if r <= acc:
                        op = o
                        break
            used_neural_choice = True
        else:
            wts = [weights_ops[o] for o in ops]
            total = sum(wts) or 1.0
            r = rng.random() * total
            acc = 0.0
            op = ops[0]
            for o, w in zip(ops, wts):
                acc += w
                if r <= acc:
                    op = o
                    break
            used_neural_choice = False

        trial = _clone_routes(current)
        removed: list[int] = []
        if op == "random_customers":
            if guide is not None:
                pool = []
                for nodes, _ in trial:
                    for n in nodes:
                        if instance.nodes[n].is_customer:
                            pool.append(int(n))
                try:
                    ranked = guide.rank_customers_for_removal(
                        instance, pool, limit=destroy_k * 3, routes=trial,
                    )
                except TypeError:
                    ranked = guide.rank_customers_for_removal(instance, pool, limit=destroy_k * 3)
                trial, removed = _destroy_ranked_customers(instance, trial, ranked, destroy_k)
            else:
                trial, removed = destroy_random_customers(instance, trial, destroy_k, rng)
            trial, _ = repair_solution(
                instance, trial, config=config, weights=weights, m=m,
                max_rounds=max(len(removed), 1), regret_k=regret_k, mode="regret",
            )
        elif op == "critical_tw":
            if guide is not None:
                pool = []
                for nodes, _ in trial:
                    for n in nodes:
                        if instance.nodes[n].is_customer:
                            pool.append(int(n))
                try:
                    ranked = guide.rank_customers_for_removal(
                        instance, pool, limit=destroy_k * 3, routes=trial,
                    )
                except TypeError:
                    ranked = guide.rank_customers_for_removal(instance, pool, limit=destroy_k * 3)
                trial, removed = _destroy_ranked_customers(instance, trial, ranked, destroy_k)
            else:
                trial, removed = destroy_critical_tw(instance, trial, destroy_k, rng)
            trial, _ = repair_solution(
                instance, trial, config=config, weights=weights, m=m,
                max_rounds=max(len(removed), 1), regret_k=regret_k, mode="critical",
            )
        elif op == "route_removal":
            if guide is not None:
                ranked_routes = guide.rank_route_indices(instance, trial, prefer_tight=True)
                used = [i for i in ranked_routes if any(instance.nodes[n].is_customer for n in trial[i][0])]
                if used:
                    ri = used[0]
                    nodes, _ = trial[ri]
                    removed = [n for n in nodes if instance.nodes[n].is_customer]
                    depot = instance.depot_index
                    trial[ri] = ([depot, depot], [None, None])
                else:
                    trial, removed = destroy_route(instance, trial, rng)
            else:
                trial, removed = destroy_route(instance, trial, rng)
            trial, _ = repair_solution(
                instance, trial, config=config, weights=weights, m=m,
                max_rounds=max(len(removed), 1), regret_k=regret_k, mode="regret",
            )
        elif op == "relocate":
            moved = relocate_customer(instance, trial, config=config, weights=weights, m=m, rng=rng)
            if moved is None:
                continue
            trial = moved
        elif op == "swap":
            moved = swap_customers(instance, trial, config=config, weights=weights, m=m, rng=rng)
            if moved is None:
                continue
            trial = moved
        elif op == "two_opt":
            if guide is not None:
                used = guide.rank_route_indices(instance, trial, prefer_tight=False)
                used = [i for i in used if any(instance.nodes[n].is_customer for n in trial[i][0])]
            else:
                used = [i for i, (nodes, _) in enumerate(trial) if any(instance.nodes[n].is_customer for n in nodes)]
            if not used:
                continue
            moved = two_opt_route(instance, trial, used[0] if guide else rng.choice(used), config=config, weights=weights, m=m)
            if moved is None:
                continue
            trial = moved
        elif op == "station_clean":
            trial = remove_station_if_redundant(instance, trial, config=config, weights=weights, m=m)

        # Optional: neural-guided single-customer reinsertion attempt for leftovers.
        if guide is not None and removed:
            unserved_probe = removed[:1]
            for cust in unserved_probe:
                # Collect a few insertion options via best_insertion only (exact).
                cand = best_insertion(instance, trial, cust, config=config, weights=weights, m=m, station_candidates=3)
                if cand is not None:
                    guide.rank_insertion_candidates([cand])
                    guide.telemetry.candidates_replayed += 1

        res = replay_solution(instance, trial, config=config, weights=weights, m=m)
        if guide is not None:
            guide.telemetry.candidates_replayed += 1
        if res.score is None:
            continue
        score_before_tuple = list(current_score.as_tuple())
        routes_before = current
        score_before_obj = current_score
        accept = res.score.dominates(current_score)
        if not accept and res.score.hard_violation_count == current_score.hard_violation_count:
            if res.score.unserved_customer_count == current_score.unserved_customer_count:
                delta = res.score.operational_scalar() - current_score.operational_scalar()
                temp = max(1e-6, 1.0 * (0.995 ** it))
                scale = max(abs(current_score.operational_scalar()), 1.0)
                if delta <= 0 or rng.random() < pow(2.718281828, -delta / (temp * scale)):
                    accept = True
        lex_improved = bool(res.score.dominates(score_before_obj))
        if accept:
            current = trial
            current_score = res.score
            accepted += 1
            if used_neural_choice and guide is not None:
                neural_accepted += 1
                guide.telemetry.accepted_neural_guided_moves += 1
            if current_score.dominates(best_score):
                best = _clone_routes(current)
                best_score = current_score
            elite_pool.append((_clone_routes(current), current_score))
            if len(elite_pool) > 8:
                # Neural/elite selection: keep lex-best elites.
                if guide is not None:
                    idx = guide.choose_elite(
                        [e[0] for e in elite_pool],
                        [e[1] for e in elite_pool],
                    )
                    elite_pool = sorted(elite_pool, key=lambda e: e[1].as_tuple())[:8]
                    _ = idx
                else:
                    elite_pool = sorted(elite_pool, key=lambda e: e[1].as_tuple())[:8]
            weights_ops[op] = weights_ops.get(op, 1.0) * 1.05
        else:
            weights_ops[op] = max(0.2, weights_ops.get(op, 1.0) * 0.95)
        hist = {
            "iter": it,
            "op": op,
            "accepted": accept,
            "neural": bool(used_neural_choice),
            "removed": [int(x) for x in removed],
            "score_before": score_before_tuple,
            "score_after": list(res.score.as_tuple()),
            "lex_improved": lex_improved,
            "score": list(current_score.as_tuple()),
        }
        history.append(hist)
        if move_logger is not None:
            try:
                move_logger(
                    {
                        "instance": str(getattr(instance, "instance_name", "")),
                        "iter": it,
                        "op": op,
                        "accepted": bool(accept),
                        "lex_improved": lex_improved,
                        "removed": [int(x) for x in removed],
                        "score_before": score_before_tuple,
                        "score_after": list(res.score.as_tuple()),
                        "routes": routes_before,
                        "score_obj": score_before_obj,
                    }
                )
            except Exception:
                pass

    if guide is not None:
        guide.telemetry.search_seconds += time.perf_counter() - t0

    return ALNSResult(
        routes=best,
        score=best_score,
        iterations=len(history),
        accepted=accepted,
        elapsed_seconds=time.perf_counter() - t0,
        history=history[-50:],
        telemetry={
            "neural_guided": guide is not None,
            "accepted_neural_guided_moves": neural_accepted,
            **(guide.telemetry.as_dict() if guide is not None else {}),
        },
    )
