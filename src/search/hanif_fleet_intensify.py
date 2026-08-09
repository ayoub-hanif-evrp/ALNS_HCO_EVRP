"""Scalable fleet compression for Hanif 1000-customer instances.

Designed to run *after* a fast feasible dense construction.
Avoids full ALNS / all-route charging DP / set partitioning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.search.deadline import SearchDeadline
from src.search.fleet_search import _compact_to_used, _nonempty_route_count, capacity_lower_bound
from src.search.repair import _clone_routes, repair_solution
from src.search.route_replay import replay_solution
from src.search.score import SearchScore


@dataclass
class IntensificationLog:
    rounds: list[dict[str, Any]] = field(default_factory=list)
    initial_k: int = 0
    final_k: int = 0
    eliminations: int = 0
    runtime_s: float = 0.0

    def as_dict(self) -> dict:
        return {
            "initial_k": self.initial_k,
            "final_k": self.final_k,
            "eliminations": self.eliminations,
            "runtime_s": self.runtime_s,
            "rounds": list(self.rounds),
        }


def _rank_smallest_routes(instance, routes: list, *, top_k: int = 12, seed: int = 0) -> list[int]:
    scored: list[tuple[tuple, int]] = []
    centroids: list[tuple[int, float, float, int]] = []
    for i, (nodes, _) in enumerate(routes):
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        if not custs:
            continue
        cx = sum(float(instance.nodes[c].x) for c in custs) / len(custs)
        cy = sum(float(instance.nodes[c].y) for c in custs) / len(custs)
        centroids.append((i, cx, cy, len(custs)))
    for i, cx, cy, n_c in centroids:
        others = [(ox, oy) for j, ox, oy, _ in centroids if j != i]
        if others:
            # Smaller mean distance => easier reinsertion into nearby routes.
            near = sum(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 for ox, oy in others) / len(others)
        else:
            near = 1e9
        tie = (i * 31 + seed * 17) % 997
        # Prefer tiny + geographically overlapping routes.
        scored.append(((n_c, near, tie), i))
    scored.sort()
    return [i for _, i in scored[: max(1, top_k)]]


def _try_eliminate_route(
    instance,
    routes: list,
    route_index: int,
    *,
    config: dict,
    weights: CostWeights,
    target_m: int,
    deadline: SearchDeadline | None,
) -> tuple[list | None, SearchScore | None]:
    """Eject one route and reinsert customers with fast Hanif recovery."""
    from src.search.fast_construct import _recover_unserved_fast

    if deadline is not None and deadline.expired():
        return None, None
    base = _clone_routes(routes)
    if route_index < 0 or route_index >= len(base):
        return None, None
    removed = base.pop(route_index)
    ejected = [n for n in removed[0] if instance.nodes[n].is_customer]
    while len(base) < target_m:
        depot = instance.depot_index
        base.append(([depot, depot], [None, None]))
    base = base[:target_m]

    trial, still = _recover_unserved_fast(
        instance, base, ejected, m=target_m, config=config, weights=weights, deadline=deadline,
    )
    if still and (deadline is None or deadline.remaining() > 2):
        trial, rep = repair_solution(
            instance, trial, config=config, weights=weights, m=target_m,
            max_rounds=10, regret_k=3, station_candidates=2, mode="regret", deadline=deadline,
        )
        if not rep.feasible:
            # One more recover pass on leftovers
            final0 = replay_solution(instance, trial, config=config, weights=weights, m=target_m)
            left = [
                i for i, n in enumerate(instance.nodes)
                if n.is_customer and i not in set(final0.served_customers)
            ]
            if left:
                trial, left = _recover_unserved_fast(
                    instance, trial, left, m=target_m, config=config, weights=weights, deadline=deadline,
                )

    trial = _compact_to_used(instance, trial)
    used = _nonempty_route_count(instance, trial)
    if used <= 0 or used > target_m:
        # Pad if compact somehow dropped too far — still must be feasible at used.
        pass
    m_check = used if used > 0 else target_m
    final = replay_solution(instance, trial, config=config, weights=weights, m=m_check)
    if not final.feasible or final.score is None:
        return None, None
    if int(final.score.unserved_customer_count) != 0:
        return None, None
    return trial, final.score


def intensify_hanif_fleet(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    deadline: SearchDeadline | None = None,
    max_rounds: int = 50,
    top_k: int = 10,
    seeds: list[int] | None = None,
    lower_bound: int | None = None,
) -> tuple[list, SearchScore, IntensificationLog]:
    """Repeatedly eliminate routes until no candidate succeeds or budget expires.

    Uses fast station-aware reinsertion only (no full ALNS / global charging DP).
    Multi-seed only rotates elimination candidate order — does not rebuild.
    """
    cfg = config or {}
    w = weights or CostWeights()
    log = IntensificationLog()
    t0 = time.perf_counter()

    current = _compact_to_used(instance, _clone_routes(routes))
    used = _nonempty_route_count(instance, current)
    m_now = int(m) if m is not None else used
    m_now = min(m_now, used) if used > 0 else m_now
    final0 = replay_solution(instance, current, config=cfg, weights=w, m=m_now)
    score = final0.score or SearchScore.worst()
    if not final0.feasible:
        log.initial_k = m_now
        log.final_k = m_now
        log.runtime_s = time.perf_counter() - t0
        return current, score, log

    current = _compact_to_used(instance, current)
    m_now = _nonempty_route_count(instance, current) or m_now
    score = replay_solution(instance, current, config=cfg, weights=w, m=m_now).score or score
    log.initial_k = int(m_now)

    lo = int(lower_bound) if lower_bound is not None else capacity_lower_bound(instance)
    order_seeds = list(seeds or [0, 1, 2, 3])
    stagnant = 0

    for rnd in range(max(1, max_rounds)):
        if deadline is not None and deadline.remaining() < 3:
            break
        if m_now <= lo:
            break
        success = False
        for seed in order_seeds:
            if deadline is not None and deadline.expired():
                break
            cands = _rank_smallest_routes(instance, current, top_k=top_k, seed=seed)
            for ri in cands:
                if deadline is not None and deadline.remaining() < 2:
                    break
                t_try = time.perf_counter()
                child = None
                if deadline is not None:
                    child = deadline.child(
                        min(12.0, max(2.0, deadline.remaining() * 0.15)),
                        stage="elim_try",
                    )
                new_routes, new_score = _try_eliminate_route(
                    instance, current, ri,
                    config=cfg, weights=w, target_m=m_now - 1, deadline=child or deadline,
                )
                ok = new_routes is not None and new_score is not None
                log.rounds.append({
                    "round": rnd,
                    "seed": seed,
                    "route_index": ri,
                    "from_k": m_now,
                    "feasible": ok,
                    "runtime": time.perf_counter() - t_try,
                })
                if ok:
                    current = _compact_to_used(instance, new_routes)
                    m_now = _nonempty_route_count(instance, current) or (m_now - 1)
                    score = new_score
                    # Re-validate at compacted size
                    conf = replay_solution(instance, current, config=cfg, weights=w, m=m_now)
                    if conf.feasible and conf.score is not None:
                        score = conf.score
                        m_now = _nonempty_route_count(instance, current) or m_now
                    log.eliminations += 1
                    success = True
                    stagnant = 0
                    break
            if success:
                break
        if not success:
            stagnant += 1
            # If still far above capacity LB, try a cold rebuild at a mid fleet size.
            if stagnant >= 1 and m_now > max(lo + 15, int(1.25 * lo)):
                from src.search.fast_construct import dense_construct
                targets = sorted({
                    max(lo, int(round(0.85 * m_now))),
                    max(lo, (lo + m_now) // 2),
                    max(lo, int(round(0.75 * m_now))),
                }, reverse=True)
                rebuilt = False
                for target in targets:
                    if target >= m_now - 2:
                        continue
                    if deadline is not None and deadline.remaining() < 15:
                        break
                    t_rb = time.perf_counter()
                    child = deadline.child(
                        min(45.0, deadline.remaining() * 0.4), stage="rebuild"
                    ) if deadline is not None else None
                    r2, sc2, feas2 = dense_construct(
                        instance, target, seed=rnd + target, config=cfg, weights=w, deadline=child,
                    )
                    r2 = _compact_to_used(instance, r2)
                    used2 = _nonempty_route_count(instance, r2)
                    conf = replay_solution(instance, r2, config=cfg, weights=w, m=used2 or target)
                    ok = bool(conf.feasible) and conf.score is not None and used2 < m_now
                    log.rounds.append({
                        "round": rnd, "seed": -1, "route_index": -1,
                        "from_k": m_now, "feasible": ok, "runtime": time.perf_counter() - t_rb,
                        "rebuild_target": target, "rebuild_used": used2,
                    })
                    if ok:
                        current, score, m_now = r2, conf.score, used2
                        log.eliminations += max(0, log.initial_k - m_now)  # informational
                        rebuilt = True
                        stagnant = 0
                        success = True
                        break
                if not rebuilt:
                    break
            else:
                break

    log.final_k = int(m_now)
    log.runtime_s = time.perf_counter() - t0
    return current, score, log


__all__ = ["IntensificationLog", "intensify_hanif_fleet"]
