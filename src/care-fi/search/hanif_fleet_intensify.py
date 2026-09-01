"""Scalable fleet compression for Hanif 1000-customer instances.

Designed to run *after* a fast feasible dense construction.
Avoids full ALNS / all-route charging DP / set partitioning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.hanif.env.cost import CostWeights
from src.hanif.search.deadline import SearchDeadline
from src.hanif.search.fleet_search import _compact_to_used, _nonempty_route_count, capacity_lower_bound
from src.hanif.search.repair import _clone_routes, regret_insertion, repair_solution
from src.hanif.search.route_replay import replay_route, replay_solution
from src.hanif.search.score import SearchScore


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


@dataclass(frozen=True)
class EliminationCandidate:
    """Constraint-aware estimate of how safely a route can be absorbed."""

    route_index: int
    risk: float
    n_customers: int
    nearest_centroid: float
    mean_tw_slack: float
    load_utilization: float
    charging_burden: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "route_index": int(self.route_index),
            "risk": float(self.risk),
            "n_customers": int(self.n_customers),
            "nearest_centroid": float(self.nearest_centroid),
            "mean_tw_slack": float(self.mean_tw_slack),
            "load_utilization": float(self.load_utilization),
            "charging_burden": float(self.charging_burden),
        }


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo <= 1e-12:
        return [0.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def _rank_legacy_routes(
    instance,
    routes: list,
    *,
    top_k: int = 12,
    seed: int = 0,
) -> list[int]:
    """Frozen direct-recovery ranking retained as the primary operator."""
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
        near = (
            sum(((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 for ox, oy in others)
            / len(others)
            if others else 1e9
        )
        tie = (i * 31 + seed * 17) % 997
        scored.append(((n_c, near, tie), i))
    scored.sort()
    return [i for _, i in scored[: max(1, top_k)]]


def rank_constraint_aware_routes(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    top_k: int = 12,
    seed: int = 0,
) -> list[EliminationCandidate]:
    """Rank routes by bounded reinsertion risk.

    A low score denotes a small, geographically overlapping, temporally loose,
    lightly loaded route with few charging visits.  Every component is derived
    from the saved incumbent; no benchmark target or future solution is used.
    """
    raw: list[dict[str, float | int]] = []
    centroids: dict[int, tuple[float, float]] = {}
    cap = max(float(instance.vehicle.load_capacity), 1e-9)

    for i, (nodes, labels) in enumerate(routes):
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        if not custs:
            continue
        cx = sum(float(instance.nodes[c].x) for c in custs) / len(custs)
        cy = sum(float(instance.nodes[c].y) for c in custs) / len(custs)
        centroids[i] = (cx, cy)
        replay = replay_route(instance, nodes, labels, config=config)
        slacks = [
            max(0.0, float(instance.nodes[e.node].due_time) - float(e.service_start))
            for e in replay.route_events
            if instance.nodes[e.node].is_customer
        ]
        load = sum(float(instance.nodes[c].demand) for c in custs)
        charges = sum(1 for n in nodes if instance.nodes[n].is_charging_station)
        raw.append({
            "route_index": i,
            "n_customers": len(custs),
            "mean_tw_slack": sum(slacks) / max(len(slacks), 1),
            "load_utilization": min(1.0, max(0.0, load / cap)),
            "charging_burden": charges / max(len(custs), 1),
        })

    for item in raw:
        i = int(item["route_index"])
        cx, cy = centroids[i]
        distances = [
            ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
            for j, (ox, oy) in centroids.items()
            if j != i
        ]
        item["nearest_centroid"] = min(distances) if distances else 0.0

    sizes = _normalize([float(x["n_customers"]) for x in raw])
    overlaps = _normalize([float(x["nearest_centroid"]) for x in raw])
    slacks = _normalize([float(x["mean_tw_slack"]) for x in raw])
    loads = _normalize([float(x["load_utilization"]) for x in raw])
    charges = _normalize([float(x["charging_burden"]) for x in raw])
    ranked: list[tuple[float, int, EliminationCandidate]] = []
    for pos, item in enumerate(raw):
        # Larger observed slack lowers risk; all other normalized terms raise it.
        risk = (
            0.40 * sizes[pos]
            + 0.20 * overlaps[pos]
            + 0.20 * (1.0 - slacks[pos])
            + 0.10 * loads[pos]
            + 0.10 * charges[pos]
        )
        ri = int(item["route_index"])
        candidate = EliminationCandidate(
            route_index=ri,
            risk=risk,
            n_customers=int(item["n_customers"]),
            nearest_centroid=float(item["nearest_centroid"]),
            mean_tw_slack=float(item["mean_tw_slack"]),
            load_utilization=float(item["load_utilization"]),
            charging_burden=float(item["charging_burden"]),
        )
        tie = (ri * 31 + int(seed) * 17) % 997
        ranked.append((risk, tie, candidate))
    ranked.sort(key=lambda x: (x[0], x[1], x[2].route_index))
    return [x[2] for x in ranked[: max(1, top_k)]]


def _try_eliminate_route(
    instance,
    routes: list,
    route_index: int,
    *,
    config: dict,
    weights: CostWeights,
    target_m: int,
    deadline: SearchDeadline | None,
    enable_regret: bool = True,
) -> tuple[list | None, SearchScore | None, dict[str, Any]]:
    """Eject one route and compare fast recovery with bounded regret repair."""
    from src.hanif.search.fast_construct import _recover_unserved_fast

    meta: dict[str, Any] = {
        "operator": "none",
        "ejected": 0,
        "direct_remaining": 0,
        "regret_insertions": 0,
        "regret_remaining": 0,
    }
    if deadline is not None and deadline.expired():
        return None, None, meta
    base = _clone_routes(routes)
    if route_index < 0 or route_index >= len(base):
        return None, None, meta
    removed = base.pop(route_index)
    ejected = [n for n in removed[0] if instance.nodes[n].is_customer]
    meta["ejected"] = len(ejected)
    while len(base) < target_m:
        depot = instance.depot_index
        base.append(([depot, depot], [None, None]))
    base = base[:target_m]

    # Candidate 1 preserves the old O(nk) recovery as a cheap incumbent.
    direct, still = _recover_unserved_fast(
        instance, _clone_routes(base), list(ejected), m=target_m, config=config,
        weights=weights, deadline=deadline,
    )
    meta["direct_remaining"] = len(still)
    candidates: list[tuple[str, list, SearchScore]] = []
    if still and (deadline is None or deadline.remaining() > 2.0):
        direct, direct_repair = repair_solution(
            instance,
            direct,
            config=config,
            weights=weights,
            m=target_m,
            max_rounds=10,
            regret_k=3,
            station_candidates=2,
            mode="regret",
            deadline=deadline,
        )
        if not direct_repair.feasible and (deadline is None or not deadline.expired()):
            served = set(
                replay_solution(
                    instance, direct, config=config, weights=weights, m=target_m,
                ).served_customers
            )
            leftovers = [
                i for i, node in enumerate(instance.nodes)
                if node.is_customer and i not in served
            ]
            if leftovers:
                direct, leftovers = _recover_unserved_fast(
                    instance,
                    direct,
                    leftovers,
                    m=target_m,
                    config=config,
                    weights=weights,
                    deadline=deadline,
                )
                meta["direct_remaining"] = len(leftovers)
    direct = _compact_to_used(instance, direct)
    used_direct = _nonempty_route_count(instance, direct)
    check_direct = replay_solution(
        instance, direct, config=config, weights=weights, m=used_direct or target_m,
    )
    if check_direct.feasible and check_direct.score is not None:
        # Fleet size is the primary objective. Preserve the proven operator and
        # spend the independent regret attempt only when it cannot absorb K-1.
        meta["operator"] = "direct"
        meta["direct_remaining"] = 0
        return direct, check_direct.score, meta
    if not enable_regret:
        return None, None, meta

    # Candidate 2 starts again from the unmodified K-1 solution and chooses the
    # most constrained ejected customer first through regret-3/2 insertion.
    regret_trial = _clone_routes(base)
    unserved = list(ejected)
    if deadline is None or deadline.remaining() > 0.75:
        max_steps = max(1, len(unserved) * 2)
        for _ in range(max_steps):
            if not unserved or (deadline is not None and deadline.expired()):
                break
            pick = regret_insertion(
                instance,
                regret_trial,
                unserved,
                regret_k=3,
                station_candidates=2,
                config=config,
                weights=weights,
                m=target_m,
                allow_new_route=False,
                deadline=deadline,
                max_full_replays=3,
            )
            if pick is None:
                pick = regret_insertion(
                    instance,
                    regret_trial,
                    unserved,
                    regret_k=2,
                    station_candidates=1,
                    config=config,
                    weights=weights,
                    m=target_m,
                    allow_new_route=False,
                    deadline=deadline,
                    max_full_replays=2,
                )
            if pick is None:
                break
            regret_trial = pick.routes
            unserved.remove(pick.customer)
            meta["regret_insertions"] += 1
    meta["regret_remaining"] = len(unserved)
    if not unserved:
        regret_trial, _ = repair_solution(
            instance,
            regret_trial,
            config=config,
            weights=weights,
            m=target_m,
            max_rounds=3,
            regret_k=3,
            station_candidates=2,
            mode="regret",
            deadline=deadline,
        )
        regret_trial = _compact_to_used(instance, regret_trial)
        used_regret = _nonempty_route_count(instance, regret_trial)
        check_regret = replay_solution(
            instance,
            regret_trial,
            config=config,
            weights=weights,
            m=used_regret or target_m,
        )
        if check_regret.feasible and check_regret.score is not None:
            candidates.append(("regret", regret_trial, check_regret.score))

    if not candidates:
        return None, None, meta
    operator, best_routes, best_score = min(candidates, key=lambda x: x[2].as_tuple())
    meta["operator"] = operator
    return best_routes, best_score, meta


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
            legacy = _rank_legacy_routes(
                instance, current, top_k=top_k, seed=seed,
            )
            # CARE ranking is deliberately lazy: the frozen direct operator
            # receives its full opportunity before fallback feature extraction.
            phases: list[tuple[str, list[tuple[int, EliminationCandidate | None]]]] = [
                ("legacy", [(ri, None) for ri in legacy]),
            ]
            phase_index = 0
            while phase_index < 2 and not success:
                if phase_index == 1:
                    if deadline is not None and deadline.remaining() < 3:
                        break
                    care = rank_constraint_aware_routes(
                        instance, current, config=cfg, top_k=top_k, seed=seed,
                    )
                    phases.append((
                        "care",
                        [
                            (candidate.route_index, candidate)
                            for candidate in care
                            if candidate.route_index not in set(legacy)
                        ],
                    ))
                ranker, candidates = phases[phase_index]
                phase_index += 1
                for ri, candidate in candidates:
                    if deadline is not None and deadline.remaining() < 2:
                        break
                    t_try = time.perf_counter()
                    child = None
                    if deadline is not None:
                        child = deadline.child(
                            min(12.0, max(2.0, deadline.remaining() * 0.15)),
                            stage="elim_try",
                        )
                    new_routes, new_score, attempt_meta = _try_eliminate_route(
                        instance, current, ri,
                        config=cfg, weights=w, target_m=m_now - 1, deadline=child or deadline,
                        enable_regret=ranker == "care",
                    )
                    ok = new_routes is not None and new_score is not None
                    log.rounds.append({
                        "round": rnd,
                        "seed": seed,
                        "route_index": ri,
                        "from_k": m_now,
                        "feasible": ok,
                        "runtime": time.perf_counter() - t_try,
                        "ranker": ranker,
                        "candidate": candidate.as_dict() if candidate is not None else None,
                        **attempt_meta,
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
                from src.hanif.search.fast_construct import dense_construct
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


__all__ = [
    "EliminationCandidate",
    "IntensificationLog",
    "intensify_hanif_fleet",
    "rank_constraint_aware_routes",
]
