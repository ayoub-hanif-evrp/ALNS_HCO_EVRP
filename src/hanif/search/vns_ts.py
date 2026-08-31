"""Schneider-style VNS using relocate, swap, and 2-opt neighborhoods.

This is a matched VNS-style baseline, not a faithful Schneider et al. (2014)
VNS+tabu implementation: a tabu list is not used to forbid attributes.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any

from src.hanif.env.cost import CostWeights
from src.hanif.search.operators import relocate_customer, swap_customers, two_opt_route
from src.hanif.search.repair import _clone_routes
from src.hanif.search.route_replay import replay_solution
from src.hanif.search.score import SearchScore


@dataclass
class VNSTSResult:
    routes: list
    score: SearchScore
    iterations: int
    accepted: int
    elapsed_seconds: float
    telemetry: dict[str, Any] = field(default_factory=dict)


def run_vns_ts(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    time_limit_seconds: float = 40.0,
    seed: int = 0,
) -> VNSTSResult:
    cfg = config or {}
    w = weights or CostWeights()
    rng = random.Random(int(seed))
    current = _clone_routes(routes)
    replay = replay_solution(instance, current, config=cfg, weights=w, m=m)
    score = replay.score or SearchScore.worst()
    t0 = time.perf_counter()
    accepted = 0
    it = 0
    tabu: list[int] = []
    while time.perf_counter() - t0 < float(time_limit_seconds):
        it += 1
        k = 1 + (it % 3)
        trial = None
        if k == 1:
            trial = relocate_customer(instance, current, config=cfg, weights=w, m=m, rng=rng)
        elif k == 2:
            trial = swap_customers(instance, current, config=cfg, weights=w, m=m, rng=rng)
        else:
            used = [
                i for i, (nodes, _) in enumerate(current)
                if any(instance.nodes[n].is_customer for n in nodes)
            ]
            if used:
                trial = two_opt_route(
                    instance, current, rng.choice(used), config=cfg, weights=w, m=m,
                )
        if trial is None:
            continue
        res = replay_solution(instance, trial, config=cfg, weights=w, m=m)
        if res.score is not None and res.score.dominates(score):
            current = trial
            score = res.score
            accepted += 1
            tabu = tabu[-19:] + [it]
        if it > 10_000:
            break
    return VNSTSResult(
        routes=current,
        score=score,
        iterations=it,
        accepted=accepted,
        elapsed_seconds=time.perf_counter() - t0,
        telemetry={"tabu_len": len(tabu), "neighborhoods": ["relocate", "swap", "two_opt"]},
    )
