"""Greedy due-time insertion constructor (no search)."""
from __future__ import annotations

import time

from src.hanif.benchmarks.common import evaluate, nonempty
from src.hanif.construction import construct_feasible

HYPERPARAMETERS = {
    "name": "Greedy insertion",
    "abbreviation": "GI",
    "customer_order": "increasing due time b_i, then a_i, then index",
    "nearby_routes": 8,
    "insert_positions": 3,
    "nearby_stations": 3,
    "charge_policy": "full recharge only when a nearby station is required for battery feasibility",
    "random_seed": None,
    "time_limit_s": None,
}


def solve_greedy(instance, *, deadline: float | None = 30.0):
    t0 = time.perf_counter()
    routes = construct_feasible(instance, deadline=deadline)
    routes = nonempty(instance, routes)
    metrics, key = evaluate(instance, routes)
    return {
        "routes": routes,
        "metrics": metrics,
        "key": key,
        "runtime_s": time.perf_counter() - t0,
        "iterations": len(routes),
        "accepted": 0,
    }
