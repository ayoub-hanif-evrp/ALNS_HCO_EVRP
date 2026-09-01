"""Phase I: ALNS-PR. The incumbent fleet size is K*."""
from __future__ import annotations

from src.alns_hco.alns_pr import solve_alns_pr
from src.alns_hco.routes import nonempty


def solve_phase1(
    instance,
    *,
    time_limit_s: float = 15.0,
    seed: int = 0,
    start_routes: list | None = None,
    compress_s: float = 0.0,
):
    del compress_s
    alns = solve_alns_pr(instance, start_routes, time_limit_s=float(time_limit_s), seed=int(seed))
    routes = nonempty(instance, alns["routes"])
    metrics = alns["metrics"]
    return {
        "routes": routes,
        "metrics": metrics,
        "runtime_s": float(alns["runtime_s"]),
        "iterations": int(alns.get("iterations") or 0),
        "accepted": int(alns.get("accepted") or 0),
        "k_star": int(metrics.used_k),
        "d_star": float(metrics.distance),
        "t_star": float(metrics.duration),
        "alns_k": int(metrics.used_k),
        "eliminations": 0,
    }
