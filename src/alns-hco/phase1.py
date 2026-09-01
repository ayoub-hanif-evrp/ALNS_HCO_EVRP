"""Phase I: the same GI + 15 s routing ALNS as the ALNS baseline, then just-enough fleet compression.

The standalone ALNS already wins on K and distance among the 15 s methods. Phase I
must inherit that solution (GI + 15 s ALNS), then try to pack further with
partial charge (Keskin & Çatay 2016; Wang et al. 2026 fleet-first ALNS).
"""
from __future__ import annotations

from src.alns_hco.fleet import compress_fleet
from src.alns_hco.routes import clone, evaluate, nonempty
from src.hanif.benchmarks.alns import solve_alns
from src.hanif.benchmarks.greedy import solve_greedy


def solve_phase1(
    instance,
    *,
    time_limit_s: float = 15.0,
    seed: int = 0,
    start_routes: list | None = None,
    compress_s: float = 8.0,
):
    if start_routes:
        initial = nonempty(instance, clone(start_routes))
        gi_runtime = 0.0
    else:
        gi = solve_greedy(instance, deadline=40.0)
        initial = gi["routes"]
        gi_runtime = float(gi["runtime_s"])
    alns = solve_alns(instance, initial, time_limit_s=float(time_limit_s), seed=int(seed))
    routes = nonempty(instance, alns["routes"])
    compressed, clog = compress_fleet(instance, routes, deadline_s=float(compress_s))
    metrics, key = evaluate(instance, compressed)
    if not metrics.feasible:
        compressed, metrics, key = routes, alns["metrics"], evaluate(instance, routes)[1]
        clog = {**clog, "reverted": True}
    return {
        "routes": compressed,
        "metrics": metrics,
        "runtime_s": gi_runtime + float(alns["runtime_s"]) + float(compress_s),
        "iterations": int(alns.get("iterations") or 0),
        "accepted": int(alns.get("accepted") or 0),
        "k_star": int(metrics.used_k),
        "d_star": float(metrics.distance),
        "t_star": float(metrics.duration),
        "alns_k": int(alns["metrics"].used_k),
        "eliminations": int(clog.get("eliminations") or 0),
        "compress": clog,
    }
