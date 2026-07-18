#!/usr/bin/env python3
"""Benchmark reference vs EV-state-first optimized decision analysis.

Usage:
  python scripts/benchmark_decision_analysis.py
  python scripts/benchmark_decision_analysis.py --repeats 20
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.instance_loader import load_instance
from src.env.decision_analysis import (
    analyze_decision_state,
    get_analysis_call_counts,
    reset_analysis_call_counts,
)
from src.env.evrptw_env import EVRPTWEnv
from src.env.recharge_search import hop_limit_hit_count, reset_hop_limit_hits


DATASET = ROOT / "dataset" / "Schneider" / "evrptw_instances"


def _find(name: str) -> Path | None:
    if not DATASET.exists():
        return None
    hits = list(DATASET.rglob(f"{name}.txt")) + list(DATASET.rglob(name))
    return hits[0] if hits else None


def _env(path: Path, *, stress_recharge: bool = False) -> EVRPTWEnv:
    inst = load_instance(path)
    env = EVRPTWEnv(
        inst,
        m=3,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
        use_depot_charging=True,
        use_horizon_check=True,
    )
    env.environment_config = {
        "implementation": "optimized",
        "cache": {"enabled": False, "cache_by_state_version": False},
        "recharge_search": {"max_station_hops": 3, "use_dominance_pruning": True},
        "action_pruning": {"enabled": True},
    }
    env.reset()
    if stress_recharge:
        # Mid-episode-like: low battery so many customers need recharge chains.
        for ev in env.evs:
            ev.remaining_battery = min(ev.remaining_battery, float(inst.vehicle.battery_capacity) * 0.15)
    return env


def _time_analysis(env: EVRPTWEnv, implementation: str, repeats: int) -> list[float]:
    times = []
    for _ in range(repeats):
        env._decision_analysis_cache = {}
        env._ev_analysis_cache = {}
        env._global_serviceability_cache = None
        env._action_mask_cache = {}
        env.last_decision_analysis = None
        env.last_global_serviceability = None
        ev = env.select_acting_ev()
        t0 = time.perf_counter()
        analyze_decision_state(env, ev, implementation=implementation)
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def _stats(xs: list[float]) -> dict:
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    p95 = xs_sorted[min(n - 1, int(0.95 * (n - 1)))]
    return {
        "mean": statistics.mean(xs),
        "median": statistics.median(xs),
        "p95": p95,
        "max": max(xs),
        "n": n,
    }


def _run_suite(label: str, stress_recharge: bool, repeats: int) -> tuple[list[float], list[float]]:
    targets = [
        ("C", "c101_21"),
        ("R", "r101_21"),
        ("RC", "rc101_21"),
    ]
    overall_ref: list[float] = []
    overall_opt: list[float] = []
    print(f"\n--- Suite: {label} ---")
    for cls, name in targets:
        path = _find(name)
        if path is None:
            alt = _find(name.replace("_21", "")) or _find(f"{cls.lower()}101")
            path = alt
        if path is None:
            print(f"[{cls}] SKIP — instance {name} not found")
            continue

        env = _env(path, stress_recharge=stress_recharge)
        reset_analysis_call_counts()
        reset_hop_limit_hits()
        ref_t = _time_analysis(env, "reference", repeats)
        opt_t = _time_analysis(env, "optimized", repeats)
        overall_ref.extend(ref_t)
        overall_opt.extend(opt_t)

        rs = _stats(ref_t)
        os_ = _stats(opt_t)
        speedup = rs["median"] / max(os_["median"], 1e-9)
        counts = get_analysis_call_counts()
        metrics = getattr(env, "_cache_metrics", {}) or {}
        print(f"\n[{cls}] {path.name}")
        print(f"  reference median: {rs['median']:.2f} ms  (p95={rs['p95']:.2f}, max={rs['max']:.2f})")
        print(f"  optimized median: {os_['median']:.2f} ms  (p95={os_['p95']:.2f}, max={os_['max']:.2f})")
        print(f"  speedup (median): {speedup:.2f}x")
        print(
            f"  call counts: recharge~={counts.get('recharge_search')} "
            f"global_svc={counts.get('global_serviceability')} "
            f"analyze={counts.get('analyze_decision_state')}"
        )
        print(f"  hop_limit_hits={hop_limit_hit_count()} cache={metrics}")

        env._decision_analysis_cache = {}
        env._ev_analysis_cache = {}
        env._global_serviceability_cache = None
        env.last_decision_analysis = None
        ev = env.select_acting_ev()
        ref = analyze_decision_state(env, ev, implementation="reference")
        env._decision_analysis_cache = {}
        env._global_serviceability_cache = None
        env._ev_analysis_cache = {}
        env.last_decision_analysis = None
        opt = analyze_decision_state(env, ev, implementation="optimized")
        same_mask = np.array_equal(ref.final_node_mask, opt.final_node_mask)
        same_svc = set(ref.global_serviceability.serviceable_customer_ids) == set(
            opt.global_serviceability.serviceable_customer_ids
        )
        print(f"  parity: final_mask={same_mask} serviceability={same_svc}")
    return overall_ref, overall_opt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=15)
    args = ap.parse_args()

    print("=" * 72)
    print("Decision analysis benchmark (reference vs optimized EV-state-first)")
    print("=" * 72)

    # Start-of-episode (often direct-feasible — both paths stay light).
    ref0, opt0 = _run_suite("start_state", stress_recharge=False, repeats=args.repeats)
    # Recharge-stressed state (customer-first reference pays per-customer chain search).
    ref1, opt1 = _run_suite("low_battery_recharge_stress", stress_recharge=True, repeats=max(3, args.repeats // 2))

    for title, overall_ref, overall_opt in (
        ("START STATE", ref0, opt0),
        ("RECHARGE-STRESS", ref1, opt1),
    ):
        if not overall_ref or not overall_opt:
            continue
        rs = _stats(overall_ref)
        os_ = _stats(overall_opt)
        speedup = rs["median"] / max(os_["median"], 1e-9)
        print("\n" + "=" * 72)
        print(f"{title} reference median: {rs['median']:.2f} ms")
        print(f"{title} optimized median: {os_['median']:.2f} ms")
        print(f"{title} speedup: {speedup:.2f}x  (target >= 3.0x on recharge-stress)")
        if title.startswith("RECHARGE") and speedup < 3.0:
            print("NOTE: <3x speedup — remaining bottleneck is typically:")
            print("  - station_feasible / charge-portion evaluation in hard masks")
            print("  - safe-continuation for basic survivors needing station mediation")
            print("  - policy/critic torch forwards (outside decision analysis)")
        print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
