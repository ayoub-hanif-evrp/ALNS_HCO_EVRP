"""Flat training schedules and TW-aware / lexicographic fleet sizing."""
from __future__ import annotations

import math
import random
from typing import Any


def tw_aware_fleet_floor(
    instance,
    *,
    target_customers_per_vehicle: float = 8.0,
) -> int:
    """Minimum fleet for hard TW learnability (capacity floor alone is insufficient).

    Capacity-based ``m_min = ceil(demand/Q)`` can be 1–2 on Cus_100 while hard
    time windows require many short routes (classic VRPTW / Schneider EVRPTW).
    Using ``ceil(n_customers / target_customers_per_vehicle)`` is a standard
    constructive heuristic floor used in OR warm-starts and NCO curricula.
    """
    n = int(getattr(instance, "num_customers", 0) or 0)
    tcpv = max(1.0, float(target_customers_per_vehicle))
    return max(1, int(math.ceil(n / tcpv))) if n > 0 else 1


def resolve_fleet_size(
    instance,
    fleet_cfg: dict[str, Any] | None,
    rng: random.Random | None = None,
    *,
    mstar_map: dict[str, int] | None = None,
) -> int:
    """Resolve training/eval fleet size from config.

    Modes
    -----
    - ``expert_mstar``: use cached lexicographic expert ``m*`` (optional ±1 jitter).
    - ``fixed_tw_aware``: ``m = max(capacity_m_min, tw_floor)``.
    - ``sample``: uniform in ``[m_lo, m_lo + extra_margin]`` (or lex band if set).
    """
    cfg = fleet_cfg or {}
    rng = rng or random.Random(0)
    n = int(getattr(instance, "num_customers", 1) or 1)
    m_cap = int(getattr(instance, "m_min", 1) or 1)
    extra = int(cfg.get("extra_margin", 4) or 0)
    mode = str(cfg.get("training_fleet_mode", cfg.get("mode", "fixed_tw_aware"))).lower()
    tw_on = bool(cfg.get("tw_aware_floor", True))
    tcpv = float(cfg.get("target_customers_per_vehicle", 8.0) or 8.0)
    tw_floor = tw_aware_fleet_floor(instance, target_customers_per_vehicle=tcpv) if tw_on else 1
    m_lo = max(m_cap, tw_floor)
    m_lo = min(n, max(1, m_lo))

    if mode in {"expert_mstar", "mstar", "cached_mstar"}:
        name = str(getattr(instance, "instance_name", "") or "")
        cached = None
        if mstar_map and name in mstar_map:
            cached = int(mstar_map[name])
        elif name:
            # Lazy load once per process if map not passed
            from src.evaluation.mstar_cache import load_mstar_map

            # Allow callers to stash a shared map on the cfg object-like dict
            shared = cfg.get("_mstar_map_cache")
            if shared is None:
                # Build a throwaway config view for path resolution
                shared = load_mstar_map({"fleet_search": cfg, "demonstrations": {
                    "mstar_cache": cfg.get("mstar_cache"),
                }})
                cfg["_mstar_map_cache"] = shared
            if name in shared:
                cached = int(shared[name])
        if cached is not None:
            m = max(1, min(n, cached))
            jitter_p = float(cfg.get("mstar_jitter_prob", 0.0) or 0.0)
            if jitter_p > 0.0 and rng.random() < jitter_p:
                m = max(1, min(n, m + rng.choice([-1, 1])))
            return m
        # Fallback if cache miss: TW-aware floor (safe learnable fleet)
        return m_lo

    if mode in {"fixed", "fixed_tw_aware", "deterministic"}:
        return m_lo

    if bool(cfg.get("lexicographic", False)):
        from src.evaluation.fleet_search import lexicographic_fleet_range

        lo, hi = lexicographic_fleet_range(instance, cfg)
        return int(rng.randint(lo, max(lo, hi)))

    m_hi = min(n, m_lo + max(0, extra))
    return int(rng.randint(m_lo, max(m_lo, m_hi)))


def sample_fleet_size(
    instance,
    extra_margin: int,
    rng: random.Random,
    *,
    fleet_cfg: dict[str, Any] | None = None,
    mstar_map: dict[str, int] | None = None,
) -> int:
    """Backward-compatible wrapper; prefer ``resolve_fleet_size`` with full config."""
    cfg = dict(fleet_cfg or {})
    cfg.setdefault("extra_margin", int(extra_margin))
    # Legacy callers assumed pure capacity sampling unless cfg opts into TW floor.
    if fleet_cfg is None:
        cfg.setdefault("tw_aware_floor", False)
        cfg.setdefault("training_fleet_mode", "sample")
    return resolve_fleet_size(instance, cfg, rng, mstar_map=mstar_map)


def build_sequential_schedule(
    num_instances: int,
    passes: int,
    *,
    shuffle_within_pass: bool = False,
    rng: random.Random | None = None,
) -> list[int]:
    """Repeat all instance indices ``passes`` times (one index per training step)."""
    if num_instances < 1:
        raise ValueError("build_sequential_schedule requires at least one training instance")
    if passes < 1:
        raise ValueError("passes must be >= 1")

    rng = rng or random.Random(0)
    base = list(range(num_instances))
    schedule: list[int] = []
    for _ in range(int(passes)):
        order = list(base)
        if shuffle_within_pass:
            rng.shuffle(order)
        schedule.extend(order)
    return schedule
