"""Fleet-size search: lexicographic min-fleet then routing cost."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

from src.hanif.env.cost import SolutionSummary


@dataclass
class FleetCandidate:
    m: int
    feasible: bool
    objective: float
    summary: SolutionSummary
    result: object = None  # underlying episode/baseline result


def lexicographic_fleet_range(
    instance,
    fleet_cfg: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Schneider-style search band for min-fleet optimization.

    m_lo = max(capacity_m_min, ceil(n / customers_per_vehicle_lo))
    m_hi = max(m_lo, min(n, ceil(n / customers_per_vehicle_hi)))

    Defaults: lo=12 cust/EV → ~9 vans on n=100; hi=5 cust/EV → ~20 vans.
    """
    cfg = fleet_cfg or {}
    n = int(getattr(instance, "num_customers", 1) or 1)
    m_cap = int(getattr(instance, "m_min", 1) or 1)
    # lo divisor → lower fleet bound; hi divisor (smaller) → upper fleet bound
    cpv_lo = max(1.0, float(cfg.get("customers_per_vehicle_lo", 12.0) or 12.0))
    cpv_hi = max(1.0, float(cfg.get("customers_per_vehicle_hi", 5.0) or 5.0))
    extra = int(cfg.get("extra_margin", 12) or 12)
    # Capacity floor is hard; TW often needs *more* than capacity m_min.
    m_lo = max(m_cap, int(math.ceil(n / cpv_lo)))
    m_hi_tw = int(math.ceil(n / cpv_hi))
    m_hi = max(m_lo, min(n, max(m_hi_tw, m_cap + max(0, extra))))
    m_lo = min(n, max(1, m_lo))
    m_hi = min(n, max(m_lo, m_hi))
    return m_lo, m_hi


def fleet_size_range(
    instance,
    extra_margin: int,
    fleet_cfg: dict[str, Any] | None = None,
) -> tuple[int, int]:
    """Compute [m_min, m_max].

    When ``fleet_cfg.lexicographic`` is true, use the Schneider min-fleet band.
    Otherwise keep the legacy TW-floor + extra_margin window.
    """
    cfg = fleet_cfg or {}
    if bool(cfg.get("lexicographic", False)):
        return lexicographic_fleet_range(instance, cfg)

    m_cap = instance.m_min
    if bool(cfg.get("tw_aware_floor", False)):
        n = int(getattr(instance, "num_customers", 0) or 0)
        tcpv = max(1.0, float(cfg.get("target_customers_per_vehicle", 8.0) or 8.0))
        tw_floor = max(1, int(math.ceil(n / tcpv))) if n > 0 else 1
        m_min = max(m_cap, tw_floor)
    else:
        m_min = m_cap
    m_max = min(instance.num_customers, m_min + int(extra_margin))
    m_max = max(m_max, m_min)
    return m_min, m_max


def _hierarchical_key(c: FleetCandidate):
    s = c.summary
    return (s.used_evs, s.total_distance, s.total_duration, s.total_charging_time)


def select_best(candidates: list[FleetCandidate], selection_mode: str = "hierarchical") -> FleetCandidate:
    """Select the best candidate.

    hierarchical: feasible first, then min used EVs, distance, duration, charging.
    scalar:       min objective among feasible; else least-cost overall.
    If no feasible candidate exists, return the least-infeasible (min objective).
    """
    feasible = [c for c in candidates if c.feasible]
    if not feasible:
        return min(candidates, key=lambda c: c.objective)

    if selection_mode == "scalar":
        return min(feasible, key=lambda c: c.objective)
    return min(feasible, key=_hierarchical_key)


def lexicographic_min_fleet_search(
    instance,
    runner: Callable[[object, int], FleetCandidate],
    *,
    fleet_cfg: dict[str, Any] | None = None,
    selection_mode: str = "hierarchical",
    extra_margin: int = 12,
) -> tuple[FleetCandidate, list[FleetCandidate]]:
    """Phase-1 min feasible fleet via high probe + binary search.

    1. Evaluate ``m_hi`` first (most likely feasible under hard TW).
    2. If feasible, binary-search the smallest feasible ``m`` in ``[m_lo, m_hi]``.
    3. If infeasible, probe ``mid`` and ``m_lo`` only (avoid O(range) expert cost)
       and return the hierarchical best among the few candidates.
    """
    cfg = fleet_cfg or {}
    m_min, m_max = fleet_size_range(instance, extra_margin, fleet_cfg=cfg)
    candidates: list[FleetCandidate] = []
    by_m: dict[int, FleetCandidate] = {}

    def _run(m: int) -> FleetCandidate:
        if m in by_m:
            return by_m[m]
        cand = runner(instance, m)
        by_m[m] = cand
        candidates.append(cand)
        return cand

    hi = _run(m_max)
    # Fast path for expert demos: only probe the upper fleet bound.
    if int(cfg.get("expert_probes", 0) or 0) == 1:
        return hi, candidates
    if not hi.feasible:
        mid = (m_min + m_max) // 2
        if mid not in by_m and mid != m_max:
            _run(mid)
        if m_min not in by_m:
            _run(m_min)
        return select_best(candidates, selection_mode), candidates

    # Binary search smallest feasible m in [m_min, m_max].
    lo, bound = m_min, m_max
    while lo < bound:
        mid = (lo + bound) // 2
        cand = _run(mid)
        if cand.feasible:
            bound = mid
        else:
            lo = mid + 1
    feasible = [c for c in candidates if c.feasible]
    best = min(feasible, key=lambda c: c.m) if feasible else select_best(candidates, selection_mode)
    return best, candidates


def fleet_search(
    instance,
    runner: Callable[[object, int], FleetCandidate],
    extra_margin: int = 5,
    selection_mode: str = "hierarchical",
    fleet_cfg: dict[str, Any] | None = None,
) -> tuple[FleetCandidate, list[FleetCandidate]]:
    """Run fleet optimization for ``instance``.

    When ``fleet_cfg.lexicographic`` is true, use min-feasible ascending search.
    Otherwise evaluate every m in [m_min, m_max] and pick the best.
    """
    cfg = fleet_cfg or {}
    if bool(cfg.get("lexicographic", False)):
        return lexicographic_min_fleet_search(
            instance,
            runner,
            fleet_cfg=cfg,
            selection_mode=selection_mode,
            extra_margin=extra_margin,
        )
    m_min, m_max = fleet_size_range(instance, extra_margin, fleet_cfg=cfg)
    candidates: list[FleetCandidate] = []
    for m in range(m_min, m_max + 1):
        candidates.append(runner(instance, m))
    best = select_best(candidates, selection_mode)
    return best, candidates
