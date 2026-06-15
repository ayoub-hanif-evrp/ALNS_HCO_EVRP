"""Fleet-size search: run the solver for several candidate fleet sizes m."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.env.cost import SolutionSummary


@dataclass
class FleetCandidate:
    m: int
    feasible: bool
    objective: float
    summary: SolutionSummary
    result: object = None  # underlying episode/baseline result


def fleet_size_range(instance, extra_margin: int) -> tuple[int, int]:
    """Compute [m_min, m_max].

    m_min = ceil(total_demand / load_capacity)
    m_max = min(num_customers, m_min + extra_margin)
    """
    m_min = instance.m_min
    m_max = min(instance.num_customers, m_min + extra_margin)
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


def fleet_search(
    instance,
    runner: Callable[[object, int], FleetCandidate],
    extra_margin: int = 5,
    selection_mode: str = "hierarchical",
) -> tuple[FleetCandidate, list[FleetCandidate]]:
    """Run ``runner`` for each m in [m_min, m_max] and pick the best.

    Args:
        instance: the EVRPTW instance.
        runner: callable(instance, m) -> FleetCandidate.
        extra_margin: how far above m_min to search.
        selection_mode: 'hierarchical' or 'scalar'.

    Returns:
        (best_candidate, all_candidates) where all_candidates is ordered by m.
    """
    m_min, m_max = fleet_size_range(instance, extra_margin)
    candidates: list[FleetCandidate] = []
    for m in range(m_min, m_max + 1):
        candidates.append(runner(instance, m))
    best = select_best(candidates, selection_mode)
    return best, candidates
