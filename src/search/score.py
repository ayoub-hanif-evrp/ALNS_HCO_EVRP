"""Official lexicographic benchmark score for EVRPTW comparisons.

Order (all minimized):
  1. hard violations
  2. unserved customers
  3. used vehicles
  4. total distance
  5. total charging time
  6. total route duration
  7. internal paper J
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class BenchmarkScore:
    hard_violations: int
    unserved_customers: int
    used_vehicles: int
    total_distance: float
    total_charging_time: float
    total_duration: float
    paper_j: float

    def as_tuple(self) -> tuple:
        return (
            int(self.hard_violations),
            int(self.unserved_customers),
            int(self.used_vehicles),
            float(self.total_distance),
            float(self.total_charging_time),
            float(self.total_duration),
            float(self.paper_j),
        )

    def dominates(self, other: "BenchmarkScore") -> bool:
        return self.as_tuple() < other.as_tuple()

    @classmethod
    def worst(cls) -> "BenchmarkScore":
        return cls(
            hard_violations=10**9,
            unserved_customers=10**9,
            used_vehicles=10**9,
            total_distance=1e18,
            total_charging_time=1e18,
            total_duration=1e18,
            paper_j=1e18,
        )

    @classmethod
    def from_env(cls, env, weights) -> "BenchmarkScore":
        objective, _breakdown, summary = env.score(weights)
        hard = (
            int(getattr(env, "battery_violation_count", 0) or 0)
            + int(getattr(env, "capacity_violation_count", 0) or 0)
            + int(getattr(env, "time_window_violation_count", 0) or 0)
            + int(getattr(env, "depot_return_violation", 0) or 0)
            + int(bool(getattr(env, "infeasible", False)))
        )
        return cls(
            hard_violations=hard,
            unserved_customers=int(summary.unserved_customers),
            used_vehicles=int(summary.used_evs),
            total_distance=float(summary.total_distance),
            total_charging_time=float(summary.total_charging_time),
            total_duration=float(summary.total_duration),
            paper_j=float(objective),
        )

    @classmethod
    def from_replay(cls, result: Any) -> "BenchmarkScore":
        hard = int(getattr(result, "hard_violation_count", 0) or 0)
        if not bool(getattr(result, "feasible", False)):
            hard = max(hard, 1)
        return cls(
            hard_violations=hard,
            unserved_customers=int(getattr(result, "unserved_customers", 0) or 0),
            used_vehicles=int(getattr(result, "used_evs", 0) or 0),
            total_distance=float(getattr(result, "distance", 0.0) or 0.0),
            total_charging_time=float(getattr(result, "charging_time", 0.0) or 0.0),
            total_duration=float(getattr(result, "duration", 0.0) or 0.0),
            paper_j=float(getattr(result, "objective", 0.0) or 0.0),
        )

    def to_report_row(self, *, feasible: bool, n_customers: int, runtime: float,
                      charging_visits: int = 0) -> dict:
        served = max(0, int(n_customers) - int(self.unserved_customers))
        return {
            "feasible": bool(feasible) and self.hard_violations == 0 and self.unserved_customers == 0,
            "served_customers": served,
            "total_customers": int(n_customers),
            "served_ratio": served / max(int(n_customers), 1),
            "used_vehicles": int(self.used_vehicles),
            "total_distance": float(self.total_distance),
            "charging_visits": int(charging_visits),
            "charging_time": float(self.total_charging_time),
            "duration": float(self.total_duration),
            "runtime": float(runtime),
            "paper_j": float(self.paper_j),
            "hard_violations": int(self.hard_violations),
            "unserved_customers": int(self.unserved_customers),
        }


def compare_benchmark(a: BenchmarkScore, b: BenchmarkScore) -> int:
    ta, tb = a.as_tuple(), b.as_tuple()
    if ta < tb:
        return -1
    if ta > tb:
        return 1
    return 0


# Back-compat alias: SearchScore mirrors BenchmarkScore field order.
@dataclass(frozen=True, order=True)
class SearchScore:
    """Legacy alias aligned to BenchmarkScore lexicographic order."""

    hard_violation_count: int
    unserved_customer_count: int
    used_evs: int
    total_distance: float
    total_charging_time: float
    total_duration: float
    paper_objective: float

    def as_tuple(self) -> tuple:
        return (
            int(self.hard_violation_count),
            int(self.unserved_customer_count),
            int(self.used_evs),
            float(self.total_distance),
            float(self.total_charging_time),
            float(self.total_duration),
            float(self.paper_objective),
        )

    def dominates(self, other: "SearchScore") -> bool:
        return self.as_tuple() < other.as_tuple()

    def to_benchmark(self) -> BenchmarkScore:
        return BenchmarkScore(
            hard_violations=self.hard_violation_count,
            unserved_customers=self.unserved_customer_count,
            used_vehicles=self.used_evs,
            total_distance=self.total_distance,
            total_charging_time=self.total_charging_time,
            total_duration=self.total_duration,
            paper_j=self.paper_objective,
        )

    @classmethod
    def from_benchmark(cls, b: BenchmarkScore) -> "SearchScore":
        return cls(
            hard_violation_count=b.hard_violations,
            unserved_customer_count=b.unserved_customers,
            used_evs=b.used_vehicles,
            total_distance=b.total_distance,
            total_charging_time=b.total_charging_time,
            total_duration=b.total_duration,
            paper_objective=b.paper_j,
        )

    @classmethod
    def worst(cls) -> "SearchScore":
        return cls.from_benchmark(BenchmarkScore.worst())

    @classmethod
    def from_env(cls, env, weights) -> "SearchScore":
        return cls.from_benchmark(BenchmarkScore.from_env(env, weights))


def compare_scores(a: SearchScore, b: SearchScore) -> int:
    return compare_benchmark(a.to_benchmark(), b.to_benchmark())


def score_from_replay(result: Any, weights=None) -> SearchScore:
    return SearchScore.from_benchmark(BenchmarkScore.from_replay(result))
