"""Official lexicographic benchmark score for EVRPTW comparisons.

Classical mode:
  hard, unserved, K, distance, duration

Heterogeneous modes:
  hard, unserved, K, Φ, distance, duration
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.hanif.evaluation.paper_objective import MODE_CLASSICAL, ranking_tuple


@dataclass(frozen=True)
class BenchmarkScore:
    hard_violations: int
    unserved_customers: int
    used_vehicles: int
    total_distance: float
    total_charging_time: float
    total_duration: float
    paper_j: float
    phi: float = 0.0
    total_station_time: float = 0.0
    objective_mode: str = MODE_CLASSICAL

    def as_tuple(self) -> tuple:
        return ranking_tuple(
            hard=self.hard_violations,
            unserved=self.unserved_customers,
            used_k=self.used_vehicles,
            phi=self.phi,
            distance=self.total_distance,
            station_time=self.total_station_time,
            duration=self.total_duration,
            charging_time=self.total_charging_time,
            paper_j=self.paper_j,
            mode=self.objective_mode,
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
            phi=1e18,
            total_station_time=1e18,
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
            total_station_time=float(summary.total_charging_time)
            + float(getattr(summary, "station_waiting_time", 0.0) or 0.0),
        )

    @classmethod
    def from_replay(cls, result: Any) -> "BenchmarkScore":
        hard = int(getattr(result, "hard_violation_count", 0) or 0)
        if not bool(getattr(result, "feasible", False)):
            hard = max(hard, 1)
        score = getattr(result, "score", None)
        return cls(
            hard_violations=hard,
            unserved_customers=int(getattr(result, "unserved_customers", 0) or 0),
            used_vehicles=int(getattr(result, "used_evs", 0) or 0),
            total_distance=float(getattr(result, "distance", 0.0) or 0.0),
            total_charging_time=float(getattr(result, "charging_time", 0.0) or 0.0),
            total_duration=float(getattr(result, "duration", 0.0) or 0.0),
            paper_j=float(getattr(result, "objective", 0.0) or 0.0),
            phi=float(getattr(score, "phi", 0.0) or 0.0),
            total_station_time=float(getattr(score, "total_station_time", 0.0) or 0.0),
            objective_mode=str(getattr(score, "objective_mode", MODE_CLASSICAL) or MODE_CLASSICAL),
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
            "phi": float(self.phi),
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


@dataclass(frozen=True)
class SearchScore:
    """Search ranking score. Comparison uses as_tuple(), not dataclass field order."""

    hard_violation_count: int
    unserved_customer_count: int
    used_evs: int
    total_distance: float
    total_charging_time: float
    total_duration: float
    paper_objective: float
    phi: float = 0.0
    total_station_time: float = 0.0
    objective_mode: str = MODE_CLASSICAL

    def as_tuple(self) -> tuple:
        return ranking_tuple(
            hard=self.hard_violation_count,
            unserved=self.unserved_customer_count,
            used_k=self.used_evs,
            phi=self.phi,
            distance=self.total_distance,
            station_time=self.total_station_time,
            duration=self.total_duration,
            charging_time=self.total_charging_time,
            paper_j=self.paper_objective,
            mode=self.objective_mode,
        )

    def dominates(self, other: "SearchScore") -> bool:
        return self.as_tuple() < other.as_tuple()

    def operational_scalar(self) -> float:
        """Scalar used by regret / SA; never overrides the fleet-first lex rank."""
        if str(self.objective_mode) == MODE_CLASSICAL:
            return float(self.total_distance)
        return float(self.phi)

    def to_benchmark(self) -> BenchmarkScore:
        return BenchmarkScore(
            hard_violations=self.hard_violation_count,
            unserved_customers=self.unserved_customer_count,
            used_vehicles=self.used_evs,
            total_distance=self.total_distance,
            total_charging_time=self.total_charging_time,
            total_duration=self.total_duration,
            paper_j=self.paper_objective,
            phi=self.phi,
            total_station_time=self.total_station_time,
            objective_mode=self.objective_mode,
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
            phi=b.phi,
            total_station_time=b.total_station_time,
            objective_mode=b.objective_mode,
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
    if getattr(result, "score", None) is not None:
        return result.score
    return SearchScore.from_benchmark(BenchmarkScore.from_replay(result))
