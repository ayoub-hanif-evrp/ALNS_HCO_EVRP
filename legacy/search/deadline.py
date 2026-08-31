"""Shared wall-clock deadline for budget-controlled search."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class SearchDeadline:
    """Cooperative deadline checked inside expensive loops."""

    deadline_monotonic: float
    started_monotonic: float
    budget_seconds: float
    stage: str = "global"
    cleanup_allowance_seconds: float = 5.0

    @classmethod
    def from_seconds(
        cls,
        budget_seconds: float | None,
        *,
        stage: str = "global",
        cleanup_allowance_seconds: float = 5.0,
    ) -> "SearchDeadline":
        now = time.perf_counter()
        if budget_seconds is None or budget_seconds <= 0:
            return cls(
                deadline_monotonic=float("inf"),
                started_monotonic=now,
                budget_seconds=float("inf"),
                stage=stage,
                cleanup_allowance_seconds=cleanup_allowance_seconds,
            )
        b = float(budget_seconds)
        return cls(
            deadline_monotonic=now + b,
            started_monotonic=now,
            budget_seconds=b,
            stage=stage,
            cleanup_allowance_seconds=cleanup_allowance_seconds,
        )

    def remaining(self) -> float:
        if self.deadline_monotonic == float("inf"):
            return float("inf")
        return max(0.0, self.deadline_monotonic - time.perf_counter())

    def expired(self) -> bool:
        return time.perf_counter() >= self.deadline_monotonic

    def elapsed(self) -> float:
        return time.perf_counter() - self.started_monotonic

    def check(self) -> bool:
        """Return True if work may continue."""
        return not self.expired()

    def child(self, seconds: float | None, *, stage: str | None = None) -> "SearchDeadline":
        """Cap a child stage to min(remaining, seconds) without exceeding parent."""
        now = time.perf_counter()
        parent_rem = self.remaining()
        if seconds is None:
            child_budget = parent_rem
        else:
            child_budget = min(float(seconds), parent_rem)
        return SearchDeadline(
            deadline_monotonic=now + max(0.0, child_budget),
            started_monotonic=now,
            budget_seconds=max(0.0, child_budget),
            stage=stage or self.stage,
            cleanup_allowance_seconds=self.cleanup_allowance_seconds,
        )

    def within_tolerance(self, runtime: float, *, frac: float = 0.05) -> bool:
        if self.budget_seconds == float("inf"):
            return True
        return float(runtime) <= float(self.budget_seconds) * (1.0 + frac) + self.cleanup_allowance_seconds


@dataclass
class StageBudgetPlan:
    """Explicit fractional stage budgets; unused time flows forward."""

    upper_bound_fraction: float = 0.15
    fleet_descent_fraction: float = 0.40
    route_pool_fraction: float = 0.10
    charging_fraction: float = 0.10
    repair_alns_fraction: float = 0.20
    rescue_fraction: float = 0.05

    def __post_init__(self) -> None:
        total = (
            self.upper_bound_fraction
            + self.fleet_descent_fraction
            + self.route_pool_fraction
            + self.charging_fraction
            + self.repair_alns_fraction
            + self.rescue_fraction
        )
        if abs(total - 1.0) > 1e-6:
            # Normalize rather than fail — configs may be approximate.
            self.upper_bound_fraction /= total
            self.fleet_descent_fraction /= total
            self.route_pool_fraction /= total
            self.charging_fraction /= total
            self.repair_alns_fraction /= total
            self.rescue_fraction /= total

    @classmethod
    def from_config(cls, config: dict | None) -> "StageBudgetPlan":
        raw = (config or {}).get("deterministic_fps_budget") or {}
        return cls(
            upper_bound_fraction=float(raw.get("upper_bound_fraction", 0.15)),
            fleet_descent_fraction=float(raw.get("fleet_descent_fraction", 0.40)),
            route_pool_fraction=float(raw.get("route_pool_fraction", 0.10)),
            charging_fraction=float(raw.get("charging_fraction", 0.10)),
            repair_alns_fraction=float(raw.get("repair_alns_fraction", 0.20)),
            rescue_fraction=float(raw.get("rescue_fraction", 0.05)),
        )

    def seconds_for(self, total_budget: float | None, name: str) -> float | None:
        if total_budget is None:
            return None
        frac = {
            "upper_bound": self.upper_bound_fraction,
            "fleet_descent": self.fleet_descent_fraction,
            "route_pool": self.route_pool_fraction,
            "charging": self.charging_fraction,
            "repair_alns": self.repair_alns_fraction,
            "rescue": self.rescue_fraction,
        }.get(name, 0.0)
        return float(total_budget) * frac


@dataclass
class StageRuntimeLedger:
    """Record per-stage wall times and whether each mandatory stage ran."""

    times: dict[str, float] = field(default_factory=dict)
    invoked: dict[str, bool] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)

    def mark(self, stage: str, seconds: float, *, invoked: bool = True, note: str = "") -> None:
        self.times[stage] = float(seconds)
        self.invoked[stage] = bool(invoked)
        if note:
            self.notes[stage] = note

    def as_dict(self) -> dict:
        return {"times": dict(self.times), "invoked": dict(self.invoked), "notes": dict(self.notes)}
