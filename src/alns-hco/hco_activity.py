"""HCO activity / cache diagnostics with explicit input→output plan diffs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


CHARGE_QTY_TOL = 1e-9


@dataclass
class HcoActivity:
    """Counters for one (instance, method, seed) solve involving HCO."""

    hco_calls: int = 0
    feasible_hco_calls: int = 0
    changed_calls: int = 0
    unchanged_calls: int = 0
    station_insertions: int = 0
    station_removals: int = 0
    station_replacements: int = 0
    charge_quantity_changes: int = 0
    objective_improvements: int = 0
    objective_ties: int = 0
    rejected_worse_plans: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    nonempty_candidate_sets: int = 0
    candidates_evaluated: int = 0
    candidates_feasible: int = 0
    eval_time_s: float = 0.0

    # Backward-compatible aliases used by older CSV flatteners
    @property
    def calls(self) -> int:
        return self.hco_calls

    @property
    def feasible_plans(self) -> int:
        return self.feasible_hco_calls

    @property
    def plans_changed(self) -> int:
        return self.changed_calls

    def record_cache_hit(self) -> None:
        self.hco_calls += 1
        self.cache_hits += 1

    def record_cache_miss(self) -> None:
        self.hco_calls += 1
        self.cache_misses += 1

    @property
    def cache_hit_rate(self) -> float | None:
        if self.hco_calls <= 0:
            return None
        return self.cache_hits / self.hco_calls

    @property
    def activation_rate(self) -> float | None:
        """changed_calls / feasible_hco_calls."""
        if self.feasible_hco_calls <= 0:
            return None
        return self.changed_calls / self.feasible_hco_calls

    @property
    def improvement_rate(self) -> float | None:
        """objective_improvements / feasible_hco_calls."""
        if self.feasible_hco_calls <= 0:
            return None
        return self.objective_improvements / self.feasible_hco_calls

    def as_dict(self) -> dict:
        d = asdict(self)
        d["calls"] = self.calls
        d["feasible_plans"] = self.feasible_plans
        d["plans_changed"] = self.plans_changed
        d["cache_hit_rate"] = self.cache_hit_rate
        d["activation_rate"] = self.activation_rate
        d["improvement_rate"] = self.improvement_rate
        return d


@dataclass
class OracleCache:
    """Cache keyed by (customers, mode, k, non_worsening flag context is outside)."""

    plans: dict = field(default_factory=dict)
    hits: int = 0
    calls: int = 0
    activity: HcoActivity = field(default_factory=HcoActivity)
