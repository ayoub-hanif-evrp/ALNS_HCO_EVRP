"""Cost / objective function for E-VRPTW solutions.

J(pi) = lambda_v * V + lambda_d * D + lambda_t * T + lambda_c * C_charge
        + lambda_p * P_charge + lambda_w * W + P(pi)

where P_charge is the monetary charging cost (station price × energy)
and P is the sum of weighted infeasibility penalties. The RL reward is
R(pi) = -J(pi).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CostWeights:
    lambda_vehicle: float = 1000.0
    lambda_distance: float = 1.0
    lambda_duration: float = 0.1
    lambda_charging: float = 0.5
    lambda_charging_price: float = 1.0
    lambda_waiting: float = 0.1
    penalty_unserved: float = 10000.0
    penalty_duplicate: float = 10000.0
    penalty_late: float = 5000.0
    penalty_battery: float = 10000.0
    penalty_capacity: float = 10000.0
    penalty_depot: float = 10000.0
    penalty_infeasible: float = 100000.0
    penalty_reserved: float = 10000.0

    @classmethod
    def from_config(cls, cost_cfg: dict) -> "CostWeights":
        return cls(**{k: float(v) for k, v in cost_cfg.items() if k in cls.__annotations__})


@dataclass
class SolutionSummary:
    """Aggregate quantities required to score a solution."""

    used_evs: int = 0
    total_distance: float = 0.0
    total_duration: float = 0.0
    total_charging_time: float = 0.0
    total_charging_price: float = 0.0
    total_waiting_time: float = 0.0
    unserved_customers: int = 0
    duplicate_customer_visits: int = 0
    total_lateness: float = 0.0
    total_battery_violation: float = 0.0
    total_capacity_violation: float = 0.0
    depot_return_violation: int = 0
    unresolved_reserved_customers: int = 0
    infeasible_flag: int = 0


def compute_cost(summary: SolutionSummary, weights: CostWeights) -> tuple[float, dict[str, float]]:
    """Return (J, breakdown) for a solution summary."""
    objective_terms = {
        "vehicle": weights.lambda_vehicle * summary.used_evs,
        "distance": weights.lambda_distance * summary.total_distance,
        "duration": weights.lambda_duration * summary.total_duration,
        "charging": weights.lambda_charging * summary.total_charging_time,
        "charging_price": weights.lambda_charging_price * summary.total_charging_price,
        "waiting": weights.lambda_waiting * summary.total_waiting_time,
    }
    penalty_terms = {
        "unserved": weights.penalty_unserved * summary.unserved_customers,
        "duplicate": weights.penalty_duplicate * summary.duplicate_customer_visits,
        "late": weights.penalty_late * summary.total_lateness,
        "battery": weights.penalty_battery * summary.total_battery_violation,
        "capacity": weights.penalty_capacity * summary.total_capacity_violation,
        "depot": weights.penalty_depot * summary.depot_return_violation,
        "infeasible": weights.penalty_infeasible * summary.infeasible_flag,
        "reserved": weights.penalty_reserved * summary.unresolved_reserved_customers,
    }
    j = sum(objective_terms.values()) + sum(penalty_terms.values())
    breakdown = {**{f"obj_{k}": v for k, v in objective_terms.items()},
                 **{f"pen_{k}": v for k, v in penalty_terms.items()},
                 "J": j}
    return float(j), breakdown


def training_reward(objective: float, scale: float = 1.0) -> float:
    """Scaled training reward: R = -J / reward_scale (stabilizes critic targets).

    The evaluation objective ``J`` can reach very large magnitudes because of
    penalty weights (10k–100k). Dividing by ``scale`` keeps advantages and
    policy gradients in a numerically stable range without changing the
    *direction* of the learning signal.

    R_train = -J / scale
    """
    scale = max(float(scale), 1.0)
    return -float(objective) / scale


def is_feasible(summary: SolutionSummary) -> bool:
    """A solution is feasible when no penalty component is triggered."""
    return (
        summary.unserved_customers == 0
        and summary.duplicate_customer_visits == 0
        and summary.total_lateness <= 1e-6
        and summary.total_battery_violation <= 1e-6
        and summary.total_capacity_violation <= 1e-6
        and summary.depot_return_violation == 0
        and summary.unresolved_reserved_customers == 0
        and summary.infeasible_flag == 0
    )
