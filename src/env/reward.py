"""Step-wise reward decomposition aligned with ``cost.compute_cost``."""
from __future__ import annotations

from dataclasses import dataclass, field

from .cost import CostWeights, SolutionSummary, compute_cost, training_reward


@dataclass
class CostSnapshot:
    """Cumulative environment quantities used for objective decomposition."""

    total_distance: float = 0.0
    total_waiting_time: float = 0.0
    total_charging_time: float = 0.0
    total_charging_price: float = 0.0
    total_lateness: float = 0.0
    total_battery_violation: float = 0.0
    total_capacity_violation: float = 0.0
    duplicate_customer_visits: int = 0
    depot_return_violation: int = 0
    used_evs: int = 0
    total_duration: float = 0.0
    infeasible_flag: int = 0
    num_served_customers: int = 0
    unresolved_reserved: int = 0


@dataclass
class StepInfo:
    ev_id: int
    selected_node: int
    node_type: str
    previous_node: int
    delta_distance: float = 0.0
    delta_travel_time: float = 0.0
    delta_duration: float = 0.0
    delta_waiting_time: float = 0.0
    delta_charging_time: float = 0.0
    delta_charging_price: float = 0.0
    delta_energy_used: float = 0.0
    charged_energy: float = 0.0
    customer_served: bool = False
    delta_battery_violation: float = 0.0
    delta_capacity_violation: float = 0.0
    delta_lateness: float = 0.0
    delta_duplicate_visits: int = 0
    delta_depot_return_violation: int = 0
    immediate_penalty: float = 0.0
    delta_objective: float = 0.0
    reward: float = 0.0
    raw_reward: float = 0.0
    shaping_reward: float = 0.0
    terminal: bool = False
    extra: dict = field(default_factory=dict)


def snapshot_env(env) -> CostSnapshot:
    inst = env.instance
    unresolved = (
        int(env.reserved_mask[inst.customer_indices].sum()) if inst.customer_indices else 0
    )
    return CostSnapshot(
        total_distance=float(env.total_distance),
        total_waiting_time=float(env.total_waiting_time),
        total_charging_time=float(env.total_charging_time),
        total_charging_price=float(env.total_charging_price),
        total_lateness=float(env.total_lateness),
        total_battery_violation=float(env.total_battery_violation),
        total_capacity_violation=float(env.total_capacity_violation),
        duplicate_customer_visits=int(env.duplicate_customer_visits),
        depot_return_violation=int(env.depot_return_violation),
        used_evs=int(env.used_evs()),
        total_duration=float(sum(ev.current_time for ev in env.evs if ev.used())),
        infeasible_flag=int(env.infeasible),
        num_served_customers=int(env.num_served_customers),
        unresolved_reserved=unresolved,
    )


def _delta_objective(before: CostSnapshot, after: CostSnapshot, weights: CostWeights) -> tuple[float, float, dict[str, float]]:
    """Return (delta_J, immediate_penalty_component, weighted breakdown)."""
    d_dist = after.total_distance - before.total_distance
    d_wait = after.total_waiting_time - before.total_waiting_time
    d_charge_t = after.total_charging_time - before.total_charging_time
    d_charge_p = after.total_charging_price - before.total_charging_price
    d_dur = after.total_duration - before.total_duration
    d_used = after.used_evs - before.used_evs

    d_batt = after.total_battery_violation - before.total_battery_violation
    d_cap = after.total_capacity_violation - before.total_capacity_violation
    d_late = after.total_lateness - before.total_lateness
    d_dup = after.duplicate_customer_visits - before.duplicate_customer_visits
    d_depot = after.depot_return_violation - before.depot_return_violation

    obj = (
        weights.lambda_vehicle * d_used
        + weights.lambda_distance * d_dist
        + weights.lambda_duration * d_dur
        + weights.lambda_charging * d_charge_t
        + weights.lambda_charging_price * d_charge_p
        + weights.lambda_waiting * d_wait
    )
    pen = (
        weights.penalty_duplicate * d_dup
        + weights.penalty_late * d_late
        + weights.penalty_battery * d_batt
        + weights.penalty_capacity * d_cap
        + weights.penalty_depot * d_depot
    )
    breakdown = {
        "delta_distance_cost": weights.lambda_distance * d_dist,
        "delta_duration_cost": weights.lambda_duration * d_dur,
        "delta_charging_time_cost": weights.lambda_charging * d_charge_t,
        "delta_charging_price_cost": weights.lambda_charging_price * d_charge_p,
        "delta_waiting_cost": weights.lambda_waiting * d_wait,
        "delta_fleet_cost": weights.lambda_vehicle * d_used,
        "delta_penalty_cost": float(pen),
    }
    return float(obj + pen), float(pen), breakdown


class StepRewardCalculator:
    """Compute per-step rewards from environment snapshots."""

    def __init__(self, weights: CostWeights, reward_scale: float = 1000.0):
        self.weights = weights
        self.reward_scale = max(float(reward_scale), 1.0)

    def step_reward(
        self,
        before: CostSnapshot,
        after: CostSnapshot,
        *,
        ev_id: int,
        selected_node: int,
        node_type: str,
        previous_node: int,
        delta_travel_time: float = 0.0,
        delta_energy_used: float = 0.0,
        charged_energy: float = 0.0,
        customer_served: bool = False,
    ) -> StepInfo:
        d_dist = after.total_distance - before.total_distance
        d_wait = after.total_waiting_time - before.total_waiting_time
        d_charge_t = after.total_charging_time - before.total_charging_time
        d_charge_p = after.total_charging_price - before.total_charging_price
        d_dur = after.total_duration - before.total_duration
        d_batt = after.total_battery_violation - before.total_battery_violation
        d_cap = after.total_capacity_violation - before.total_capacity_violation
        d_late = after.total_lateness - before.total_lateness
        d_dup = after.duplicate_customer_visits - before.duplicate_customer_visits
        d_depot = after.depot_return_violation - before.depot_return_violation

        delta_j, immediate_pen, breakdown = _delta_objective(before, after, self.weights)
        reward = -delta_j / self.reward_scale

        return StepInfo(
            ev_id=ev_id,
            selected_node=selected_node,
            node_type=node_type,
            previous_node=previous_node,
            delta_distance=d_dist,
            delta_travel_time=delta_travel_time,
            delta_duration=d_dur,
            delta_waiting_time=d_wait,
            delta_charging_time=d_charge_t,
            delta_charging_price=d_charge_p,
            delta_energy_used=delta_energy_used,
            charged_energy=charged_energy,
            customer_served=customer_served,
            delta_battery_violation=d_batt,
            delta_capacity_violation=d_cap,
            delta_lateness=d_late,
            delta_duplicate_visits=d_dup,
            delta_depot_return_violation=d_depot,
            immediate_penalty=immediate_pen,
            delta_objective=delta_j,
            reward=reward,
            raw_reward=reward,
            extra=breakdown,
        )

    def terminal_adjustment(self, before: CostSnapshot, summary: SolutionSummary) -> StepInfo:
        """Penalties / costs not captured incrementally during the rollout."""
        after = CostSnapshot(
            total_distance=summary.total_distance,
            total_waiting_time=summary.total_waiting_time,
            total_charging_time=summary.total_charging_time,
            total_charging_price=summary.total_charging_price,
            total_lateness=summary.total_lateness,
            total_battery_violation=summary.total_battery_violation,
            total_capacity_violation=summary.total_capacity_violation,
            duplicate_customer_visits=summary.duplicate_customer_visits,
            depot_return_violation=summary.depot_return_violation,
            used_evs=summary.used_evs,
            total_duration=summary.total_duration,
            infeasible_flag=summary.infeasible_flag,
            num_served_customers=0,
            unresolved_reserved=summary.unresolved_reserved_customers,
        )

        # Global penalties not attributed per step (unserved / reserved / infeasible).
        w = self.weights
        terminal_j = (
            w.penalty_unserved * summary.unserved_customers
            + w.penalty_reserved * summary.unresolved_reserved_customers
            + w.penalty_infeasible * summary.infeasible_flag
        )
        terminal_j += w.lambda_vehicle * max(0, summary.used_evs - before.used_evs)

        reward = -terminal_j / self.reward_scale
        return StepInfo(
            ev_id=-1,
            selected_node=-1,
            node_type="terminal",
            previous_node=-1,
            immediate_penalty=terminal_j,
            delta_objective=terminal_j,
            reward=reward,
            raw_reward=reward,
            terminal=True,
            extra={"terminal_penalty_cost": terminal_j},
        )

    def verify_alignment(
        self,
        step_infos: list[StepInfo],
        final_objective: float,
        *,
        atol: float = 1e-3,
    ) -> tuple[float, float, bool]:
        """Compare reconstructed objective from step deltas to ``compute_cost`` result."""
        reconstructed = sum(s.delta_objective for s in step_infos)
        mismatch = abs(reconstructed - float(final_objective))
        rel = mismatch / max(1.0, abs(float(final_objective)))
        return mismatch, rel, mismatch <= atol
