"""Time-window diagnostics: mask reasons, expired customers, temporal action features."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.hanif.env.masks import (
    EPS,
    compute_feasible_mask,
    customer_feasible,
    depot_recharge_feasible,
    station_feasible,
)

DYNAMIC_ACTION_FEAT_DIM = 8


@dataclass
class MaskReasonCounts:
    feasible_customers_count: int = 0
    masked_by_time_window_count: int = 0
    masked_by_capacity_count: int = 0
    masked_by_battery_count: int = 0
    masked_by_reachability_count: int = 0
    masked_by_already_served_count: int = 0
    masked_by_reservation_count: int = 0
    feasible_stations_count: int = 0
    feasible_depot: int = 0

    def to_dict(self) -> dict[str, float]:
        total_c = (
            self.feasible_customers_count
            + self.masked_by_time_window_count
            + self.masked_by_capacity_count
            + self.masked_by_battery_count
            + self.masked_by_reachability_count
            + self.masked_by_already_served_count
            + self.masked_by_reservation_count
        )
        total_c = max(total_c, 1)
        return {
            "feasible_customers_count": float(self.feasible_customers_count),
            "masked_by_time_window_count": float(self.masked_by_time_window_count),
            "masked_by_capacity_count": float(self.masked_by_capacity_count),
            "masked_by_battery_count": float(self.masked_by_battery_count),
            "masked_by_reachability_count": float(self.masked_by_reachability_count),
            "masked_by_already_served_count": float(self.masked_by_already_served_count),
            "masked_by_reservation_count": float(self.masked_by_reservation_count),
            "feasible_stations_count": float(self.feasible_stations_count),
            "feasible_depot": float(self.feasible_depot),
            "feasible_customer_ratio": self.feasible_customers_count / total_c,
            "time_window_masked_customer_ratio": self.masked_by_time_window_count / total_c,
        }


def customer_service_start(instance, ev, j: int) -> tuple[float, float, float, float]:
    """Return (arrival, waiting, service_start, slack) for customer j from ev."""
    i = ev.current_node
    travel = float(instance.travel_time_matrix[i, j])
    arrival = float(ev.current_time + travel)
    ready = float(instance.nodes[j].ready_time)
    due = float(instance.nodes[j].due_time)
    waiting = max(0.0, ready - arrival)
    service_start = max(arrival, ready)
    slack = due - service_start
    return arrival, waiting, service_start, slack


def is_customer_time_expired(instance, ev, j: int, served_mask, reserved_mask) -> bool:
    """True if unserved customer j cannot be served in time by this EV (or globally)."""
    if served_mask[j] or reserved_mask[j]:
        return False
    ok, reason = customer_feasible(instance, ev, j, served_mask, reserved_mask, use_horizon_check=True)
    if ok:
        return False
    if reason == "customer_time_window_infeasible" or reason == "time_window_late":
        return True
    _, _, service_start, _ = customer_service_start(instance, ev, j)
    return service_start > float(instance.nodes[j].due_time) + EPS


def count_time_expired_customers(env) -> int:
    """Unserved customers with no feasible service due to closed time windows."""
    analysis = getattr(env, "last_decision_analysis", None)
    if (
        analysis is not None
        and int(getattr(analysis, "state_version", -1)) == int(getattr(env, "state_version", -2))
        and analysis.global_serviceability is not None
    ):
        gs = analysis.global_serviceability
        return int(getattr(gs, "globally_expired_customers_count", getattr(env, "last_expired_count", 0)))
    cached = getattr(env, "last_expired_count", None)
    if cached is not None and getattr(env, "_global_serviceability_cache", None) is not None:
        return int(cached)
    env.get_global_serviceability()
    return int(getattr(env, "last_expired_count", 0))


def aggregate_mask_reasons(
    instance,
    env_state,
    ev,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    analysis=None,
) -> tuple[MaskReasonCounts, np.ndarray, dict[int, str]]:
    """Aggregate mask diagnostics from the canonical decision analysis."""
    from src.hanif.env.decision_analysis import analyze_decision_state

    if analysis is None:
        analysis = getattr(env_state, "last_decision_analysis", None)
    fresh = (
        analysis is not None
        and int(analysis.state_version) == int(getattr(env_state, "state_version", -1))
        and int(analysis.active_ev_id) == int(ev.ev_id)
    )
    if not fresh:
        if hasattr(env_state, "analyze_decision"):
            analysis = env_state.analyze_decision(ev)
        else:
            analysis = analyze_decision_state(env_state, ev)
    result = analysis.mask_result
    counts = MaskReasonCounts()
    feasible = analysis.final_node_mask
    reasons = analysis.node_reasons
    # Prefer precomputed reason counts from analysis when present.
    if analysis.reason_counts:
        mrc = analysis.reason_counts
        counts.feasible_customers_count = int(mrc.get("feasible_customers", 0))
        counts.feasible_stations_count = int(mrc.get("feasible_stations", 0))
        counts.masked_by_time_window_count = int(mrc.get("masked_by_time_window", 0))
        counts.masked_by_battery_count = int(mrc.get("masked_by_battery", 0))
        counts.masked_by_capacity_count = int(mrc.get("masked_by_capacity", 0))
        counts.masked_by_already_served_count = int(mrc.get("masked_by_already_served", 0))
        counts.masked_by_reservation_count = int(mrc.get("masked_by_reservation", 0))
        return counts, feasible, reasons
    for j in instance.customer_indices:
        if feasible[j]:
            counts.feasible_customers_count += 1
            continue
        r = reasons.get(j, "")
        if r in ("customer_already_served", "already_served", "hard_customer_already_served"):
            counts.masked_by_already_served_count += 1
        elif r in ("customer_reserved", "reserved_by_other", "hard_customer_reserved"):
            counts.masked_by_reservation_count += 1
        elif r in ("customer_capacity_infeasible", "capacity_exceeded", "hard_customer_capacity_infeasible"):
            counts.masked_by_capacity_count += 1
        elif r in (
            "customer_battery_unreachable", "battery_cannot_reach", "battery_would_strand",
            "customer_infeasible_after_service_battery", "hard_customer_battery_unreachable",
        ):
            counts.masked_by_battery_count += 1
        elif r in (
            "customer_time_window_infeasible", "time_window_late", "depot_window_late",
            "horizon_exceeded", "customer_infeasible_after_service_time",
            "hard_customer_time_window_infeasible",
        ):
            counts.masked_by_time_window_count += 1
        elif r in (
            "customer_no_safe_continuation", "customer_no_safe_direct_depot",
            "customer_no_reachable_station", "customer_station_has_no_feasible_charge",
            "customer_infeasible_no_safe_continuation",
            "customer_infeasible_no_reachable_station_or_depot",
        ):
            counts.masked_by_reachability_count += 1
    for s in instance.charging_station_indices:
        if feasible[s]:
            counts.feasible_stations_count += 1
    depot = instance.depot_index
    if feasible[depot]:
        counts.feasible_depot = 1
    if result is not None and getattr(result, "mask_reason_counts", None):
        mrc = result.mask_reason_counts
        counts.feasible_customers_count = max(
            counts.feasible_customers_count, int(mrc.get("feasible_customers", 0)),
        )
        counts.feasible_stations_count = max(
            counts.feasible_stations_count, int(mrc.get("feasible_stations", 0)),
        )
    return counts, feasible, reasons


def station_is_necessary(instance, ev, served_mask, reserved_mask, use_depot_charging: bool) -> bool:
    """True if any charging station could be necessary (logit-bias helper)."""
    for s in instance.charging_station_indices:
        necessary, _ = selected_station_is_necessary(
            instance, ev, s, served_mask, reserved_mask,
            use_depot_charging=use_depot_charging,
        )
        if necessary:
            return True
    return False


def _feasible_customer_indices(
    instance, ev, served_mask, reserved_mask, *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> list[int]:
    out: list[int] = []
    for j in instance.customer_indices:
        ok, _ = customer_feasible(
            instance, ev, j, served_mask, reserved_mask, use_horizon_check,
            use_depot_charging, charging_portions,
        )
        if ok:
            out.append(j)
    return out


def _count_feasible_after_station_recharge(
    instance,
    ev,
    station: int,
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> int:
    from src.hanif.env.action_space import evaluate_charge_portions_for_node

    analysis = evaluate_charge_portions_for_node(
        instance, ev, station, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    if analysis.best is None:
        return 0
    from src.hanif.env.action_space import _feasible_customer_indices
    feasible_now = len(_feasible_customer_indices(
        instance, ev, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    ))
    return feasible_now + analysis.best.enabled_customers


def simulate_departure_after_best_charge(
    instance, ev, station, served_mask, reserved_mask, portions,
    use_horizon_check, use_depot_charging,
) -> tuple[float, float]:
    from src.hanif.env.action_space import evaluate_charge_portions_for_node
    from src.hanif.env.charge_masks import simulate_recharge_departure

    analysis = evaluate_charge_portions_for_node(
        instance, ev, station, served_mask, reserved_mask,
        charging_portions=portions or [0.25, 0.5, 0.75, 1.0],
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
    )
    if analysis.best is None:
        return ev.current_time, ev.remaining_battery
    return simulate_recharge_departure(
        instance, ev, station, analysis.best.portion, free=instance.nodes[station].is_depot,
    )[:2]


def selected_station_is_necessary(
    env_or_instance,
    ev,
    selected_station: int,
    served_mask=None,
    reserved_mask=None,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    analysis=None,
) -> tuple[bool, dict[str, float]]:
    """Return whether the selected station is useful for the current EV decision.

    Prefer ``analysis.active_ev_analysis.recharge_node_results`` — do not recompute.
    """
    if hasattr(env_or_instance, "instance"):
        env = env_or_instance
        inst = env.instance
        served = env.served_mask if served_mask is None else served_mask
        reserved = env.reserved_mask if reserved_mask is None else reserved_mask
        portions = charging_portions or getattr(env, "charging_portions", None)
        ev_ref = ev
    else:
        inst = env_or_instance
        served = served_mask
        reserved = reserved_mask
        portions = charging_portions
        ev_ref = ev

    if selected_station < 0 or not inst.nodes[selected_station].is_charging_station:
        return False, {
            "selected_station_necessary": 0.0,
            "unnecessary_selected_station": 1.0,
            "selected_station_no_continuation": 1.0,
            "selected_station_enables_customer_count": 0.0,
        }

    # Hot path: reuse cached recharge-node result from DecisionStateAnalysis.
    ea = None
    if analysis is not None:
        ea = getattr(analysis, "active_ev_analysis", None)
    elif hasattr(env_or_instance, "last_decision_analysis"):
        last = env_or_instance.last_decision_analysis
        if (
            last is not None
            and int(getattr(last, "state_version", -1)) == int(getattr(env_or_instance, "state_version", -2))
            and int(getattr(last, "active_ev_id", -1)) == int(ev_ref.ev_id)
        ):
            ea = getattr(last, "active_ev_analysis", None)
    if ea is not None:
        nr = ea.recharge_node_results.get(int(selected_station))
        if nr is not None:
            useful = bool(nr.useful)
            return useful, {
                "selected_station_necessary": float(useful),
                "unnecessary_selected_station": float(not useful),
                "selected_station_no_continuation": float(not useful and not nr.enables_depot_return),
                "selected_station_enables_customer_count": float(nr.enabled_customer_count),
            }
        # Station absent from recharge labels → not useful for continuation.
        return False, {
            "selected_station_necessary": 0.0,
            "unnecessary_selected_station": 1.0,
            "selected_station_no_continuation": 1.0,
            "selected_station_enables_customer_count": 0.0,
        }

    from src.hanif.env.action_space import compute_station_usefulness, compute_global_serviceability
    from src.hanif.env.analysis_counters import bump_outside_analyze

    bump_outside_analyze("station_usefulness_recompute_calls")
    all_done = int(np.sum(served[inst.customer_indices])) >= inst.num_customers if inst.customer_indices else True
    if hasattr(env_or_instance, "get_global_serviceability"):
        gs = env_or_instance.get_global_serviceability()
    else:
        gs = compute_global_serviceability(
            inst, [ev_ref], served, reserved,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
        )
    u = compute_station_usefulness(
        inst, ev_ref, selected_station, served, reserved,
        all_customers_served=all_done,
        global_svc=gs,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=portions,
    )
    diag: dict[str, float] = {
        "selected_station_necessary": float(u.useful),
        "unnecessary_selected_station": float(not u.useful),
        "selected_station_no_continuation": float(not u.useful and not u.enables_depot_return),
        "selected_station_enables_customer_count": float(u.enabled_customers_count),
    }
    return u.useful, diag


def compute_dynamic_action_features(env, ev) -> np.ndarray:
    """Per-node temporal features for pointer bias [N, DYNAMIC_ACTION_FEAT_DIM]."""
    inst = env.instance
    n = inst.num_nodes
    horizon = inst.planning_horizon if inst.planning_horizon > 0 else 1.0
    max_dist = float(inst.distance_matrix.max() * max(n, 1)) or 1.0
    feats = np.zeros((n, DYNAMIC_ACTION_FEAT_DIM), dtype=np.float32)
    served = env.served_mask
    reserved = env.reserved_mask
    for j in range(n):
        node = inst.nodes[j]
        travel = float(inst.travel_time_matrix[ev.current_node, j]) / horizon
        if node.is_customer:
            arr, wait, ss, slack = customer_service_start(inst, ev, j)
            ok, _ = customer_feasible(inst, ev, j, served, reserved, True)
            feats[j] = [
                arr / horizon,
                ss / horizon,
                max(0.0, slack) / horizon,
                wait / horizon,
                travel,
                float(node.is_customer),
                float(node.is_charging_station or (node.is_depot and inst.depot_has_charging)),
                float(ok),
            ]
        elif node.is_charging_station or node.is_depot:
            arr = float(ev.current_time + inst.travel_time_matrix[ev.current_node, j]) / horizon
            feats[j] = [arr, arr, 0.0, 0.0, travel, 0.0, 1.0, 0.0]
        else:
            feats[j, 4] = travel
    _ = max_dist  # reserved for future distance-normalized features
    return feats


@dataclass
class UrgentRankStats:
    selected_customer_slack_rank: int = -1
    selected_in_top5_urgent: float = 0.0
    selected_in_top10_urgent: float = 0.0
    served_customer_slack: float = 0.0
    mean_feasible_customer_slack: float = 0.0
    min_feasible_customer_slack: float = 0.0
    max_feasible_customer_slack: float = 0.0


def urgent_rank_stats(env, ev, selected_node: int, feasible_mask: np.ndarray, k: int = 10) -> UrgentRankStats:
    inst = env.instance
    slacks: list[tuple[float, int]] = []
    for j in inst.customer_indices:
        if not feasible_mask[j]:
            continue
        _, _, _, slack = customer_service_start(inst, ev, j)
        slacks.append((slack, j))
    slacks.sort(key=lambda x: x[0])
    stats = UrgentRankStats()
    if slacks:
        stats.min_feasible_customer_slack = float(slacks[0][0])
        stats.max_feasible_customer_slack = float(slacks[-1][0])
        stats.mean_feasible_customer_slack = float(np.mean([s for s, _ in slacks]))
    if inst.nodes[selected_node].is_customer and feasible_mask[selected_node]:
        _, _, _, slack = customer_service_start(inst, ev, selected_node)
        stats.served_customer_slack = float(slack)
        rank = next((i + 1 for i, (_, nid) in enumerate(slacks) if nid == selected_node), -1)
        stats.selected_customer_slack_rank = rank
        stats.selected_in_top5_urgent = float(rank > 0 and rank <= 5)
        stats.selected_in_top10_urgent = float(rank > 0 and rank <= k)
    return stats
