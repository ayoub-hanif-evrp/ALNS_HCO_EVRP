"""Compact rollout traces for representative-episode solution diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from src.env.cost import CostWeights
from src.env.policy_action import apply_policy_action
from src.env.evrptw_env import EVRPTWEnv
from src.env.tw_diagnostics import aggregate_mask_reasons, count_time_expired_customers


def _node_type(inst, node_id: int) -> str:
    n = inst.nodes[node_id]
    if n.is_depot:
        return "depot"
    if n.is_charging_station:
        return "station"
    if n.is_customer:
        return "customer"
    return "other"


def _serialize_nodes(instance) -> list[dict[str, Any]]:
    out = []
    for n in instance.nodes:
        out.append({
            "id": int(n.id),
            "string_id": n.string_id,
            "type": _node_type(instance, n.id),
            "x": float(n.x),
            "y": float(n.y),
            "demand": float(n.demand),
            "ready_time": float(n.ready_time),
            "due_time": float(n.due_time),
            "service_time": float(n.service_time),
        })
    return out


def _serialize_routes(env: EVRPTWEnv, instance) -> list[dict[str, Any]]:
    routes = []
    for ev in env.evs:
        if not ev.used() and not ev.finished:
            continue
        visits = []
        for k, node in enumerate(ev.route_nodes):
            visits.append({
                "node": int(node),
                "type": _node_type(instance, node),
                "arrival": float(ev.arrival_times[k]) if k < len(ev.arrival_times) else None,
                "departure": float(ev.departure_times[k]) if k < len(ev.departure_times) else None,
                "battery": float(ev.battery_trace[k]) if k < len(ev.battery_trace) else None,
                "capacity": float(ev.capacity_trace[k]) if k < len(ev.capacity_trace) else None,
            })
        routes.append({
            "ev_id": int(ev.ev_id),
            "route_nodes": [int(x) for x in ev.route_nodes],
            "arrival_times": [float(x) for x in ev.arrival_times],
            "departure_times": [float(x) for x in ev.departure_times],
            "battery_trace": [float(x) for x in ev.battery_trace],
            "capacity_trace": [float(x) for x in ev.capacity_trace],
            "route_distance": float(ev.route_distance),
            "total_waiting_time": float(ev.total_waiting_time),
            "total_charging_time": float(ev.total_charging_time),
            "charging_actions": list(ev.charging_actions),
            "visits": visits,
            "finished": bool(ev.finished),
        })
    return routes


def run_traced_episode(
    instance,
    policy,
    m: int,
    weights: CostWeights,
    *,
    mode: str = "greedy",
    device: str = "cpu",
    charging_portions: list[float] | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    max_steps_factor: int = 6,
    hard_time_windows: bool = True,
    selection_tag: str = "unknown",
    log_row: dict | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one episode and return (EpisodeResult, compact trace dict)."""
    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=charging_portions or policy.charging_portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        max_steps_factor=max_steps_factor,
        hard_time_windows=hard_time_windows,
    )
    inst = instance
    decisions: list[dict[str, Any]] = []
    env_transitions: list[dict[str, Any]] = []
    decision_idx = 0

    node_embeddings = policy.encode(instance, device=device)

    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            break

        close_info = env.try_auto_close_route(ev)
        if close_info is not None:
            env_transitions.append({
                "index": decision_idx,
                "type": "auto_close",
                "ev_id": int(close_info.get("ev_id", ev.ev_id)),
                "info": {k: close_info.get(k) for k in (
                    "type", "from_node", "to_node", "delta_travel_time",
                    "delta_energy_used", "battery_before", "battery_after",
                )},
            })
            decision_idx += 1
            continue

        mask_counts, feasible_mask, mask_reasons = aggregate_mask_reasons(
            inst, env, ev,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
        )
        mask_result = env.last_action_mask_result
        mc = mask_counts.to_dict()
        n_feas_cust = int(mc.get("feasible_customers_count", 0))
        n_feas_st = int(mc.get("feasible_stations_count", 0))
        n_feas_dep = int(mc.get("feasible_depot", 0))

        action = policy.select_action(env, ev, node_embeddings, mask_result, mode=mode)

        batt_before = float(ev.remaining_battery)
        time_before = float(ev.current_time)
        cap_before = float(ev.remaining_capacity)
        pre_expired = count_time_expired_customers(env)
        pre_served = int(env.num_served_customers)

        info = apply_policy_action(
            env, ev, action, mask_result=mask_result, collect_diagnostics=True,
        )
        node = inst.nodes[action.node]
        atype = str(info.get("type", "unknown"))
        if node.is_depot and action.depot_finish:
            atype = "depot_finish"
        elif node.is_depot:
            atype = "depot_recharge"

        service_start = info.get("service_start")
        slack = None
        if node.is_customer and service_start is not None:
            slack = float(node.due_time) - float(service_start)

        decisions.append({
            "decision_index": decision_idx,
            "ev_id": int(ev.ev_id),
            "action_type": atype,
            "node": int(action.node),
            "node_type": _node_type(inst, action.node),
            "charge_portion": action.charge_portion,
            "depot_finish": bool(action.depot_finish),
            "usefulness_before_action": info.get("usefulness_before_action"),
            "usefulness_reason_before_action": info.get("usefulness_reason_before_action"),
            "best_charge_portion_for_station_usefulness": info.get(
                "best_charge_portion_for_station_usefulness",
            ),
            "battery_before": batt_before,
            "battery_after": float(ev.remaining_battery),
            "capacity_before": cap_before,
            "capacity_after": float(ev.remaining_capacity),
            "time_before": time_before,
            "time_after": float(ev.current_time),
            "arrival": float(info.get("arrival", ev.current_time)),
            "departure": float(info.get("departure", ev.current_time)),
            "waiting": float(info.get("waiting", 0.0)),
            "charging_time": float(info.get("charging_time", 0.0)),
            "service_start": float(service_start) if service_start is not None else None,
            "service_slack": slack,
            "customer_served": bool(info.get("customer_served", False)),
            "feasible_customers": n_feas_cust,
            "feasible_stations": n_feas_st,
            "feasible_depot": n_feas_dep,
            "masked_by_time_window": float(mc.get("masked_by_time_window_count", 0)),
            "masked_by_battery": float(mc.get("masked_by_battery_count", 0)),
            "masked_by_capacity": float(mc.get("masked_by_capacity_count", 0)),
            "expired_before": pre_expired,
            "expired_after": count_time_expired_customers(env),
            "unserved_before": int(inst.num_customers - pre_served),
            "unserved_after": int(inst.num_customers - env.num_served_customers),
            "lateness": float(info.get("lateness", 0.0)),
        })
        decision_idx += 1

    for ev in env.evs:
        if not ev.finished:
            close_info = env.auto_close_route(ev, forced=True, return_info=True)
            env_transitions.append({
                "index": decision_idx,
                "type": "auto_close_forced",
                "ev_id": int(ev.ev_id),
                "info": {k: close_info.get(k) for k in ("type", "from_node", "to_node", "lateness")},
            })
            decision_idx += 1

    j, breakdown, summary = env.score(weights)
    served = int(env.num_served_customers)
    trace = {
        "meta": {
            "trace_source": "final_policy_replay",
            "trace_semantics": "Final-policy replay of selected instance (not exact training rollout)",
            "selection_tag": selection_tag,
            "instance_name": getattr(instance, "instance_name", "?"),
            "epoch": int((log_row or {}).get("epoch", -1)),
            "cycle": int((log_row or {}).get("cycle", -1)),
            "m": int(m),
            "mode": mode,
            "objective": float(j),
            "feasible": bool(env.feasible_solution()),
            "served_ratio": served / max(inst.num_customers, 1),
            "unserved_customers": int(summary.unserved_customers),
            "used_evs": int(summary.used_evs),
            "total_distance": float(summary.total_distance),
            "total_charging_time": float(summary.total_charging_time),
            "customer_lateness": float(getattr(summary, "customer_lateness", env.customer_lateness)),
            "depot_lateness": float(getattr(summary, "depot_lateness", env.depot_lateness)),
            "battery_violation": float(summary.total_battery_violation),
            "relative_objective_mismatch": float((log_row or {}).get("relative_objective_mismatch", 0.0)),
        },
        "nodes": _serialize_nodes(instance),
        "decisions": decisions,
        "env_transitions": env_transitions,
        "routes": _serialize_routes(env, instance),
        "served_mask": [bool(x) for x in env.served_mask.tolist()],
        "summary": {
            "objective": float(j),
            "breakdown": {k: float(v) for k, v in breakdown.items()},
            "unserved_customers": int(summary.unserved_customers),
            "used_evs": int(summary.used_evs),
            "total_distance": float(summary.total_distance),
            "total_duration": float(summary.total_duration),
            "total_charging_time": float(summary.total_charging_time),
            "total_waiting_time": float(summary.total_waiting_time),
            "customer_lateness": float(getattr(summary, "customer_lateness", 0.0)),
            "depot_lateness": float(getattr(summary, "depot_lateness", 0.0)),
            "total_battery_violation": float(summary.total_battery_violation),
        },
    }

    from src.training.rollout import EpisodeResult

    result = EpisodeResult(
        env=env,
        objective=j,
        breakdown=breakdown,
        summary=summary,
        log_prob_sum=torch.tensor(0.0),
        entropy_sum=torch.tensor(0.0),
        feasible=env.feasible_solution(),
        num_decisions=len(decisions),
        done_reason=env.get_done_reason(),
        fleet_size=m,
    )
    return result, trace
