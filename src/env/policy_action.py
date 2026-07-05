"""Unified policy action application with pre-action diagnostics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .action_space import (
    compute_depot_finish_validity,
    compute_depot_recharge_usefulness,
    compute_full_feasible_actions,
    compute_global_serviceability,
    compute_station_usefulness,
    evaluate_charge_portions_for_node,
)
from .ev_state import EVState
from .tw_diagnostics import count_time_expired_customers


@dataclass
class PolicyAction:
    """Explicit EVRPTW action: next node, charge portion, depot finish vs recharge."""

    node: int
    charge_portion: float | None = None
    depot_finish: bool = False


# Public alias used by baselines and tests.
Action = PolicyAction


def resolve_depot_action(
    env,
    ev: EVState,
    node: int,
    charge_portion: float | None = None,
) -> PolicyAction:
    """Choose explicit depot finish vs recharge from the current feasible mask."""
    from .masks import depot_mode_feasible

    inst = env.instance
    if node != inst.depot_index:
        return PolicyAction(node=node, charge_portion=charge_portion, depot_finish=False)

    ok_r, ok_f = depot_mode_feasible(
        inst, env, ev,
        use_horizon_check=env.use_horizon_check,
        use_depot_charging=env.use_depot_charging,
        charging_portions=env.charging_portions,
    )
    all_done = env.num_served_customers >= inst.num_customers
    if all_done and ok_f:
        return PolicyAction(node=node, charge_portion=None, depot_finish=True)
    if ok_r:
        return PolicyAction(
            node=node,
            charge_portion=charge_portion if charge_portion is not None else 1.0,
            depot_finish=False,
        )
    if ok_f:
        return PolicyAction(node=node, charge_portion=None, depot_finish=True)
    return PolicyAction(node=node, charge_portion=charge_portion, depot_finish=False)


def policy_action_from_output(action: Any) -> PolicyAction:
    return PolicyAction(
        node=int(action.node),
        charge_portion=getattr(action, "charge_portion", None),
        depot_finish=bool(getattr(action, "depot_finish", False)),
    )


def compute_pre_action_context(
    env,
    ev: EVState,
    node: int,
    *,
    charge_portion: float | None = None,
    depot_finish: bool = False,
) -> dict:
    """Pre-action usefulness, serviceability, and feasible-mask snapshot."""
    inst = env.instance
    gs = compute_global_serviceability(
        inst, env.evs, env.served_mask, env.reserved_mask,
        use_horizon_check=env.use_horizon_check,
        use_depot_charging=env.use_depot_charging,
        charging_portions=env.charging_portions,
    )
    mask, reasons, summary, _ = compute_full_feasible_actions(
        inst, env, ev,
        env.use_horizon_check, env.use_depot_charging, env.charging_portions,
    )
    all_done = env.num_served_customers >= inst.num_customers
    node_obj = inst.nodes[node]
    ctx: dict = {
        "pre_global_serviceability": gs.to_dict(),
        "pre_feasible_customers": float(summary.has_customer),
        "pre_feasible_stations": float(summary.has_useful_station),
        "pre_feasible_depot_recharge": float(summary.has_useful_depot_recharge),
        "pre_feasible_depot_finish": float(summary.has_depot_finish),
        "pre_expired_customers": float(count_time_expired_customers(env)),
        "pre_num_served": int(env.num_served_customers),
        "selected_charge_portion": charge_portion,
        "depot_finish": bool(depot_finish),
    }

    if node_obj.is_charging_station:
        u = compute_station_usefulness(
            inst, ev, node, env.served_mask, env.reserved_mask,
            all_customers_served=all_done,
            global_svc=gs,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
            charging_portions=env.charging_portions,
        )
        cp = evaluate_charge_portions_for_node(
            inst, ev, node, env.served_mask, env.reserved_mask,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
            charging_portions=env.charging_portions,
        )
        ctx.update({
            "usefulness_before_action": bool(u.useful),
            "usefulness_reason_before_action": u.reason,
            "useful_station": bool(u.useful),
            "enabled_customers": int(u.enabled_customers_count),
            **cp.to_log_dict(),
        })
    elif node_obj.is_depot and not depot_finish:
        useful, reason, enables = compute_depot_recharge_usefulness(
            inst, ev, env.served_mask, env.reserved_mask,
            all_customers_served=all_done,
            global_svc=gs,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
            charging_portions=env.charging_portions,
        )
        ctx.update({
            "usefulness_before_action": bool(useful),
            "usefulness_reason_before_action": reason,
            "useful_depot_recharge": bool(useful),
            "depot_recharge_enables_customer_count": int(enables),
        })
    elif node_obj.is_depot and depot_finish:
        fv = compute_depot_finish_validity(
            inst, ev, env.served_mask, env.reserved_mask, env.num_served_customers,
            global_svc=gs,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
            charging_portions=env.charging_portions,
        )
        ctx.update({
            "depot_finish_validity_reason": fv.reason,
            "valid_depot_finish": fv.valid and fv.category != "invalid",
            "early_depot_finish": fv.category == "early",
            "forced_depot_finish": fv.category == "forced",
            "useful_station_exists_before_finish": bool(fv.useful_station_exists),
            "useful_depot_recharge_before_finish": bool(fv.depot_recharge_could_enable_service),
            "depot_recharge_could_enable_service": bool(fv.depot_recharge_could_enable_service),
            "future_service_after_recharge_exists": bool(fv.finish_masked_due_to_future_service),
            "finish_masked_due_to_future_service_possible": bool(fv.finish_masked_due_to_future_service),
            "finish_would_reduce_global_serviceability": bool(fv.finish_would_reduce_global_serviceability),
        })

    _ = mask, reasons  # pre-action mask available for debugging via env.last_feasible_summary
    return ctx


def _attach_post_action_metrics(env, info: dict, pre: dict) -> dict:
    inst = env.instance
    gs_after = compute_global_serviceability(
        inst, env.evs, env.served_mask, env.reserved_mask,
        use_horizon_check=env.use_horizon_check,
        use_depot_charging=env.use_depot_charging,
        charging_portions=env.charging_portions,
    )
    info["post_global_serviceability"] = gs_after.to_dict()
    info["globally_serviceable_customers_after"] = float(
        gs_after.globally_serviceable_customers_count,
    )
    info["newly_expired_customers_after"] = float(
        max(0, count_time_expired_customers(env) - int(pre.get("pre_expired_customers", 0)))
    )
    pre_enabled = int(pre.get("enabled_customers", pre.get("depot_recharge_enables_customer_count", 0)))
    if pre_enabled > 0:
        info["newly_enabled_customers_after_action"] = pre_enabled
    info["usefulness_after_action"] = info.get("usefulness_before_action")
    return info


def apply_policy_action(env, ev: EVState, action: Any) -> dict:
    """Apply a policy action using pre-action diagnostics (single entry point)."""
    pa = action if isinstance(action, PolicyAction) else policy_action_from_output(action)
    pre = compute_pre_action_context(
        env, ev, pa.node,
        charge_portion=pa.charge_portion,
        depot_finish=pa.depot_finish,
    )
    info = env.apply_action(
        ev, pa.node, pa.charge_portion,
        depot_finish=pa.depot_finish,
        pre_action_context=pre,
    )
    return _attach_post_action_metrics(env, info, pre)
