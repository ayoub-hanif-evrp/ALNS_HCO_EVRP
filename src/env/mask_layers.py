"""Hard feasibility vs optional heuristic pruning mask layers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .action_space import (
    ActionProgressTracker,
    FeasibleActionSummary,
    GlobalServiceability,
    compute_depot_finish_validity,
    compute_depot_recharge_usefulness,
    compute_station_usefulness,
    would_be_no_progress_action,
)
from .charge_masks import feasible_charge_portions
from .ev_state import EVState
from .masks import (
    customer_feasible,
    depot_finish_feasible,
    depot_recharge_feasible,
    station_feasible,
)


@dataclass
class ActionPruningConfig:
    """Safe-only pruning defaults — unproven strategic pruning stays off."""

    enabled: bool = True
    block_exact_noop_actions: bool = True
    block_repeated_zero_progress_loops: bool = True
    prune_dominated_recharge_labels: bool = True
    prune_station_when_customer_available: bool = False
    # Legacy aliases kept for older configs / tests.
    block_no_progress_loops: bool = True
    prune_dominated_stations: bool = False

    @classmethod
    def from_env_config(cls, cfg: dict | None) -> "ActionPruningConfig":
        raw = (cfg or {}).get("action_pruning") or {}
        block_loops = bool(raw.get(
            "block_repeated_zero_progress_loops",
            raw.get("block_no_progress_loops", True),
        ))
        return cls(
            enabled=bool(raw.get("enabled", True)),
            block_exact_noop_actions=bool(raw.get("block_exact_noop_actions", True)),
            block_repeated_zero_progress_loops=block_loops,
            prune_dominated_recharge_labels=bool(raw.get("prune_dominated_recharge_labels", True)),
            prune_station_when_customer_available=bool(
                raw.get("prune_station_when_customer_available", False)
            ),
            block_no_progress_loops=block_loops,
            # Only prune stations for usefulness when explicitly opting into
            # "customer available ⇒ stations useless" (scientifically unsafe).
            prune_dominated_stations=bool(
                raw.get(
                    "prune_dominated_stations",
                    raw.get("prune_station_when_customer_available", False),
                )
            ),
        )


def _hard_reason(reason: str) -> str:
    if reason.startswith("hard_"):
        return reason
    return f"hard_{reason}"


def _prune_reason(reason: str) -> str:
    if reason.startswith("pruned_"):
        return reason
    return f"pruned_{reason}"


def compute_hard_feasibility_mask(
    instance,
    env_state,
    ev: EVState,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[np.ndarray, dict[int, str], dict[int, np.ndarray], dict[int, list[str]], np.ndarray, list[str]]:
    """Physical EVRPTW feasibility only — no usefulness or dominance pruning.

    When ``env_state._pending_customer_prefilter`` is set by the optimized
    analyzer, only prefilter survivors run expensive ``customer_feasible``
    (safe-continuation / recharge) analysis.
    """
    n = instance.num_nodes
    hard = np.zeros(n, dtype=bool)
    hard_reasons: dict[int, str] = {}
    hard_charge: dict[int, np.ndarray] = {}
    hard_charge_reasons: dict[int, list[str]] = {}
    served = env_state.served_mask
    reserved = env_state.reserved_mask
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    all_done = env_state.num_served_customers >= instance.num_customers

    pending = getattr(env_state, "_pending_customer_prefilter", None)
    ev_analysis = getattr(env_state, "_pending_ev_analysis", None)
    if ev_analysis is not None:
        cust = np.asarray(instance.customer_indices, dtype=np.int64)
        basic = ev_analysis.direct_basic_customer_mask
        safe = ev_analysis.direct_safe_customer_mask
        for i, j in enumerate(cust):
            jj = int(j)
            if safe[i]:
                hard[jj] = True
            else:
                hard[jj] = False
                reason = ev_analysis.direct_customer_reasons.get(jj, "customer_infeasible")
                hard_reasons[jj] = _hard_reason(reason)
    elif pending is not None:
        basic_ok = pending["basic_ok"]
        cust = pending["customer_indices"]
        reject = pending["reject_reasons"]
        for j, reason in reject.items():
            hard[int(j)] = False
            hard_reasons[int(j)] = _hard_reason(reason)
        survivors = cust[basic_ok] if len(cust) else []
        for j in survivors:
            ok, reason = customer_feasible(
                instance, ev, int(j), served, reserved,
                use_horizon_check, use_depot_charging, portions,
            )
            hard[int(j)] = ok
            if not ok:
                hard_reasons[int(j)] = _hard_reason(reason)
    else:
        for j in instance.customer_indices:
            ok, reason = customer_feasible(
                instance, ev, j, served, reserved,
                use_horizon_check, use_depot_charging, portions,
            )
            hard[j] = ok
            if not ok:
                hard_reasons[j] = _hard_reason(reason)

    for s in instance.charging_station_indices:
        ok, reason = station_feasible(
            instance, ev, s, served, reserved,
            use_horizon_check, use_depot_charging, portions,
        )
        if not ok:
            hard[s] = False
            hard_reasons[s] = _hard_reason(reason)
            continue
        cm, cr = feasible_charge_portions(
            instance, ev, s, served, reserved, portions,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
        )
        if not cm.any():
            hard[s] = False
            hard_reasons[s] = "hard_station_no_feasible_charge"
            continue
        hard[s] = True
        hard_reasons[s] = "hard_station_feasible"
        hard_charge[s] = cm
        hard_charge_reasons[s] = cr

    depot = instance.depot_index
    ok_finish_phys, finish_reason = depot_finish_feasible(instance, ev, use_horizon_check)
    hard_finish = False
    finish_hard_reason = _hard_reason(finish_reason)
    if all_done and ok_finish_phys:
        hard_finish = True
        finish_hard_reason = "finish_all_customers_served"
    elif ok_finish_phys:
        hard_customers = [c for c in instance.customer_indices if hard[c]]
        if hard_customers:
            hard_finish = False
            finish_hard_reason = "finish_invalid_customer_available"
        else:
            hard_finish = True
            finish_hard_reason = "hard_depot_finish_physical"

    ok_recharge_phys, recharge_reason = depot_recharge_feasible(
        instance, ev, served, reserved,
        use_depot_charging, use_horizon_check, portions,
    )
    hard_recharge = False
    recharge_hard_reason = _hard_reason(recharge_reason)
    if ok_recharge_phys and not all_done:
        cm, cr = feasible_charge_portions(
            instance, ev, depot, served, reserved, portions,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
        )
        if cm.any():
            hard_recharge = True
            recharge_hard_reason = "hard_depot_recharge_feasible"
            hard_charge[depot] = cm
            hard_charge_reasons[depot] = cr
        else:
            recharge_hard_reason = "hard_depot_no_feasible_charge"

    hard[depot] = hard_recharge or hard_finish
    if hard[depot]:
        hard_reasons[depot] = recharge_hard_reason if hard_recharge else finish_hard_reason
    else:
        hard_reasons[depot] = finish_hard_reason if not hard_recharge else recharge_hard_reason

    depot_mode_mask = np.array([hard_recharge, hard_finish], dtype=bool)
    depot_mode_reasons = [recharge_hard_reason, finish_hard_reason]
    return hard, hard_reasons, hard_charge, hard_charge_reasons, depot_mode_mask, depot_mode_reasons


def compute_optional_pruning_mask(
    instance,
    env_state,
    ev: EVState,
    hard_mask: np.ndarray,
    hard_reasons: dict[int, str],
    *,
    hard_depot_mode: np.ndarray,
    global_svc: GlobalServiceability,
    pruning: ActionPruningConfig,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[np.ndarray, dict[int, str], np.ndarray, list[str], dict[int, np.ndarray], dict[int, list[str]]]:
    """Heuristic pruning layer — True means action survives pruning."""
    n = instance.num_nodes
    pruning_mask = hard_mask.copy()
    pruning_reasons: dict[int, str] = {}
    served = env_state.served_mask
    reserved = env_state.reserved_mask
    all_done = env_state.num_served_customers >= instance.num_customers
    progress: ActionProgressTracker | None = getattr(env_state, "progress_tracker", None)
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])

    if not pruning.enabled:
        prune_charge = {k: np.ones_like(v, dtype=bool) for k, v in getattr(env_state, "_hard_charge", {}).items()}
        return pruning_mask, pruning_reasons, np.array([True, True]), ["", ""], prune_charge, {}

    from .action_space import _feasible_customer_indices
    from .charge_masks import safe_depot_return_from

    ev_analysis = getattr(env_state, "_pending_ev_analysis", None)

    if ev_analysis is not None:
        cust = np.asarray(instance.customer_indices, dtype=np.int64)
        feasible_now = [int(j) for j in cust[ev_analysis.direct_safe_customer_mask]]
        ok_dep_now = bool(ev_analysis.safe_depot_return_direct)
    else:
        feasible_now = _feasible_customer_indices(
            instance, ev, served, reserved,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
        )
        ok_dep_now, _ = safe_depot_return_from(
            instance, ev.current_node, ev.current_time, ev.remaining_battery,
            use_horizon_check=use_horizon_check,
        )
    feasible_now_count = len(feasible_now)

    hard_charge = getattr(env_state, "_hard_charge", {})

    for s in instance.charging_station_indices:
        if not hard_mask[s]:
            continue
        nr = None
        if ev_analysis is not None:
            nr = ev_analysis.recharge_node_results.get(int(s))
        usefulness = compute_station_usefulness(
            instance, ev, s, served, reserved,
            all_customers_served=all_done,
            global_svc=global_svc,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
            feasible_now=feasible_now,
            ok_dep_now=ok_dep_now,
            precomputed_charge_mask=hard_charge.get(s),
            recharge_node_result=nr,
        )
        # Unsafe heuristic: "customer feasible ⇒ every station useless".
        # Only applied when explicitly enabled.
        if (
            pruning.prune_station_when_customer_available
            and not usefulness.useful
            and usefulness.reason == "station_unnecessary_customer_feasible_now"
        ):
            pruning_mask[s] = False
            pruning_reasons[s] = _prune_reason(usefulness.reason)
            continue
        # Proactive charging is strategically allowed when customers are feasible.
        # Expensive no-progress simulation only when no customer is feasible now
        # or an active zero-progress loop was detected.
        if pruning.block_repeated_zero_progress_loops or pruning.block_exact_noop_actions:
            if progress is not None and bool(progress.loop_detected):
                pruning_mask[s] = False
                pruning_reasons[s] = _prune_reason("no_progress_near_zero_loop")
            elif feasible_now_count == 0:
                no_prog, np_reason = would_be_no_progress_action(
                    instance, ev, s, served, reserved,
                    action_kind="station", depot_finish=False,
                    all_customers_served=all_done,
                    global_svc=global_svc,
                    progress=progress,
                    use_horizon_check=use_horizon_check,
                    use_depot_charging=use_depot_charging,
                    charging_portions=portions,
                    recharge_node_result=nr,
                    feasible_before_count=feasible_now_count,
                    ok_dep_before=ok_dep_now,
                )
                if no_prog:
                    pruning_mask[s] = False
                    pruning_reasons[s] = _prune_reason(np_reason)

    depot = instance.depot_index
    finish_val = compute_depot_finish_validity(
        instance, ev, served, reserved, env_state.num_served_customers,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=portions,
        ev_analysis=ev_analysis,
    )
    hard_recharge = bool(hard_depot_mode[0])
    depot_nr = ev_analysis.recharge_node_results.get(int(depot)) if ev_analysis is not None else None
    if depot_nr is not None and not all_done:
        useful_recharge = bool(depot_nr.enabled_customer_count > 0)
        ur_reason = (
            "depot_recharge_enables_service" if useful_recharge else "depot_recharge_no_future_service"
        )
    else:
        useful_recharge, ur_reason, _ = compute_depot_recharge_usefulness(
            instance, ev, served, reserved,
            all_customers_served=all_done,
            global_svc=global_svc,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
            feasible_now_count=feasible_now_count,
            feasible_now_indices=feasible_now,
            precomputed_charge_mask=hard_charge.get(depot),
        )
    allow_recharge = hard_recharge and useful_recharge
    if hard_recharge and not useful_recharge:
        pruning_reasons[depot] = _prune_reason(ur_reason)
    if hard_recharge and useful_recharge and (
        pruning.block_repeated_zero_progress_loops or pruning.block_exact_noop_actions
    ):
        no_prog, np_reason = would_be_no_progress_action(
            instance, ev, depot, served, reserved,
            action_kind="depot_recharge", depot_finish=False,
            all_customers_served=all_done,
            global_svc=global_svc,
            progress=progress,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
            recharge_node_result=depot_nr,
            feasible_before_count=feasible_now_count,
            ok_dep_before=ok_dep_now,
        )
        if no_prog:
            allow_recharge = False
            pruning_reasons[depot] = _prune_reason(np_reason)

    allow_finish = finish_val.valid and finish_val.category in {"valid", "forced"}
    if not allow_finish and finish_val.category == "invalid":
        pruning_reasons[depot] = finish_val.reason

    pruning_mask[depot] = allow_recharge or allow_finish
    depot_prune = np.array([allow_recharge, allow_finish], dtype=bool)
    depot_prune_reasons = [
        "" if allow_recharge else pruning_reasons.get(depot, finish_val.reason),
        finish_val.reason if allow_finish else finish_val.reason,
    ]

    prune_charge: dict[int, np.ndarray] = {}
    prune_charge_reasons: dict[int, list[str]] = {}
    return pruning_mask, pruning_reasons, depot_prune, depot_prune_reasons, prune_charge, prune_charge_reasons


def combine_mask_layers(
    hard_mask: np.ndarray,
    pruning_mask: np.ndarray,
    hard_reasons: dict[int, str],
    pruning_reasons: dict[int, str],
) -> tuple[np.ndarray, dict[int, str]]:
    final = hard_mask & pruning_mask
    final_reasons: dict[int, str] = {}
    for idx in range(len(final)):
        if final[idx]:
            continue
        if not hard_mask[idx]:
            final_reasons[idx] = hard_reasons.get(idx, "hard_infeasible")
        else:
            final_reasons[idx] = pruning_reasons.get(idx, "pruned")
    return final, final_reasons
