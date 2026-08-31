"""Canonical exact EVRPTW action masks — single semantics for train/eval/replay."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .action_space import (
    GlobalServiceability,
    compute_full_feasible_actions,
    evaluate_charge_portions_for_node,
)
from .charge_masks import feasible_charge_portions
from .ev_state import EVState
from .masks import depot_mode_feasible

MASK_SEMANTICS_VERSION = "exact_original_evrptw"


def _perf_flags(env) -> dict:
    """Performance toggles — never change hard feasibility semantics."""
    raw = getattr(env, "environment_config", None) or {}
    perf = raw.get("performance") or {}
    return {
        "cache_action_masks": bool(perf.get("cache_action_masks", True)),
        "include_charge_portion_details": bool(perf.get("include_charge_portion_details", False)),
        "use_optimized_mask_path": bool(perf.get("use_optimized_mask_path", True)),
    }


def _charge_masks_from_hard_layer(
    instance,
    node_mask: np.ndarray,
    summary,
    hard_charge: dict[int, np.ndarray],
    hard_charge_reasons: dict[int, list[str]],
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    """Reuse hard-layer charge masks — identical to ``feasible_charge_portions`` output."""
    charge_masks: dict[int, np.ndarray] = {}
    charge_reasons: dict[int, list[str]] = {}
    for s in instance.charging_station_indices:
        if not node_mask[s] or s not in hard_charge:
            continue
        charge_masks[s] = hard_charge[s].copy()
        charge_reasons[s] = list(hard_charge_reasons.get(s, []))
    depot = instance.depot_index
    if (
        node_mask[depot]
        and summary.has_useful_depot_recharge
        and depot in hard_charge
    ):
        charge_masks[depot] = hard_charge[depot].copy()
        charge_reasons[depot] = list(hard_charge_reasons.get(depot, []))
    return charge_masks, charge_reasons


@dataclass
class ChargePortionResult:
    portion: float
    feasible: bool
    reason: str
    battery_after_charge: float
    departure_time: float
    enables_depot_return: bool
    enabled_customer_count: int
    enabled_customer_ids: list[int]
    best_return_margin: float
    usefulness_score: tuple = field(default_factory=tuple)


@dataclass
class ActionMaskResult:
    state_version: int
    node_mask: np.ndarray
    node_reasons: dict[int, str]
    charge_masks: dict[int, np.ndarray]
    charge_reasons: dict[int, list[str]]
    depot_mode_mask: np.ndarray
    depot_mode_reasons: list[str]
    feasible_customer_count: int
    feasible_station_count: int
    feasible_depot_recharge: bool
    feasible_depot_finish: bool
    global_serviceability: GlobalServiceability
    mask_reason_counts: dict[str, int]
    mask_semantics: str = MASK_SEMANTICS_VERSION
    computation_time_ms: float = 0.0
    charge_portion_details: dict[int, list[ChargePortionResult]] = field(default_factory=dict)
    hard_node_mask: np.ndarray | None = None
    hard_node_reasons: dict[int, str] = field(default_factory=dict)
    pruning_node_mask: np.ndarray | None = None
    pruning_node_reasons: dict[int, str] = field(default_factory=dict)
    hard_feasible_customer_count: int = 0
    hard_feasible_station_count: int = 0

    @property
    def feasible_mask(self) -> np.ndarray:
        return self.node_mask

    @property
    def final_node_mask(self) -> np.ndarray:
        return self.node_mask

    @property
    def final_node_reasons(self) -> dict[int, str]:
        return self.node_reasons


def _count_mask_reasons(
    instance,
    env_state,
    node_mask: np.ndarray,
    node_reasons: dict[int, str],
    depot_mode_mask: np.ndarray | None = None,
) -> dict[str, int]:
    counts: dict[str, int] = {
        "feasible_customers": 0,
        "feasible_stations": 0,
        "feasible_depot_recharge": 0,
        "feasible_depot_finish": 0,
        "masked_by_time_window": 0,
        "masked_by_battery": 0,
        "masked_by_capacity": 0,
        "masked_by_no_safe_continuation": 0,
        "masked_by_no_feasible_charge": 0,
        "masked_as_useless_station": 0,
        "masked_as_no_progress": 0,
        "masked_as_invalid_depot_finish": 0,
        "masked_by_already_served": 0,
        "masked_by_reservation": 0,
    }
    tw_reasons = {"customer_time_window_infeasible", "time_window_late", "horizon_exceeded"}
    batt_reasons = {
        "customer_battery_unreachable", "battery_cannot_reach", "battery_would_strand",
        "customer_infeasible_after_service_battery", "station_battery_unreachable",
        "station_unreachable_battery", "battery_cannot_reach_depot",
    }
    cap_reasons = {"customer_capacity_infeasible", "capacity_exceeded"}
    cont_reasons = {
        "customer_no_safe_continuation", "customer_no_safe_direct_depot",
        "customer_no_reachable_station", "customer_station_has_no_feasible_charge",
        "customer_infeasible_no_safe_continuation",
        "customer_infeasible_no_reachable_station_or_depot",
    }
    charge_reasons = {"station_no_feasible_charge_portion", "depot_no_feasible_charge_portion"}
    useless_station = {
        "station_no_useful_continuation", "station_not_useful",
        "depot_recharge_not_useful",
    }
    no_progress = {"station_no_progress", "no_progress_near_zero_loop", "no_progress_no_improvement"}
    invalid_finish = {"depot_finish_invalid", "early_depot_finish_blocked"}

    for j in instance.customer_indices:
        if node_mask[j]:
            counts["feasible_customers"] += 1
            continue
        r = node_reasons.get(j, "")
        if r in {"customer_already_served", "already_served"}:
            counts["masked_by_already_served"] += 1
        elif r in {"customer_reserved", "reserved_by_other"}:
            counts["masked_by_reservation"] += 1
        elif r in tw_reasons:
            counts["masked_by_time_window"] += 1
        elif r in batt_reasons:
            counts["masked_by_battery"] += 1
        elif r in cap_reasons:
            counts["masked_by_capacity"] += 1
        elif r in cont_reasons:
            counts["masked_by_no_safe_continuation"] += 1
        elif r in charge_reasons:
            counts["masked_by_no_feasible_charge"] += 1

    for s in instance.charging_station_indices:
        if node_mask[s]:
            counts["feasible_stations"] += 1
        else:
            r = node_reasons.get(s, "")
            if r in useless_station:
                counts["masked_as_useless_station"] += 1
            elif r in no_progress:
                counts["masked_as_no_progress"] += 1
            elif r in charge_reasons:
                counts["masked_by_no_feasible_charge"] += 1

    depot = instance.depot_index
    dep_reason = node_reasons.get(depot, "")
    if depot_mode_mask is not None and len(depot_mode_mask) >= 2:
        if depot_mode_mask[0]:
            counts["feasible_depot_recharge"] = 1
        if depot_mode_mask[1]:
            counts["feasible_depot_finish"] = 1
        if not depot_mode_mask[1] and dep_reason in invalid_finish:
            counts["masked_as_invalid_depot_finish"] = 1
    elif node_mask[depot]:
        if "recharge" in dep_reason:
            counts["feasible_depot_recharge"] = 1
        if "finish" in dep_reason:
            counts["feasible_depot_finish"] = 1
    elif dep_reason in invalid_finish:
        counts["masked_as_invalid_depot_finish"] = 1

    unserved = max(
        int(np.sum(~getattr(env_state, "served_mask", np.zeros(instance.num_nodes, dtype=bool))[instance.customer_indices])),
        1,
    )
    counts["feasible_customer_ratio_numerator"] = counts["feasible_customers"]
    counts["feasible_customer_ratio_denominator"] = unserved
    return counts


def _build_charge_details(
    instance,
    ev: EVState,
    env_state,
    node: int,
    portions: list[float],
    *,
    use_horizon_check: bool,
    use_depot_charging: bool,
) -> tuple[np.ndarray, list[str], list[ChargePortionResult]]:
    served = env_state.served_mask
    reserved = env_state.reserved_mask
    mask, reasons = feasible_charge_portions(
        instance, ev, node, served, reserved,
        portions=portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
    )
    analysis = evaluate_charge_portions_for_node(
        instance, ev, node, served, reserved,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=portions,
    )
    details: list[ChargePortionResult] = []
    from .charge_masks import simulate_recharge_departure, safe_depot_return_from

    free = instance.nodes[node].is_depot
    for idx, p in enumerate(portions):
        feasible = bool(mask[idx])
        if not feasible:
            details.append(ChargePortionResult(
                portion=float(p), feasible=False, reason=reasons[idx],
                battery_after_charge=0.0, departure_time=0.0,
                enables_depot_return=False, enabled_customer_count=0,
                enabled_customer_ids=[], best_return_margin=0.0,
            ))
            continue
        dep_t, batt, _, _ = simulate_recharge_departure(instance, ev, node, p, free=free)
        ok_dep, _ = safe_depot_return_from(instance, node, dep_t, batt, use_horizon_check=use_horizon_check)
        eval_match = next((e for e in analysis.all_feasible if abs(e.portion - p) < 1e-9), None)
        enabled = int(eval_match.enabled_customers) if eval_match else 0
        score = (
            enabled,
            int(ok_dep),
            -float(eval_match.charging_time) if eval_match else 0.0,
        )
        details.append(ChargePortionResult(
            portion=float(p), feasible=True, reason=reasons[idx],
            battery_after_charge=float(batt), departure_time=float(dep_t),
            enables_depot_return=bool(ok_dep), enabled_customer_count=enabled,
            enabled_customer_ids=[], best_return_margin=0.0,
            usefulness_score=score,
        ))
    return mask, reasons, details


def compute_action_masks_reference(env, ev: EVState) -> ActionMaskResult:
    """Reference exact mask — full usefulness, continuation, and charge-portion checks."""
    from .feasibility_scratch import feasibility_scratch_scope

    with feasibility_scratch_scope():
        return _compute_action_masks_reference_inner(env, ev)


def _compute_action_masks_reference_inner(env, ev: EVState) -> ActionMaskResult:
    t0 = time.perf_counter()
    inst = env.instance
    portions = list(getattr(env, "charging_portions", None) or [0.25, 0.5, 0.75, 1.0])
    state_version = int(getattr(env, "state_version", 0))

    node_mask, node_reasons, summary, global_svc = compute_full_feasible_actions(
        inst, env, ev,
        env.use_horizon_check, env.use_depot_charging, portions,
        global_svc=None,
    )

    charge_masks: dict[int, np.ndarray] = {}
    charge_reasons: dict[int, list[str]] = {}
    charge_details: dict[int, list[ChargePortionResult]] = {}

    for s in inst.charging_station_indices:
        if not node_mask[s]:
            continue
        cm, cr, cd = _build_charge_details(
            inst, ev, env, s, portions,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
        )
        charge_masks[s] = cm
        charge_reasons[s] = cr
        charge_details[s] = cd

    depot = inst.depot_index
    if node_mask[depot] and summary.has_useful_depot_recharge:
        cm, cr, cd = _build_charge_details(
            inst, ev, env, depot, portions,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
        )
        charge_masks[depot] = cm
        charge_reasons[depot] = cr
        charge_details[depot] = cd

    ok_recharge, ok_finish = depot_mode_feasible(
        inst, env, ev,
        use_horizon_check=env.use_horizon_check,
        use_depot_charging=env.use_depot_charging,
        charging_portions=portions,
    )
    depot_mode_mask = getattr(env, "_last_depot_mode_mask", np.array([False, False], dtype=bool))
    depot_mode_reasons = list(getattr(env, "_last_depot_mode_reasons", ["depot_recharge_masked", "depot_finish_masked"]))
    if depot_mode_mask is None or len(depot_mode_mask) < 2:
        depot_mode_mask = np.array([
            bool(summary.has_useful_depot_recharge and ok_recharge),
            bool(summary.has_depot_finish and ok_finish),
        ], dtype=bool)

    reason_counts = _count_mask_reasons(inst, env, node_mask, node_reasons, depot_mode_mask)
    hard_mask = getattr(env, "_last_hard_node_mask", node_mask)
    reason_counts["hard_feasible_actions"] = int(getattr(env, "_last_hard_counts", int(hard_mask.sum())))
    reason_counts["final_feasible_actions"] = int(node_mask.sum())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ActionMaskResult(
        state_version=state_version,
        node_mask=node_mask,
        node_reasons=node_reasons,
        charge_masks=charge_masks,
        charge_reasons=charge_reasons,
        depot_mode_mask=depot_mode_mask,
        depot_mode_reasons=depot_mode_reasons,
        feasible_customer_count=int(np.sum(node_mask[inst.customer_indices])),
        feasible_station_count=int(np.sum(node_mask[inst.charging_station_indices])),
        feasible_depot_recharge=bool(depot_mode_mask[0]),
        feasible_depot_finish=bool(depot_mode_mask[1]),
        global_serviceability=global_svc,
        mask_reason_counts=reason_counts,
        computation_time_ms=elapsed_ms,
        charge_portion_details=charge_details,
        hard_node_mask=hard_mask,
        hard_node_reasons=getattr(env, "_last_hard_reasons", {}),
        pruning_node_mask=getattr(env, "_last_pruning_mask", node_mask),
        pruning_node_reasons=getattr(env, "_last_prune_reasons", {}),
        hard_feasible_customer_count=int(np.sum(hard_mask[inst.customer_indices])),
        hard_feasible_station_count=int(np.sum(hard_mask[inst.charging_station_indices])),
    )


def compute_action_masks_optimized_exact(env, ev: EVState) -> ActionMaskResult:
    """Exact mask path — reuses hard-layer charge masks; skips redundant analysis."""
    from .feasibility_scratch import feasibility_scratch_scope

    with feasibility_scratch_scope():
        return _compute_action_masks_optimized_inner(env, ev)


def _compute_action_masks_optimized_inner(env, ev: EVState) -> ActionMaskResult:
    t0 = time.perf_counter()
    inst = env.instance
    portions = list(getattr(env, "charging_portions", None) or [0.25, 0.5, 0.75, 1.0])
    state_version = int(getattr(env, "state_version", 0))
    perf = _perf_flags(env)

    node_mask, node_reasons, summary, global_svc = compute_full_feasible_actions(
        inst, env, ev,
        env.use_horizon_check, env.use_depot_charging, portions,
        global_svc=None,
    )

    hard_charge = getattr(env, "_hard_charge", {})
    hard_charge_reasons = getattr(env, "_hard_charge_reasons", {})
    charge_masks, charge_reasons = _charge_masks_from_hard_layer(
        inst, node_mask, summary, hard_charge, hard_charge_reasons,
    )

    charge_details: dict[int, list[ChargePortionResult]] = {}
    if perf["include_charge_portion_details"]:
        for node in charge_masks:
            _, _, cd = _build_charge_details(
                inst, ev, env, node, portions,
                use_horizon_check=env.use_horizon_check,
                use_depot_charging=env.use_depot_charging,
            )
            charge_details[node] = cd

    depot_mode_mask = getattr(env, "_last_depot_mode_mask", None)
    depot_mode_reasons = list(getattr(env, "_last_depot_mode_reasons", ["depot_recharge_masked", "depot_finish_masked"]))
    if depot_mode_mask is None or len(depot_mode_mask) < 2:
        depot_mode_mask = np.array([
            bool(summary.has_useful_depot_recharge),
            bool(summary.has_depot_finish),
        ], dtype=bool)

    reason_counts = _count_mask_reasons(inst, env, node_mask, node_reasons, depot_mode_mask)
    hard_mask = getattr(env, "_last_hard_node_mask", node_mask)
    reason_counts["hard_feasible_actions"] = int(getattr(env, "_last_hard_counts", int(hard_mask.sum())))
    reason_counts["final_feasible_actions"] = int(node_mask.sum())
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return ActionMaskResult(
        state_version=state_version,
        node_mask=node_mask,
        node_reasons=node_reasons,
        charge_masks=charge_masks,
        charge_reasons=charge_reasons,
        depot_mode_mask=depot_mode_mask,
        depot_mode_reasons=depot_mode_reasons,
        feasible_customer_count=int(np.sum(node_mask[inst.customer_indices])),
        feasible_station_count=int(np.sum(node_mask[inst.charging_station_indices])),
        feasible_depot_recharge=bool(depot_mode_mask[0]),
        feasible_depot_finish=bool(depot_mode_mask[1]),
        global_serviceability=global_svc,
        mask_reason_counts=reason_counts,
        computation_time_ms=elapsed_ms,
        charge_portion_details=charge_details,
        hard_node_mask=hard_mask,
        hard_node_reasons=getattr(env, "_last_hard_reasons", {}),
        pruning_node_mask=getattr(env, "_last_pruning_mask", node_mask),
        pruning_node_reasons=getattr(env, "_last_prune_reasons", {}),
        hard_feasible_customer_count=int(np.sum(hard_mask[inst.customer_indices])),
        hard_feasible_station_count=int(np.sum(hard_mask[inst.charging_station_indices])),
    )


def compute_action_masks(env, ev: EVState) -> ActionMaskResult:
    """Canonical exact mask entry point for training, eval, and replay."""
    perf = _perf_flags(env)
    use_opt = perf["use_optimized_mask_path"]
    if perf["cache_action_masks"]:
        key = (int(getattr(env, "state_version", 0)), int(ev.ev_id))
        cache = getattr(env, "_action_mask_cache", None)
        if cache is None:
            cache = {}
            env._action_mask_cache = cache
        cached = cache.get(key)
        if cached is not None:
            return cached
        result = (
            compute_action_masks_optimized_exact(env, ev)
            if use_opt
            else compute_action_masks_reference(env, ev)
        )
        cache[key] = result
        return result
    if use_opt:
        return compute_action_masks_optimized_exact(env, ev)
    return compute_action_masks_reference(env, ev)


def masks_are_equivalent(a: ActionMaskResult, b: ActionMaskResult) -> tuple[bool, str]:
    """Return (equal, diff_message) for parity testing."""
    if not np.array_equal(a.node_mask, b.node_mask):
        diff = np.where(a.node_mask != b.node_mask)[0]
        return False, f"node_mask differs at indices {diff.tolist()}"
    if a.node_reasons != b.node_reasons:
        keys = set(a.node_reasons) | set(b.node_reasons)
        for k in keys:
            if a.node_reasons.get(k) != b.node_reasons.get(k):
                return False, f"node_reasons[{k}]: {a.node_reasons.get(k)} vs {b.node_reasons.get(k)}"
    if set(a.charge_masks.keys()) != set(b.charge_masks.keys()):
        return False, f"charge_masks keys differ: {set(a.charge_masks)} vs {set(b.charge_masks)}"
    for node in a.charge_masks:
        if not np.array_equal(a.charge_masks[node], b.charge_masks[node]):
            return False, f"charge_mask[{node}] differs"
    if not np.array_equal(a.depot_mode_mask, b.depot_mode_mask):
        return False, f"depot_mode_mask: {a.depot_mode_mask} vs {b.depot_mode_mask}"
    ga, gb = a.global_serviceability, b.global_serviceability
    if ga.globally_serviceable_customers_count != gb.globally_serviceable_customers_count:
        return False, (
            f"global_serviceability count: {ga.globally_serviceable_customers_count} "
            f"vs {gb.globally_serviceable_customers_count}"
        )
    return True, "ok"


def validate_action_against_mask(
    mask_result: ActionMaskResult,
    env,
    node: int,
    charge_portion: float | None,
    depot_finish: bool,
) -> None:
    """Fail-fast if action is inconsistent with the canonical mask."""
    from .mask_errors import InvalidActionMaskError, StaleMaskError

    if mask_result.state_version != getattr(env, "state_version", 0):
        raise StaleMaskError(
            f"mask state_version={mask_result.state_version} != env.state_version={env.state_version}"
        )
    if not mask_result.node_mask[node]:
        raise InvalidActionMaskError(
            f"node {node} not in node_mask; reason={mask_result.node_reasons.get(node, '?')}"
        )
    inst = env.instance
    if inst.nodes[node].is_charging_station or (inst.nodes[node].is_depot and not depot_finish):
        if node not in mask_result.charge_masks:
            raise InvalidActionMaskError(f"station/depot-recharge node {node} has no charge_masks entry")
        cm = mask_result.charge_masks[node]
        if not cm.any():
            raise InvalidActionMaskError(f"empty charge mask for node {node}")
        if charge_portion is None:
            raise InvalidActionMaskError(f"charge_portion required for node {node}")
        portions = list(getattr(env, "charging_portions", None) or [0.25, 0.5, 0.75, 1.0])
        try:
            idx = portions.index(float(charge_portion))
        except ValueError:
            idx = min(range(len(portions)), key=lambda i: abs(portions[i] - float(charge_portion)))
        if not cm[idx]:
            raise InvalidActionMaskError(
                f"charge portion {charge_portion} infeasible for node {node}"
            )
    if inst.nodes[node].is_depot:
        mode_idx = 1 if depot_finish else 0
        if not mask_result.depot_mode_mask[mode_idx]:
            raise InvalidActionMaskError(
                f"depot mode {'finish' if depot_finish else 'recharge'} not feasible"
            )
