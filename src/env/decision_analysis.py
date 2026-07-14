"""Canonical once-per-decision-state analysis (reference + optimized).

Training / eval / dashboard MUST call ``analyze_decision_state`` and reuse the
returned ``DecisionStateAnalysis``. Low-level mask helpers may run inside the
analyzer only.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .action_masks import (
    ActionMaskResult,
    compute_action_masks_optimized_exact,
    compute_action_masks_reference,
)
from .ev_state import EVState
from .masks import EPS

# Process-wide counters for gate / performance reports.
_CALL_COUNTS: dict[str, int] = {
    "analyze_decision_state": 0,
    "analyze_decision_state_optimized": 0,
    "analyze_decision_state_reference": 0,
    "global_serviceability": 0,
    "dynamic_feature_builder": 0,
}


def get_analysis_call_counts() -> dict[str, int]:
    return dict(_CALL_COUNTS)


def reset_analysis_call_counts() -> None:
    for k in _CALL_COUNTS:
        _CALL_COUNTS[k] = 0


@dataclass
class PrefilterStats:
    total_customers: int = 0
    customers_rejected_by_vectorized_prefilter: int = 0
    customers_sent_to_continuation_analysis: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "total_customers": self.total_customers,
            "customers_rejected_by_vectorized_prefilter": self.customers_rejected_by_vectorized_prefilter,
            "customers_sent_to_continuation_analysis": self.customers_sent_to_continuation_analysis,
        }


@dataclass
class TimingBreakdown:
    basic_customer_filter_ms: float = 0.0
    safe_continuation_ms: float = 0.0
    recharge_simulation_ms: float = 0.0
    charging_chain_search_ms: float = 0.0
    global_serviceability_ms: float = 0.0
    hard_mask_ms: float = 0.0
    pruning_ms: float = 0.0
    dynamic_features_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "basic_customer_filter_ms": self.basic_customer_filter_ms,
            "safe_continuation_ms": self.safe_continuation_ms,
            "recharge_simulation_ms": self.recharge_simulation_ms,
            "charging_chain_search_ms": self.charging_chain_search_ms,
            "global_serviceability_ms": self.global_serviceability_ms,
            "hard_mask_ms": self.hard_mask_ms,
            "pruning_ms": self.pruning_ms,
            "dynamic_features_ms": self.dynamic_features_ms,
            "total_ms": self.total_ms,
        }


@dataclass
class DecisionStateAnalysis:
    """Single source of truth for one (env state_version, active EV) decision."""

    state_version: int
    active_ev_id: int

    hard_node_mask: np.ndarray
    pruning_node_mask: np.ndarray
    final_node_mask: np.ndarray
    node_reasons: dict[int, str]

    hard_charge_masks: dict[int, np.ndarray]
    pruning_charge_masks: dict[int, np.ndarray]
    final_charge_masks: dict[int, np.ndarray]

    depot_mode_mask: np.ndarray
    depot_recharge_reason: str = ""
    depot_finish_reason: str = ""

    direct_customer_analysis: Any = None
    customer_continuation_analysis: Any = None
    recharge_state_analysis: Any = None
    global_serviceability: Any = None

    dynamic_candidate_features: np.ndarray | None = None
    global_features: np.ndarray | None = None
    mask_summary_features: np.ndarray | None = None

    mask_result: ActionMaskResult | None = None
    mask_reason_counts: dict[str, int] = field(default_factory=dict)
    prefilter_stats: PrefilterStats = field(default_factory=PrefilterStats)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    cache_hit: bool = False
    implementation: str = "optimized"

    # Back-compat aliases used by earlier code.
    @property
    def hard_masks(self) -> np.ndarray:
        return self.hard_node_mask

    @property
    def pruning_masks(self) -> np.ndarray:
        return self.pruning_node_mask

    @property
    def final_masks(self) -> np.ndarray:
        return self.final_node_mask

    @property
    def charge_masks(self) -> dict[int, np.ndarray]:
        return self.final_charge_masks

    def assert_fresh(self, env, ev: EVState) -> None:
        if int(self.state_version) != int(getattr(env, "state_version", -1)):
            raise RuntimeError(
                f"stale analysis: analysis.state_version={self.state_version} "
                f"env.state_version={env.state_version}"
            )
        if int(self.active_ev_id) != int(ev.ev_id):
            raise RuntimeError(
                f"analysis EV mismatch: analysis.active_ev_id={self.active_ev_id} "
                f"ev.ev_id={ev.ev_id}"
            )


def vectorized_basic_customer_prefilter(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, str], PrefilterStats]:
    """Cheap vectorized customer prefilter (no safe-continuation yet).

    Returns:
        basic_ok: bool over ``customer_indices``
        customer_indices
        reject_reasons: node_id -> exact cheap reject reason
        stats
    """
    instance.ensure_hot_arrays()
    cust = np.asarray(instance.customer_indices, dtype=np.int64)
    stats = PrefilterStats(total_customers=int(cust.size))
    reject: dict[int, str] = {}
    if cust.size == 0:
        return np.zeros(0, dtype=bool), cust, reject, stats

    demands = instance.demand_array()
    ready = instance.ready_time_array()
    due = instance.due_time_array()
    r = ev.current_node
    travel_t = instance.travel_time_matrix[r, cust]
    travel_e = instance.energy_matrix[r, cust]
    arrival = ev.current_time + travel_t
    service_start = np.maximum(arrival, ready[cust])

    served_b = served_mask[cust]
    reserved_b = reserved_mask[cust]
    cap_ok = demands[cust] <= ev.remaining_capacity + EPS
    batt_ok = travel_e <= ev.remaining_battery + EPS
    tw_ok = service_start <= due[cust] + EPS

    basic = (~served_b) & (~reserved_b) & cap_ok & batt_ok & tw_ok
    for i, j in enumerate(cust):
        if basic[i]:
            continue
        jj = int(j)
        if served_b[i]:
            reject[jj] = "customer_already_served"
        elif reserved_b[i]:
            reject[jj] = "customer_reserved"
        elif not cap_ok[i]:
            reject[jj] = "customer_capacity_infeasible"
        elif not batt_ok[i]:
            reject[jj] = "customer_battery_unreachable"
        else:
            reject[jj] = "customer_time_window_infeasible"

    stats.customers_rejected_by_vectorized_prefilter = int((~basic).sum())
    stats.customers_sent_to_continuation_analysis = int(basic.sum())
    return basic, cust, reject, stats


def _build_analysis_from_mask(
    env,
    ev: EVState,
    mask: ActionMaskResult,
    *,
    implementation: str,
    timing: TimingBreakdown,
    prefilter_stats: PrefilterStats | None = None,
    dyn: np.ndarray | None = None,
) -> DecisionStateAnalysis:
    from src.models.dynamic_candidate import compute_dynamic_candidate_features
    from src.models.state_features import global_state_features, mask_summary_features

    t_dyn0 = time.perf_counter()
    if dyn is None:
        _CALL_COUNTS["dynamic_feature_builder"] += 1
        dyn = compute_dynamic_candidate_features(env, ev, mask)
    timing.dynamic_features_ms = (time.perf_counter() - t_dyn0) * 1000.0

    hard = mask.hard_node_mask if mask.hard_node_mask is not None else mask.node_mask
    prune = mask.pruning_node_mask if mask.pruning_node_mask is not None else mask.node_mask.copy()
    depot_reasons = list(mask.depot_mode_reasons or ["", ""])
    hard_charge = dict(getattr(env, "_hard_charge", {}) or {})
    if not hard_charge:
        hard_charge = dict(mask.charge_masks)

    analysis = DecisionStateAnalysis(
        state_version=int(getattr(env, "state_version", 0)),
        active_ev_id=int(ev.ev_id),
        hard_node_mask=hard.copy(),
        pruning_node_mask=prune.copy(),
        final_node_mask=mask.node_mask.copy(),
        node_reasons=dict(mask.node_reasons),
        hard_charge_masks=hard_charge,
        pruning_charge_masks={},
        final_charge_masks=dict(mask.charge_masks),
        depot_mode_mask=mask.depot_mode_mask.copy(),
        depot_recharge_reason=depot_reasons[0] if len(depot_reasons) > 0 else "",
        depot_finish_reason=depot_reasons[1] if len(depot_reasons) > 1 else "",
        global_serviceability=mask.global_serviceability,
        dynamic_candidate_features=dyn,
        global_features=np.asarray(global_state_features(env, mask), dtype=np.float32),
        mask_summary_features=np.asarray(mask_summary_features(env, mask), dtype=np.float32),
        mask_result=mask,
        mask_reason_counts=dict(mask.mask_reason_counts),
        prefilter_stats=prefilter_stats or PrefilterStats(),
        timing=timing,
        implementation=implementation,
    )
    return analysis


def analyze_decision_state_reference(env, ev: EVState) -> DecisionStateAnalysis:
    """Clarity-first reference analysis (exact semantics)."""
    _CALL_COUNTS["analyze_decision_state_reference"] += 1
    t0 = time.perf_counter()
    mask = compute_action_masks_reference(env, ev)
    timing = TimingBreakdown(
        hard_mask_ms=float(mask.computation_time_ms),
        total_ms=0.0,
    )
    analysis = _build_analysis_from_mask(
        env, ev, mask, implementation="reference", timing=timing,
    )
    analysis.timing.total_ms = (time.perf_counter() - t0) * 1000.0
    return analysis


def analyze_decision_state_optimized(env, ev: EVState) -> DecisionStateAnalysis:
    """Vectorized / cached analysis with identical exact semantics."""
    _CALL_COUNTS["analyze_decision_state_optimized"] += 1
    env.instance.ensure_hot_arrays()
    state_version = int(getattr(env, "state_version", 0))
    instance_id = str(getattr(env.instance, "instance_name", id(env.instance)))
    cache_key = (instance_id, state_version, int(ev.ev_id))
    cache = getattr(env, "_decision_analysis_cache", None)
    metrics = getattr(env, "_cache_metrics", None)
    if cache is None:
        cache = {}
        env._decision_analysis_cache = cache

    env_cfg = getattr(env, "environment_config", None) or {}
    cache_cfg = env_cfg.get("cache") or {}
    use_cache = bool(cache_cfg.get("enabled", True)) and bool(
        cache_cfg.get("cache_by_state_version", True)
    )
    if use_cache and cache_key in cache:
        hit = cache[cache_key]
        hit.cache_hit = True
        if metrics is not None:
            metrics["hits"] = int(metrics.get("hits", 0)) + 1
        return hit

    t0 = time.perf_counter()
    t_basic0 = time.perf_counter()
    basic, cust, reject, pref_stats = vectorized_basic_customer_prefilter(
        env.instance, ev, env.served_mask, env.reserved_mask,
    )
    basic_ms = (time.perf_counter() - t_basic0) * 1000.0
    # Store prefilter for hard-mask layer to consume (same decision scope).
    env._pending_customer_prefilter = {
        "basic_ok": basic,
        "customer_indices": cust,
        "reject_reasons": reject,
        "stats": pref_stats,
    }

    t_mask0 = time.perf_counter()
    mask = compute_action_masks_optimized_exact(env, ev)
    mask_ms = (time.perf_counter() - t_mask0) * 1000.0
    env._pending_customer_prefilter = None

    timing = TimingBreakdown(
        basic_customer_filter_ms=basic_ms,
        hard_mask_ms=mask_ms,
    )
    analysis = _build_analysis_from_mask(
        env, ev, mask,
        implementation="optimized",
        timing=timing,
        prefilter_stats=pref_stats,
    )
    analysis.timing.total_ms = (time.perf_counter() - t0) * 1000.0
    analysis.cache_hit = False

    if use_cache:
        cache[cache_key] = analysis
        if metrics is not None:
            metrics["misses"] = int(metrics.get("misses", 0)) + 1
            metrics["size"] = len(cache)
    return analysis


def analyze_decision_state(
    env,
    ev: EVState,
    *,
    implementation: str | None = None,
) -> DecisionStateAnalysis:
    """Official entry — used by training, eval, dashboard, baselines."""
    _CALL_COUNTS["analyze_decision_state"] += 1
    env_cfg = getattr(env, "environment_config", None) or {}
    impl = (implementation or str(env_cfg.get("implementation", "optimized"))).lower()
    if impl in {"reference", "ref", "clarity"}:
        analysis = analyze_decision_state_reference(env, ev)
    else:
        analysis = analyze_decision_state_optimized(env, ev)
    env.last_decision_analysis = analysis
    env.last_action_mask_result = analysis.mask_result
    env.last_global_serviceability = analysis.global_serviceability
    return analysis


def analyses_are_equivalent(
    a: DecisionStateAnalysis,
    b: DecisionStateAnalysis,
    *,
    feature_atol: float = 1e-5,
) -> tuple[bool, str]:
    from .action_masks import masks_are_equivalent

    if a.mask_result is not None and b.mask_result is not None:
        ok, msg = masks_are_equivalent(a.mask_result, b.mask_result)
        if not ok:
            return False, f"mask: {msg}"
    if not np.array_equal(a.final_node_mask, b.final_node_mask):
        return False, "final_node_mask differs"
    if not np.array_equal(a.depot_mode_mask, b.depot_mode_mask):
        return False, "depot_mode_mask differs"
    ga, gb = a.global_serviceability, b.global_serviceability
    if ga.globally_serviceable_customers_count != gb.globally_serviceable_customers_count:
        return False, "serviceability count differs"
    if set(getattr(ga, "serviceable_customer_ids", [])) != set(getattr(gb, "serviceable_customer_ids", [])):
        return False, "serviceable ids differ"
    if a.dynamic_candidate_features is not None and b.dynamic_candidate_features is not None:
        fa = a.dynamic_candidate_features.copy()
        fb = b.dynamic_candidate_features.copy()
        if fa.shape[1] > 16 and not (fa[:, 16].any() and fb[:, 16].any()):
            fa[:, 16] = 0.0
            fb[:, 16] = 0.0
        if not np.allclose(fa, fb, atol=feature_atol, equal_nan=True):
            diff = float(np.max(np.abs(fa - fb)))
            return False, f"dynamic features max_abs_diff={diff}"
    return True, "ok"
