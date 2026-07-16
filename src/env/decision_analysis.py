"""Canonical once-per-decision-state analysis (reference + optimized).

Training / eval / dashboard MUST call ``analyze_decision_state`` and reuse the
returned ``DecisionStateAnalysis``. Low-level mask helpers may run inside the
analyzer only.

Optimized path is EV-state-first:
  for each unfinished EV → one recharge search + vectorized customers
  global_serviceability = OR of EV serviceable masks
  active EV analysis reused for hard/pruning/depot/station/features
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
    "recharge_search": 0,
    "ev_decision_analysis": 0,
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
    decision_analysis_ms: float = 0.0
    direct_customer_vectorized_ms: float = 0.0
    recharge_search_ms: float = 0.0
    label_customer_vectorized_ms: float = 0.0
    safe_continuation_ms: float = 0.0
    global_serviceability_ms: float = 0.0
    hard_mask_ms: float = 0.0
    pruning_ms: float = 0.0
    dynamic_feature_ms: float = 0.0
    # Legacy aliases
    basic_customer_filter_ms: float = 0.0
    recharge_simulation_ms: float = 0.0
    charging_chain_search_ms: float = 0.0
    dynamic_features_ms: float = 0.0
    total_ms: float = 0.0

    def to_dict(self) -> dict[str, float]:
        return {
            "decision_analysis_ms": self.decision_analysis_ms,
            "direct_customer_vectorized_ms": self.direct_customer_vectorized_ms,
            "recharge_search_ms": self.recharge_search_ms,
            "label_customer_vectorized_ms": self.label_customer_vectorized_ms,
            "safe_continuation_ms": self.safe_continuation_ms,
            "global_serviceability_ms": self.global_serviceability_ms,
            "hard_mask_ms": self.hard_mask_ms,
            "pruning_ms": self.pruning_ms,
            "dynamic_feature_ms": self.dynamic_feature_ms or self.dynamic_features_ms,
            "basic_customer_filter_ms": self.basic_customer_filter_ms,
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

    ev_analyses: dict[int, Any] = field(default_factory=dict)
    active_ev_analysis: Any = None

    dynamic_candidate_features: np.ndarray | None = None
    global_features: np.ndarray | None = None
    mask_summary_features: np.ndarray | None = None

    mask_result: ActionMaskResult | None = None
    mask_reason_counts: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)
    prefilter_stats: PrefilterStats = field(default_factory=PrefilterStats)
    timing: TimingBreakdown = field(default_factory=TimingBreakdown)
    cache_stats: dict[str, int] = field(default_factory=dict)
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
    """Cheap vectorized customer prefilter (no safe-continuation yet)."""
    from .ev_decision_analysis import vectorized_basic_from_state

    basic, cust, reject = vectorized_basic_from_state(
        instance,
        node=ev.current_node,
        current_time=ev.current_time,
        battery=ev.remaining_battery,
        capacity=ev.remaining_capacity,
        served_mask=served_mask,
        reserved_mask=reserved_mask,
    )
    stats = PrefilterStats(
        total_customers=int(cust.size),
        customers_rejected_by_vectorized_prefilter=int((~basic).sum()) if cust.size else 0,
        customers_sent_to_continuation_analysis=int(basic.sum()) if cust.size else 0,
    )
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
    ev_analyses: dict | None = None,
    active_ev_analysis: Any = None,
    cache_stats: dict | None = None,
) -> DecisionStateAnalysis:
    from src.models.dynamic_candidate import compute_dynamic_candidate_features
    from src.models.state_features import global_state_features, mask_summary_features

    t_dyn0 = time.perf_counter()
    if dyn is None:
        _CALL_COUNTS["dynamic_feature_builder"] += 1
        dyn = compute_dynamic_candidate_features(env, ev, mask)
    timing.dynamic_feature_ms = (time.perf_counter() - t_dyn0) * 1000.0
    timing.dynamic_features_ms = timing.dynamic_feature_ms

    hard = mask.hard_node_mask if mask.hard_node_mask is not None else mask.node_mask
    prune = mask.pruning_node_mask if mask.pruning_node_mask is not None else mask.node_mask.copy()
    depot_reasons = list(mask.depot_mode_reasons or ["", ""])
    hard_charge = dict(getattr(env, "_hard_charge", {}) or {})
    if not hard_charge:
        hard_charge = dict(mask.charge_masks)

    reason_counts = dict(mask.mask_reason_counts)
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
        ev_analyses=dict(ev_analyses or {}),
        active_ev_analysis=active_ev_analysis,
        dynamic_candidate_features=dyn,
        global_features=np.asarray(global_state_features(env, mask), dtype=np.float32),
        mask_summary_features=np.asarray(mask_summary_features(env, mask), dtype=np.float32),
        mask_result=mask,
        mask_reason_counts=reason_counts,
        reason_counts=reason_counts,
        prefilter_stats=prefilter_stats or PrefilterStats(),
        timing=timing,
        cache_stats=dict(cache_stats or getattr(env, "_cache_metrics", {}) or {}),
        implementation=implementation,
    )
    return analysis


def analyze_decision_state_reference(env, ev: EVState) -> DecisionStateAnalysis:
    """Clarity-first reference analysis (exact semantics, customer-first GS)."""
    from .serviceability import compute_global_serviceability_recharge_aware_reference
    from .analysis_counters import bump, enter_analyze, exit_analyze

    _CALL_COUNTS["analyze_decision_state_reference"] += 1
    t0 = time.perf_counter()
    enter_analyze()
    try:
        # Force customer-first global serviceability for true reference semantics/timing.
        env._global_serviceability_cache = None
        env.last_decision_analysis = None
        gs = compute_global_serviceability_recharge_aware_reference(
            env.instance, env.evs, env.served_mask, env.reserved_mask,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
            charging_portions=env.charging_portions,
            environment_config=getattr(env, "environment_config", None),
        )
        env._global_serviceability_cache = gs
        env.last_global_serviceability = gs
        if hasattr(env, "_sync_expired_from_serviceability"):
            env._sync_expired_from_serviceability(gs)
        _CALL_COUNTS["global_serviceability"] += 1
        bump("global_serviceability_calls")

        mask = compute_action_masks_reference(env, ev)
        mask.global_serviceability = gs
        timing = TimingBreakdown(
            hard_mask_ms=float(mask.computation_time_ms),
            global_serviceability_ms=(time.perf_counter() - t0) * 1000.0,
            total_ms=0.0,
        )
        analysis = _build_analysis_from_mask(
            env, ev, mask, implementation="reference", timing=timing,
        )
        analysis.timing.total_ms = (time.perf_counter() - t0) * 1000.0
        analysis.timing.decision_analysis_ms = analysis.timing.total_ms
        return analysis
    finally:
        exit_analyze()


def analyze_decision_state_optimized(env, ev: EVState) -> DecisionStateAnalysis:
    """EV-state-first vectorized / cached analysis with identical exact semantics."""
    _CALL_COUNTS["analyze_decision_state_optimized"] += 1
    env.instance.ensure_hot_arrays()
    state_version = int(getattr(env, "state_version", 0))
    instance_id = str(getattr(env.instance, "instance_name", id(env.instance)))
    cache_key = (instance_id, state_version, int(ev.ev_id))
    cache = getattr(env, "_decision_analysis_cache", None)
    metrics = getattr(env, "_cache_metrics", None)
    if metrics is None:
        metrics = {
            "hits": 0, "misses": 0, "invalidations": 0, "size": 0,
            "ev_analysis_hits": 0, "ev_analysis_misses": 0,
            "duplicate_computations_avoided": 0,
        }
        env._cache_metrics = metrics
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
        metrics["hits"] = int(metrics.get("hits", 0)) + 1
        metrics["duplicate_computations_avoided"] = (
            int(metrics.get("duplicate_computations_avoided", 0)) + 1
        )
        return hit

    t0 = time.perf_counter()
    from .analysis_counters import enter_analyze, exit_analyze
    enter_analyze()
    env._inside_analyze_decision = True
    try:
        return _analyze_decision_state_optimized_inner(
            env, ev, cache_key=cache_key, cache=cache, metrics=metrics,
            use_cache=use_cache, env_cfg=env_cfg, t0=t0,
        )
    finally:
        env._inside_analyze_decision = False
        exit_analyze()


def _analyze_decision_state_optimized_inner(
    env,
    ev: EVState,
    *,
    cache_key,
    cache,
    metrics,
    use_cache: bool,
    env_cfg: dict,
    t0: float,
) -> DecisionStateAnalysis:
    from .ev_decision_analysis import compute_all_ev_analyses, get_or_compute_ev_analysis
    from .serviceability import compute_global_serviceability_recharge_aware

    # 1) EV-state-first analyses for unfinished EVs (cached by state_version).
    t_ev0 = time.perf_counter()
    env._analysis_active_ev_id = int(ev.ev_id)
    ev_analyses = compute_all_ev_analyses(env)
    env._analysis_active_ev_id = None
    active_ev_analysis = ev_analyses.get(int(ev.ev_id))
    if active_ev_analysis is None:
        active_ev_analysis = get_or_compute_ev_analysis(env, ev)
        ev_analyses[int(ev.ev_id)] = active_ev_analysis
    _CALL_COUNTS["ev_decision_analysis"] += len(ev_analyses)
    n_recharge = sum(1 for a in ev_analyses.values() if a.recharge_labels)
    _CALL_COUNTS["recharge_search"] += n_recharge
    from .analysis_counters import bump
    bump("recharge_search_calls", n_recharge)
    ev_ms = (time.perf_counter() - t_ev0) * 1000.0

    # Aggregate timing from active EV analysis.
    at = active_ev_analysis.timing or {}
    pref_stats = PrefilterStats(
        total_customers=int(active_ev_analysis.direct_basic_customer_mask.size),
        customers_rejected_by_vectorized_prefilter=int(
            (~active_ev_analysis.direct_basic_customer_mask).sum()
        ),
        customers_sent_to_continuation_analysis=int(
            active_ev_analysis.direct_basic_customer_mask.sum()
        ),
    )

    # 2) Global serviceability once from OR of EV masks.
    t_gs0 = time.perf_counter()
    if getattr(env, "_global_serviceability_cache", None) is None:
        _CALL_COUNTS["global_serviceability"] += 1
        bump("global_serviceability_calls")
        gs = compute_global_serviceability_recharge_aware(
            env.instance, env.evs, env.served_mask, env.reserved_mask,
            use_horizon_check=env.use_horizon_check,
            use_depot_charging=env.use_depot_charging,
            charging_portions=env.charging_portions,
            environment_config=env_cfg,
            ev_analyses=ev_analyses,
            env=env,
        )
        env._global_serviceability_cache = gs
        env.last_global_serviceability = gs
        if hasattr(env, "_sync_expired_from_serviceability"):
            env._sync_expired_from_serviceability(gs)
        env._svc_computations_this_state = int(
            getattr(env, "_svc_computations_this_state", 0)
        ) + 1
    else:
        gs = env._global_serviceability_cache
        metrics["duplicate_computations_avoided"] = (
            int(metrics.get("duplicate_computations_avoided", 0)) + 1
        )
    gs_ms = (time.perf_counter() - t_gs0) * 1000.0

    # 3) Expose analyses to hard/pruning layers.
    env._pending_ev_analysis = active_ev_analysis
    env._pending_customer_prefilter = {
        "basic_ok": active_ev_analysis.direct_basic_customer_mask,
        "customer_indices": np.asarray(env.instance.customer_indices, dtype=np.int64),
        "reject_reasons": dict(active_ev_analysis.direct_customer_reasons),
        "stats": pref_stats,
    }

    t_mask0 = time.perf_counter()
    mask = compute_action_masks_optimized_exact(env, ev)
    mask_ms = (time.perf_counter() - t_mask0) * 1000.0
    # Ensure global svc on mask is the shared object.
    mask.global_serviceability = gs

    env._pending_customer_prefilter = None
    env._pending_ev_analysis = None

    timing = TimingBreakdown(
        decision_analysis_ms=0.0,
        direct_customer_vectorized_ms=float(at.get("direct_customer_vectorized_ms", 0.0)),
        recharge_search_ms=float(at.get("recharge_search_ms", 0.0)),
        label_customer_vectorized_ms=float(at.get("label_customer_vectorized_ms", 0.0)),
        safe_continuation_ms=float(at.get("safe_continuation_ms", 0.0)),
        global_serviceability_ms=gs_ms,
        hard_mask_ms=mask_ms,
        basic_customer_filter_ms=float(at.get("direct_customer_vectorized_ms", 0.0)),
        charging_chain_search_ms=float(at.get("recharge_search_ms", 0.0)),
        recharge_simulation_ms=ev_ms,
    )
    analysis = _build_analysis_from_mask(
        env, ev, mask,
        implementation="optimized",
        timing=timing,
        prefilter_stats=pref_stats,
        ev_analyses=ev_analyses,
        active_ev_analysis=active_ev_analysis,
        cache_stats=metrics,
    )
    analysis.timing.total_ms = (time.perf_counter() - t0) * 1000.0
    analysis.timing.decision_analysis_ms = analysis.timing.total_ms
    analysis.cache_hit = False

    if use_cache:
        cache[cache_key] = analysis
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
    from .analysis_counters import bump

    _CALL_COUNTS["analyze_decision_state"] += 1
    bump("analyze_decision_state_calls")
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
