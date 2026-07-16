"""EV-state-first decision analysis: one recharge search per EV, vectorized customers.

This replaces customer-first nested loops (for each customer × each EV × recharge
search) with:

    for each unfinished EV:
        vectorized direct customer feasibility
        one bounded recharge-label search
        vectorized customers from each label
        combine into this EV's serviceable-customer mask

Global serviceability is the OR of all EV serviceable masks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .charge_masks import safe_depot_return_from
from .ev_state import EVState
from .masks import (
    EPS,
    customer_feasible,
    depot_finish_feasible,
    depot_recharge_feasible,
    station_feasible,
)
from .recharge_search import (
    RechargeLabel,
    RechargeSearchConfig,
    bounded_recharge_reachability,
)


@dataclass
class RechargeNodeResult:
    """Aggregated recharge-label outcomes for one depot/station node."""

    node: int
    labels: list[RechargeLabel] = field(default_factory=list)
    basic_customer_mask: np.ndarray | None = None  # over customer_indices
    safe_customer_mask: np.ndarray | None = None
    enabled_customer_ids: list[int] = field(default_factory=list)
    enabled_customer_count: int = 0
    enables_depot_return: bool = False
    feasible: bool = False
    useful: bool = False
    useful_reason: str = ""


@dataclass
class EVDecisionAnalysis:
    """Reusable per-EV analysis for one (instance, state_version, ev)."""

    ev_id: int
    state_version: int

    direct_basic_customer_mask: np.ndarray  # over customer_indices
    direct_safe_customer_mask: np.ndarray
    direct_customer_reasons: dict[int, str] = field(default_factory=dict)

    recharge_labels: list[RechargeLabel] = field(default_factory=list)
    recharge_node_results: dict[int, RechargeNodeResult] = field(default_factory=dict)

    serviceable_customer_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    serviceability_mode_by_customer: dict[int, str] = field(default_factory=dict)

    safe_depot_return_direct: bool = False
    safe_depot_return_after_recharge: bool = False

    feasible_station_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    useful_station_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))

    feasible_depot_recharge: bool = False
    useful_depot_recharge: bool = False
    feasible_depot_finish: bool = False

    timing: dict[str, float] = field(default_factory=dict)
    hop_limit_hits: int = 0


def _customer_arrays(instance):
    instance.ensure_hot_arrays()
    cust = np.asarray(instance.customer_indices, dtype=np.int64)
    return (
        cust,
        instance.demand_array(),
        instance.ready_time_array(),
        instance.due_time_array(),
        instance.service_time_array(),
    )


def vectorized_basic_from_state(
    instance,
    *,
    node: int,
    current_time: float,
    battery: float,
    capacity: float,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """Cheap vectorized customer prefilter from a hypothetical EV state."""
    cust, demands, ready, due, _svc = _customer_arrays(instance)
    reject: dict[int, str] = {}
    if cust.size == 0:
        return np.zeros(0, dtype=bool), cust, reject

    travel_t = instance.travel_time_matrix[node, cust]
    travel_e = instance.energy_matrix[node, cust]
    arrival = current_time + travel_t
    service_start = np.maximum(arrival, ready[cust])

    served_b = served_mask[cust]
    reserved_b = reserved_mask[cust]
    cap_ok = demands[cust] <= capacity + EPS
    batt_ok = travel_e <= battery + EPS
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
    return basic, cust, reject


def vectorized_safe_direct_depot_after_service(
    instance,
    *,
    node: int,
    current_time: float,
    battery: float,
    capacity: float,
    basic_ok: np.ndarray,
    cust: np.ndarray,
    use_horizon_check: bool,
) -> np.ndarray:
    """Vectorized safe direct-depot continuation for basic-ok customers."""
    if cust.size == 0:
        return np.zeros(0, dtype=bool)
    depot = instance.depot_index
    demands = instance.demand_array()
    ready = instance.ready_time_array()
    svc = instance.service_time_array()
    travel = instance.travel_time_matrix
    energy = instance.energy_matrix
    due_dep = float(instance.nodes[depot].due_time)
    horizon = float(instance.planning_horizon)

    travel_t = travel[node, cust]
    travel_e = energy[node, cust]
    arrival = current_time + travel_t
    service_start = np.maximum(arrival, ready[cust])
    service_end = service_start + svc[cust]
    battery_after = battery - travel_e
    capacity_after = capacity - demands[cust]

    e_to_dep = energy[cust, depot]
    t_to_dep = travel[cust, depot]
    arr_dep = service_end + t_to_dep
    ok = (
        basic_ok
        & (battery_after >= -EPS)
        & (capacity_after >= -EPS)
        & (battery_after + EPS >= e_to_dep)
        & (arr_dep <= due_dep + EPS)
    )
    if use_horizon_check:
        ok = ok & (arr_dep <= horizon + EPS)
    return ok


def _exact_safe_for_survivors(
    instance,
    ev_sim: EVState,
    survivors: np.ndarray,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool,
    use_depot_charging: bool,
    charging_portions: list[float] | None,
) -> tuple[np.ndarray, dict[int, str]]:
    """Exact customer_feasible only for prefiltered survivors (node-index bool)."""
    n = instance.num_nodes
    safe = np.zeros(n, dtype=bool)
    reasons: dict[int, str] = {}
    for j in survivors:
        jj = int(j)
        ok, reason = customer_feasible(
            instance, ev_sim, jj, served_mask, reserved_mask,
            use_horizon_check, use_depot_charging, charging_portions,
        )
        safe[jj] = ok
        if not ok:
            reasons[jj] = reason
    return safe, reasons


def _annotate_label_customers(
    instance,
    label: RechargeLabel,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool,
    use_depot_charging: bool,
    charging_portions: list[float] | None,
) -> None:
    """Attach vectorized + exact customer masks onto a recharge label."""
    basic, cust, _ = vectorized_basic_from_state(
        instance,
        node=label.node,
        current_time=label.departure_time,
        battery=label.battery,
        capacity=ev.remaining_capacity,
        served_mask=served_mask,
        reserved_mask=reserved_mask,
    )
    label.basic_customer_mask = basic

    # Fast path: vectorized direct depot return after service.
    depot_ok = vectorized_safe_direct_depot_after_service(
        instance,
        node=label.node,
        current_time=label.departure_time,
        battery=label.battery,
        capacity=ev.remaining_capacity,
        basic_ok=basic,
        cust=cust,
        use_horizon_check=use_horizon_check,
    )
    safe_over_cust = depot_ok.copy()
    need_exact = cust[basic & ~depot_ok]
    sim = EVState(
        ev_id=ev.ev_id,
        current_node=label.node,
        current_time=label.departure_time,
        remaining_battery=label.battery,
        remaining_capacity=ev.remaining_capacity,
    )
    if need_exact.size:
        for j in need_exact:
            ok, _ = customer_feasible(
                instance, sim, int(j), served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, charging_portions,
            )
            # Map back into customer-index order.
            idx = int(np.where(cust == j)[0][0])
            safe_over_cust[idx] = ok

    label.safe_customer_mask = safe_over_cust
    enabled = cust[safe_over_cust]
    label.enabled_customer_ids = [int(x) for x in enabled.tolist()]
    label.enabled_customer_count = int(enabled.size)


def compute_ev_decision_analysis(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    state_version: int = 0,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    recharge_config: RechargeSearchConfig | None = None,
    include_station_masks: bool = True,
    force_direct_only: bool = False,
) -> EVDecisionAnalysis:
    """Compute one reusable EVDecisionAnalysis (exact EVRPTW semantics)."""
    t0 = time.perf_counter()
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    cfg = recharge_config or RechargeSearchConfig()
    cust, _d, _r, _du, _s = _customer_arrays(instance)
    n_cust = int(cust.size)
    stations = list(instance.charging_station_indices)
    depot = instance.depot_index

    timing: dict[str, float] = {}

    # --- Direct vectorized basic + exact safe continuation ---
    t_dir = time.perf_counter()
    direct_basic, cust, reject = vectorized_basic_from_state(
        instance,
        node=ev.current_node,
        current_time=ev.current_time,
        battery=ev.remaining_battery,
        capacity=ev.remaining_capacity,
        served_mask=served_mask,
        reserved_mask=reserved_mask,
    )
    timing["direct_customer_vectorized_ms"] = (time.perf_counter() - t_dir) * 1000.0

    t_safe = time.perf_counter()
    direct_depot_ok = vectorized_safe_direct_depot_after_service(
        instance,
        node=ev.current_node,
        current_time=ev.current_time,
        battery=ev.remaining_battery,
        capacity=ev.remaining_capacity,
        basic_ok=direct_basic,
        cust=cust,
        use_horizon_check=use_horizon_check,
    )
    direct_safe = direct_depot_ok.copy()
    direct_reasons = dict(reject)
    need_exact = cust[direct_basic & ~direct_depot_ok]
    if need_exact.size:
        for j in need_exact:
            ok, reason = customer_feasible(
                instance, ev, int(j), served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, portions,
            )
            idx = int(np.where(cust == j)[0][0])
            direct_safe[idx] = ok
            if not ok:
                direct_reasons[int(j)] = reason
            else:
                direct_reasons.pop(int(j), None)
    # Also mark basic survivors that passed depot_ok as feasible.
    for j in cust[direct_depot_ok]:
        direct_reasons.pop(int(j), None)
    timing["safe_continuation_ms"] = (time.perf_counter() - t_safe) * 1000.0

    mode_by_cust: dict[int, str] = {
        int(j): "direct" for j in cust[direct_safe]
    }

    # Fast path: if this EV can already serve every remaining customer directly,
    # skip recharge search (exact — recharge cannot disable a direct path).
    remaining_mask = (~served_mask[cust]) & (~reserved_mask[cust])
    skip_recharge = (
        force_direct_only
        or (not remaining_mask.any())
        or bool(np.all(direct_safe | ~remaining_mask))
    )

    if skip_recharge:
        timing["recharge_search_ms"] = 0.0
        timing["label_customer_vectorized_ms"] = 0.0
        ok_dep_now, _ = safe_depot_return_from(
            instance, ev.current_node, ev.current_time, ev.remaining_battery,
            use_horizon_check=use_horizon_check,
        )
        # Station/depot physical checks belong to the hard-mask layer for the
        # active EV; skip them here so GS across many EVs stays cheap.
        timing["total_ms"] = (time.perf_counter() - t0) * 1000.0
        return EVDecisionAnalysis(
            ev_id=int(ev.ev_id),
            state_version=int(state_version),
            direct_basic_customer_mask=direct_basic,
            direct_safe_customer_mask=direct_safe,
            direct_customer_reasons=direct_reasons,
            recharge_labels=[],
            recharge_node_results={},
            serviceable_customer_mask=direct_safe.copy(),
            serviceability_mode_by_customer=mode_by_cust,
            safe_depot_return_direct=bool(ok_dep_now),
            safe_depot_return_after_recharge=False,
            feasible_station_mask=np.zeros(len(stations), dtype=bool),
            useful_station_mask=np.zeros(len(stations), dtype=bool),
            feasible_depot_recharge=False,
            useful_depot_recharge=False,
            feasible_depot_finish=False,
            timing=timing,
            hop_limit_hits=0,
        )

    # --- One recharge-chain search per EV ---
    t_rech = time.perf_counter()
    reach = bounded_recharge_reachability(
        instance, ev,
        portions=portions,
        use_horizon_check=use_horizon_check,
        config=cfg,
    )
    timing["recharge_search_ms"] = (time.perf_counter() - t_rech) * 1000.0

    t_lab = time.perf_counter()
    all_labels: list[RechargeLabel] = []
    node_results: dict[int, RechargeNodeResult] = {}
    via_recharge = np.zeros(n_cust, dtype=bool)
    via_basic_any = np.zeros(n_cust, dtype=bool)
    # Phase A: cheap vectorized basic + direct-depot continuation per label.
    label_meta: list[tuple[RechargeLabel, np.ndarray, np.ndarray, bool]] = []
    for node, labs in reach.labels_by_node.items():
        if instance.nodes[node].is_depot and not use_depot_charging:
            continue
        for lab in labs:
            basic, cust_arr, _ = vectorized_basic_from_state(
                instance,
                node=lab.node,
                current_time=lab.departure_time,
                battery=lab.battery,
                capacity=ev.remaining_capacity,
                served_mask=served_mask,
                reserved_mask=reserved_mask,
            )
            depot_ok = vectorized_safe_direct_depot_after_service(
                instance,
                node=lab.node,
                current_time=lab.departure_time,
                battery=lab.battery,
                capacity=ev.remaining_capacity,
                basic_ok=basic,
                cust=cust_arr,
                use_horizon_check=use_horizon_check,
            )
            ok_dep, _ = safe_depot_return_from(
                instance, lab.node, lab.departure_time, lab.battery,
                use_horizon_check=use_horizon_check,
            )
            lab.basic_customer_mask = basic
            lab.safe_customer_mask = depot_ok.copy()  # may grow after exact phase
            lab.enables_depot_return = bool(ok_dep)
            via_basic_any |= basic
            via_recharge |= depot_ok
            all_labels.append(lab)
            label_meta.append((lab, basic, depot_ok, bool(ok_dep)))

    # Customers already serviceable via direct or vectorized depot-ok after recharge.
    serviceable = direct_safe | via_recharge

    # Phase B: exact safe-continuation only for customers that pass some label's
    # basic prefilter but are not yet proven serviceable (avoids O(labels×N) exact).
    need_exact_mask = via_basic_any & ~serviceable
    if need_exact_mask.any():
        for lab, basic, depot_ok, _ok_dep in label_meta:
            pending = basic & need_exact_mask & ~serviceable
            if not pending.any():
                continue
            sim = EVState(
                ev_id=ev.ev_id,
                current_node=lab.node,
                current_time=lab.departure_time,
                remaining_battery=lab.battery,
                remaining_capacity=ev.remaining_capacity,
            )
            safe_extra = np.zeros(n_cust, dtype=bool)
            for j in cust[pending]:
                ok, _ = customer_feasible(
                    instance, sim, int(j), served_mask, reserved_mask,
                    use_horizon_check, use_depot_charging, portions,
                )
                if ok:
                    jj = int(np.where(cust == j)[0][0])
                    safe_extra[jj] = True
                    serviceable[jj] = True
                    via_recharge[jj] = True
            lab.safe_customer_mask = depot_ok | safe_extra
            lab.enabled_customer_ids = [int(x) for x in cust[lab.safe_customer_mask].tolist()]
            lab.enabled_customer_count = int(lab.safe_customer_mask.sum())
            need_exact_mask = via_basic_any & ~serviceable
            if not need_exact_mask.any():
                break

    # Finalize per-label enabled counts for labels that never entered exact phase.
    for lab, basic, depot_ok, _ok_dep in label_meta:
        if lab.enabled_customer_ids is None:
            lab.safe_customer_mask = depot_ok
            lab.enabled_customer_ids = [int(x) for x in cust[depot_ok].tolist()]
            lab.enabled_customer_count = int(depot_ok.sum())

    # Aggregate per recharge node.
    for lab, basic, depot_ok, ok_dep in label_meta:
        node = int(lab.node)
        nr = node_results.get(node)
        if nr is None:
            nr = RechargeNodeResult(node=node)
            node_results[node] = nr
        nr.labels.append(lab)
        if nr.basic_customer_mask is None:
            nr.basic_customer_mask = basic.copy()
            nr.safe_customer_mask = np.asarray(lab.safe_customer_mask).copy()
        else:
            nr.basic_customer_mask |= basic
            nr.safe_customer_mask |= np.asarray(lab.safe_customer_mask)
        nr.enables_depot_return = nr.enables_depot_return or ok_dep
        nr.feasible = True

    for node, nr in node_results.items():
        enabled_ids = [int(x) for x in cust[nr.safe_customer_mask].tolist()]
        nr.enabled_customer_ids = enabled_ids
        nr.enabled_customer_count = len(enabled_ids)
        useful = bool(enabled_ids) or nr.enables_depot_return
        nr.useful = useful
        if enabled_ids:
            nr.useful_reason = (
                "depot_recharge_enables_service"
                if instance.nodes[node].is_depot
                else "station_unlocks_service"
            )
        elif nr.enables_depot_return:
            nr.useful_reason = (
                "depot_enables_return"
                if instance.nodes[node].is_depot
                else "station_enables_depot_return"
            )
        for j, hops in (
            (int(j), int(lab.hops))
            for lab in nr.labels
            for j in (cust[lab.safe_customer_mask] if lab.safe_customer_mask is not None else [])
        ):
            if j in mode_by_cust:
                continue
            if instance.nodes[node].is_depot:
                mode_by_cust[j] = "via_depot_recharge"
            elif hops <= 0:
                mode_by_cust[j] = "via_station"
            else:
                mode_by_cust[j] = "via_recharge_chain"

    timing["label_customer_vectorized_ms"] = (time.perf_counter() - t_lab) * 1000.0

    ok_dep_now, _ = safe_depot_return_from(
        instance, ev.current_node, ev.current_time, ev.remaining_battery,
        use_horizon_check=use_horizon_check,
    )
    safe_after_rech = any(nr.enables_depot_return for nr in node_results.values())

    # Station physical feasibility only when needed (active-EV hard masks).
    feas_st = np.zeros(len(stations), dtype=bool)
    use_st = np.zeros(len(stations), dtype=bool)
    ok_rech_phys = False
    useful_depot = False
    ok_finish_phys = False
    if include_station_masks:
        for si, s in enumerate(stations):
            ok_s, _ = station_feasible(
                instance, ev, s, served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, portions,
            )
            feas_st[si] = ok_s
            nr = node_results.get(int(s))
            use_st[si] = bool(ok_s and nr is not None and nr.useful)

        ok_rech_phys, _ = depot_recharge_feasible(
            instance, ev, served_mask, reserved_mask,
            use_depot_charging, use_horizon_check, portions,
        )
        depot_nr = node_results.get(int(depot))
        useful_depot = bool(
            ok_rech_phys and depot_nr is not None and depot_nr.enabled_customer_count > 0
        )
        ok_finish_phys, _ = depot_finish_feasible(instance, ev, use_horizon_check)
    else:
        depot_nr = node_results.get(int(depot))
        useful_depot = bool(depot_nr is not None and depot_nr.enabled_customer_count > 0)

    timing["total_ms"] = (time.perf_counter() - t0) * 1000.0
    return EVDecisionAnalysis(
        ev_id=int(ev.ev_id),
        state_version=int(state_version),
        direct_basic_customer_mask=direct_basic,
        direct_safe_customer_mask=direct_safe,
        direct_customer_reasons=direct_reasons,
        recharge_labels=all_labels,
        recharge_node_results=node_results,
        serviceable_customer_mask=serviceable,
        serviceability_mode_by_customer=mode_by_cust,
        safe_depot_return_direct=bool(ok_dep_now),
        safe_depot_return_after_recharge=bool(safe_after_rech),
        feasible_station_mask=feas_st,
        useful_station_mask=use_st,
        feasible_depot_recharge=bool(ok_rech_phys),
        useful_depot_recharge=useful_depot,
        feasible_depot_finish=bool(ok_finish_phys),
        timing=timing,
        hop_limit_hits=int(reach.hop_limit_hits),
    )


def get_or_compute_ev_analysis(
    env,
    ev: EVState,
    *,
    use_horizon_check: bool | None = None,
    use_depot_charging: bool | None = None,
    charging_portions: list[float] | None = None,
    include_station_masks: bool | None = None,
    force_direct_only: bool = False,
) -> EVDecisionAnalysis:
    """Cached EVDecisionAnalysis keyed by (instance_id, state_version, ev_id, flags)."""
    instance = env.instance
    state_version = int(getattr(env, "state_version", 0))
    instance_id = str(getattr(instance, "instance_name", id(instance)))
    active_id = getattr(env, "_analysis_active_ev_id", None)
    if include_station_masks is None:
        include_station_masks = (
            active_id is None or int(ev.ev_id) == int(active_id)
        )
    key = (
        instance_id,
        state_version,
        int(ev.ev_id),
        bool(include_station_masks),
        bool(force_direct_only),
    )
    cache = getattr(env, "_ev_analysis_cache", None)
    metrics = getattr(env, "_cache_metrics", None)
    if cache is None:
        cache = {}
        env._ev_analysis_cache = cache

    env_cfg = getattr(env, "environment_config", None) or {}
    cache_cfg = env_cfg.get("cache") or {}
    use_cache = bool(cache_cfg.get("enabled", True)) and bool(
        cache_cfg.get("cache_by_state_version", True)
    )
    if use_cache and key in cache:
        if metrics is not None:
            metrics["ev_analysis_hits"] = int(metrics.get("ev_analysis_hits", 0)) + 1
            metrics["hits"] = int(metrics.get("hits", 0)) + 1
        return cache[key]

    if metrics is not None:
        metrics["ev_analysis_misses"] = int(metrics.get("ev_analysis_misses", 0)) + 1
        metrics["misses"] = int(metrics.get("misses", 0)) + 1

    uh = env.use_horizon_check if use_horizon_check is None else use_horizon_check
    ud = env.use_depot_charging if use_depot_charging is None else use_depot_charging
    portions = charging_portions if charging_portions is not None else list(env.charging_portions)
    cfg = RechargeSearchConfig.from_env_config(env_cfg)
    analysis = compute_ev_decision_analysis(
        instance, ev, env.served_mask, env.reserved_mask,
        state_version=state_version,
        use_horizon_check=uh,
        use_depot_charging=ud,
        charging_portions=portions,
        recharge_config=cfg,
        include_station_masks=bool(include_station_masks),
        force_direct_only=bool(force_direct_only),
    )
    if use_cache:
        cache[key] = analysis
        if metrics is not None:
            metrics["size"] = len(cache) + len(getattr(env, "_decision_analysis_cache", {}) or {})
    return analysis


def compute_all_ev_analyses(env) -> dict[int, EVDecisionAnalysis]:
    """Direct-first across peer EVs; always fully analyze the active EV.

    Peer EVs use direct-only first for global coverage. Recharge search runs for
    peers only when coverage is incomplete. The active EV always gets a full
    analysis (needed for exact depot-finish / station usefulness).
    """
    unfinished = [ev for ev in env.evs if not ev.finished]
    active_id = getattr(env, "_analysis_active_ev_id", None)

    cust = np.asarray(env.instance.customer_indices, dtype=np.int64)
    remaining = (~env.served_mask[cust]) & (~env.reserved_mask[cust])
    out: dict[int, EVDecisionAnalysis] = {}

    # Active EV: always full analysis for exact decision masks.
    active_ev = None
    if active_id is not None:
        active_ev = next((e for e in unfinished if int(e.ev_id) == int(active_id)), None)
    if active_ev is not None:
        out[int(active_id)] = get_or_compute_ev_analysis(
            env, active_ev,
            include_station_masks=True,
            force_direct_only=False,
        )

    peers = [e for e in unfinished if active_ev is None or int(e.ev_id) != int(active_ev.ev_id)]
    covered = np.zeros(cust.size, dtype=bool)
    if active_id in out:
        covered |= np.asarray(out[int(active_id)].serviceable_customer_mask, dtype=bool)

    # Peer pass 1: direct-only.
    for ev in peers:
        ea = get_or_compute_ev_analysis(
            env, ev, include_station_masks=False, force_direct_only=True,
        )
        out[int(ev.ev_id)] = ea
        covered |= np.asarray(ea.serviceable_customer_mask, dtype=bool)

    if (not remaining.any()) or bool(np.all(covered | ~remaining)):
        return out

    # Peer pass 2: upgrade with recharge until coverage complete.
    covered = np.asarray(out[int(active_id)].serviceable_customer_mask, dtype=bool) if active_id in out else np.zeros(cust.size, dtype=bool)
    for ev in peers:
        ea = get_or_compute_ev_analysis(
            env, ev, include_station_masks=False, force_direct_only=False,
        )
        out[int(ev.ev_id)] = ea
        covered |= np.asarray(ea.serviceable_customer_mask, dtype=bool)
        if remaining.any() and bool(np.all(covered | ~remaining)):
            break
    return out
