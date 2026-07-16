"""Recharge-aware per-customer serviceability with TW expiration categories.

Categories (mutually refined — TW expiration is NOT a synonym for unserviceable):
* serviceable — direct or recharge-chain path exists for some unfinished EV
* time_window_expired — no unfinished EV can begin service within due time,
  even after a valid bounded recharge chain
* battery / capacity / no_safe_continuation — other structural unserviceability

Hot path uses EV-state-first analysis (one recharge search per EV, not per customer).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .charge_masks import feasible_charge_portions
from .ev_state import EVState
from .masks import EPS, customer_feasible
from .recharge_search import (
    RechargeSearchConfig,
    bounded_recharge_reachability,
    customer_reachable_via_recharge_chain,
)


@dataclass
class CustomerServiceabilityResult:
    serviceable: bool
    mode: str
    via_node: int | None
    charge_portion: float | None
    arrival_time: float | None
    service_start: float | None
    safe_continuation: bool
    reason: str
    category: str = "unknown"  # serviceable|time_window_expired|battery|capacity|no_safe_continuation|other


def _classify_reason(reason: str) -> str:
    if reason in {
        "customer_time_window_infeasible", "time_window_late", "horizon_exceeded",
        "time_window_expired",
    }:
        return "time_window_expired"
    if reason in {
        "customer_battery_unreachable", "battery_cannot_reach",
        "customer_infeasible_after_service_battery",
        "customer_no_reachable_station",
    }:
        return "battery"
    if reason in {"customer_capacity_infeasible", "capacity_exceeded"}:
        return "capacity"
    if reason in {
        "customer_no_safe_continuation", "customer_no_safe_direct_depot",
        "customer_station_has_no_feasible_charge",
        "customer_infeasible_no_safe_continuation",
        "customer_infeasible_no_reachable_station_or_depot",
        "no_recharge_chain",
    }:
        return "no_safe_continuation"
    return "other"


def _earliest_service_start_any_path(
    instance,
    ev: EVState,
    customer: int,
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool,
    use_depot_charging: bool,
    charging_portions: list[float] | None,
    config: RechargeSearchConfig | None,
) -> float | None:
    """Earliest possible service-start time via direct or recharge-chain path."""
    from .tw_diagnostics import customer_service_start

    earliest: float | None = None
    ok, _ = customer_feasible(
        instance, ev, customer, served_mask, reserved_mask,
        use_horizon_check, use_depot_charging, charging_portions,
    )
    if ok:
        _, _, ss, _ = customer_service_start(instance, ev, customer)
        earliest = float(ss)

    reach = bounded_recharge_reachability(
        instance, ev,
        portions=charging_portions,
        use_horizon_check=use_horizon_check,
        config=config,
    )
    due = float(instance.nodes[customer].due_time)
    for node, labs in reach.labels_by_node.items():
        if instance.nodes[node].is_depot and not use_depot_charging:
            continue
        for lab in labs:
            travel = float(instance.travel_time_matrix[lab.node, customer])
            energy = float(instance.energy_matrix[lab.node, customer])
            if lab.battery + EPS < energy:
                continue
            arrival = lab.departure_time + travel
            ready = float(instance.nodes[customer].ready_time)
            ss = max(arrival, ready)
            if ss > due + EPS:
                continue
            sim = EVState(
                ev_id=ev.ev_id,
                current_node=lab.node,
                current_time=lab.departure_time,
                remaining_battery=lab.battery,
                remaining_capacity=ev.remaining_capacity,
            )
            ok_c, _ = customer_feasible(
                instance, sim, customer, served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, charging_portions,
            )
            if ok_c and (earliest is None or ss < earliest):
                earliest = float(ss)
    return earliest


def compute_ev_customer_serviceability(
    instance,
    ev: EVState,
    customer: int,
    served_mask,
    reserved_mask,
    *,
    allow_station_recharge: bool = True,
    allow_depot_recharge: bool = True,
    skip_direct: bool = False,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    recharge_config: RechargeSearchConfig | None = None,
    ev_analysis=None,
) -> CustomerServiceabilityResult:
    """Whether ``ev`` can serve ``customer`` via direct or recharge-chain path.

    When ``ev_analysis`` is provided, reuse its masks (no per-customer recharge search).
    """
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    cfg = recharge_config or RechargeSearchConfig()
    if served_mask[customer] or reserved_mask[customer]:
        return CustomerServiceabilityResult(
            False, "unserviceable", None, None, None, None, False,
            "already_served", category="other",
        )

    # Fast path: reuse EV-state-first analysis.
    if ev_analysis is not None:
        cust = np.asarray(instance.customer_indices, dtype=np.int64)
        idx = int(np.where(cust == int(customer))[0][0]) if customer in set(cust.tolist()) else -1
        if idx >= 0 and bool(ev_analysis.serviceable_customer_mask[idx]):
            mode = ev_analysis.serviceability_mode_by_customer.get(int(customer), "direct")
            return CustomerServiceabilityResult(
                True, mode, None, None, None, None, True, mode, category="serviceable",
            )
        reason = ev_analysis.direct_customer_reasons.get(int(customer), "no_recharge_chain")
        return CustomerServiceabilityResult(
            False, "unserviceable", None, None, None, None, False, reason,
            category=_classify_reason(reason),
        )

    reason = "customer_infeasible"
    if not skip_direct:
        ok, reason = customer_feasible(
            instance, ev, customer, served_mask, reserved_mask,
            use_horizon_check, use_depot_charging, portions,
        )
        if ok:
            from .tw_diagnostics import customer_service_start
            _, _, ss, _ = customer_service_start(instance, ev, customer)
            return CustomerServiceabilityResult(
                True, "direct", customer, None,
                float(ev.current_time + instance.travel_time_matrix[ev.current_node, customer]),
                float(ss), True, "direct", category="serviceable",
            )

    # One-hop station / depot (fast path) then bounded multi-hop chain.
    if allow_station_recharge:
        from .masks import station_feasible
        for station in instance.charging_station_indices:
            ok_s, _ = station_feasible(
                instance, ev, station, served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, portions,
            )
            if not ok_s:
                continue
            cm, _ = feasible_charge_portions(
                instance, ev, station, served_mask, reserved_mask, portions,
                use_horizon_check=use_horizon_check,
                use_depot_charging=use_depot_charging,
            )
            if not cm.any():
                continue
            for idx, p in enumerate(portions):
                if not cm[idx]:
                    continue
                sim_ev = _simulate_charge(instance, ev, station, p)
                ok_c, _ = customer_feasible(
                    instance, sim_ev, customer, served_mask, reserved_mask,
                    use_horizon_check, use_depot_charging, portions,
                )
                if ok_c:
                    from .tw_diagnostics import customer_service_start
                    _, _, ss, _ = customer_service_start(instance, sim_ev, customer)
                    return CustomerServiceabilityResult(
                        True, "via_station", station, float(p),
                        float(sim_ev.current_time), float(ss), True, "via_station",
                        category="serviceable",
                    )

    if allow_depot_recharge and use_depot_charging:
        depot = instance.depot_index
        from .masks import depot_recharge_feasible
        ok_d, _ = depot_recharge_feasible(
            instance, ev, served_mask, reserved_mask,
            use_depot_charging, use_horizon_check, portions,
        )
        if ok_d:
            cm, _ = feasible_charge_portions(
                instance, ev, depot, served_mask, reserved_mask, portions,
                use_horizon_check=use_horizon_check,
                use_depot_charging=use_depot_charging,
            )
            for idx, p in enumerate(portions):
                if not cm[idx]:
                    continue
                sim_ev = _simulate_charge(instance, ev, depot, p)
                ok_c, _ = customer_feasible(
                    instance, sim_ev, customer, served_mask, reserved_mask,
                    use_horizon_check, use_depot_charging, portions,
                )
                if ok_c:
                    from .tw_diagnostics import customer_service_start
                    _, _, ss, _ = customer_service_start(instance, sim_ev, customer)
                    return CustomerServiceabilityResult(
                        True, "via_depot_recharge", depot, float(p),
                        float(sim_ev.current_time), float(ss), True, "via_depot_recharge",
                        category="serviceable",
                    )

    ok_chain, chain_reason, lab = customer_reachable_via_recharge_chain(
        instance, ev, customer, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=portions,
        config=cfg,
    )
    if ok_chain and lab is not None:
        return CustomerServiceabilityResult(
            True, "via_recharge_chain", lab.node, lab.charge_portion,
            lab.arrival_time, lab.departure_time, True, "via_recharge_chain",
            category="serviceable",
        )

    fail_reason = reason if reason != "customer_infeasible" else chain_reason
    return CustomerServiceabilityResult(
        False, "unserviceable", None, None, None, None, False, fail_reason,
        category=_classify_reason(fail_reason),
    )


def _simulate_charge(instance, ev: EVState, node: int, portion: float) -> EVState:
    from .charge_masks import simulate_recharge_departure
    free = instance.nodes[node].is_depot
    dep_t, batt, _, _ = simulate_recharge_departure(instance, ev, node, portion, free=free)
    return EVState(
        ev_id=ev.ev_id,
        current_node=node,
        current_time=dep_t,
        remaining_battery=batt,
        remaining_capacity=ev.remaining_capacity,
        route_distance=ev.route_distance,
        steps_since_last_customer_service=ev.steps_since_last_customer_service,
        num_customers_served=ev.num_customers_served,
    )


def _earliest_ss_from_ev_analysis(instance, ev_analysis, customer: int) -> float | None:
    """Cheapest earliest service-start estimate from stored EV analysis labels."""
    cust = np.asarray(instance.customer_indices, dtype=np.int64)
    if customer not in set(cust.tolist()):
        return None
    idx = int(np.where(cust == int(customer))[0][0])
    if not bool(ev_analysis.serviceable_customer_mask[idx]):
        return None
    ready = float(instance.nodes[customer].ready_time)
    earliest: float | None = None
    # Direct: use current EV would require the EV object; approximate via reasons.
    for lab in ev_analysis.recharge_labels:
        if lab.safe_customer_mask is None or not bool(lab.safe_customer_mask[idx]):
            continue
        travel = float(instance.travel_time_matrix[lab.node, customer])
        ss = max(lab.departure_time + travel, ready)
        if earliest is None or ss < earliest:
            earliest = ss
    return earliest


def _is_time_window_expired_from_analyses(
    instance,
    ev_analyses: dict,
    customer: int,
) -> bool:
    due = float(instance.nodes[customer].due_time)
    for ev_a in ev_analyses.values():
        cust = np.asarray(instance.customer_indices, dtype=np.int64)
        idx = int(np.where(cust == int(customer))[0][0])
        # If serviceable, clearly not TW-expired.
        if bool(ev_a.serviceable_customer_mask[idx]):
            return False
        # Check whether any label (or direct basic) can arrive within due ignoring safe-cont.
        if bool(ev_a.direct_basic_customer_mask[idx]):
            # Basic OK means TW is currently meetable from that EV.
            return False
        for lab in ev_a.recharge_labels:
            if lab.basic_customer_mask is None:
                continue
            if not bool(lab.basic_customer_mask[idx]):
                continue
            travel = float(instance.travel_time_matrix[lab.node, customer])
            ready = float(instance.nodes[customer].ready_time)
            ss = max(lab.departure_time + travel, ready)
            if ss <= due + EPS:
                return False
    return True


def _is_time_window_expired_global(
    instance,
    active_evs: list[EVState],
    customer: int,
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool,
    use_depot_charging: bool,
    charging_portions: list[float] | None,
    config: RechargeSearchConfig | None,
) -> bool:
    """TW expired iff no unfinished EV can begin service within due time on any path."""
    due = float(instance.nodes[customer].due_time)
    for ev in active_evs:
        ss = _earliest_service_start_any_path(
            instance, ev, customer, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
            config=config,
        )
        if ss is not None and ss <= due + EPS:
            return False
    return True


def compute_global_serviceability_recharge_aware_reference(
    instance,
    evs: list[EVState],
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    environment_config: dict | None = None,
):
    """Clarity-first customer×EV serviceability (slow reference for parity / benches)."""
    from .action_space import GlobalServiceability

    remaining = [j for j in instance.customer_indices if not served_mask[j]]
    gs = GlobalServiceability(remaining_customers_count=len(remaining))
    if not remaining:
        gs.no_service_possible = False
        return gs

    cfg = RechargeSearchConfig.from_env_config(environment_config)
    active_evs = [ev for ev in evs if not ev.finished]
    serviceable: set[int] = set()
    tw_expired: list[int] = []
    batt_ids: list[int] = []
    cap_ids: list[int] = []
    cont_ids: list[int] = []
    other_unsvc: list[int] = []
    recharge_pending: list[int] = []
    fail_category: dict[int, str] = {}

    for j in remaining:
        for ev in active_evs:
            ok, _ = customer_feasible(
                instance, ev, j, served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, charging_portions,
            )
            if ok:
                serviceable.add(j)
                break
        else:
            recharge_pending.append(j)

    for j in recharge_pending:
        best_cat = "other"
        for ev in active_evs:
            res = compute_ev_customer_serviceability(
                instance, ev, j, served_mask, reserved_mask,
                skip_direct=True,
                use_horizon_check=use_horizon_check,
                use_depot_charging=use_depot_charging,
                charging_portions=charging_portions,
                recharge_config=cfg,
            )
            if res.serviceable:
                serviceable.add(j)
                break
            best_cat = res.category
        else:
            fail_category[j] = best_cat

    for j in remaining:
        if j in serviceable:
            continue
        cat = fail_category.get(j, "other")
        demand = float(instance.nodes[j].demand)
        any_capacity = any(ev.remaining_capacity + EPS >= demand for ev in active_evs)
        if not any_capacity:
            cap_ids.append(j)
            continue
        if _is_time_window_expired_global(
            instance, active_evs, j, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
            config=cfg,
        ):
            tw_expired.append(j)
            continue
        if cat == "battery":
            batt_ids.append(j)
        elif cat == "no_safe_continuation":
            cont_ids.append(j)
        else:
            other_unsvc.append(j)

    unserviceable = sorted(set(remaining) - serviceable)
    gs.globally_serviceable_customers_count = len(serviceable)
    gs.serviceable_customer_ids = sorted(serviceable)
    gs.time_window_expired_customer_ids = sorted(set(tw_expired))
    gs.battery_unserviceable_customer_ids = sorted(set(batt_ids))
    gs.capacity_unserviceable_customer_ids = sorted(set(cap_ids))
    gs.no_safe_continuation_customer_ids = sorted(set(cont_ids))
    gs.globally_unserviceable_customer_ids = unserviceable
    gs.globally_expired_customers_count = len(gs.time_window_expired_customer_ids)
    gs.globally_unreachable_battery_count = len(gs.battery_unserviceable_customer_ids)
    gs.globally_unreachable_capacity_count = len(gs.capacity_unserviceable_customer_ids)
    gs.globally_no_safe_continuation_count = len(gs.no_safe_continuation_customer_ids)
    gs.globally_unserviceable_customers_count = len(unserviceable)
    gs.no_service_possible = len(serviceable) == 0
    return gs


def compute_global_serviceability_recharge_aware(
    instance,
    evs: list[EVState],
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    environment_config: dict | None = None,
    ev_analyses: dict | None = None,
    env=None,
):
    """Global serviceability via EV-state-first OR of per-EV serviceable masks.

    Prefer passing ``ev_analyses`` (or ``env`` for caching). Do not run recharge
    search once per customer.
    """
    from .action_space import GlobalServiceability
    from .ev_decision_analysis import (
        compute_ev_decision_analysis,
        get_or_compute_ev_analysis,
    )

    remaining = [j for j in instance.customer_indices if not served_mask[j]]
    gs = GlobalServiceability(remaining_customers_count=len(remaining))
    if not remaining:
        gs.no_service_possible = False
        return gs

    cfg = RechargeSearchConfig.from_env_config(environment_config)
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    active_evs = [ev for ev in evs if not ev.finished]
    cust = np.asarray(instance.customer_indices, dtype=np.int64)
    n_cust = int(cust.size)

    analyses: dict[int, object] = {}
    if ev_analyses is not None:
        analyses = {int(k): v for k, v in ev_analyses.items()}
    elif env is not None:
        for ev in active_evs:
            analyses[int(ev.ev_id)] = get_or_compute_ev_analysis(
                env, ev,
                use_horizon_check=use_horizon_check,
                use_depot_charging=use_depot_charging,
                charging_portions=portions,
            )
    else:
        state_version = 0
        for ev in active_evs:
            analyses[int(ev.ev_id)] = compute_ev_decision_analysis(
                instance, ev, served_mask, reserved_mask,
                state_version=state_version,
                use_horizon_check=use_horizon_check,
                use_depot_charging=use_depot_charging,
                charging_portions=portions,
                recharge_config=cfg,
            )

    global_mask = np.zeros(n_cust, dtype=bool)
    for ev_a in analyses.values():
        global_mask |= np.asarray(ev_a.serviceable_customer_mask, dtype=bool)

    serviceable = {int(j) for j in cust[global_mask]}

    tw_expired: list[int] = []
    batt_ids: list[int] = []
    cap_ids: list[int] = []
    cont_ids: list[int] = []
    other_unsvc: list[int] = []

    for j in remaining:
        if j in serviceable:
            continue
        demand = float(instance.nodes[j].demand)
        any_capacity = any(
            ev.remaining_capacity + EPS >= demand for ev in active_evs
        )
        if not any_capacity:
            cap_ids.append(j)
            continue

        # Prefer analysis-based TW expiration (no extra recharge searches).
        if analyses and _is_time_window_expired_from_analyses(instance, analyses, j):
            tw_expired.append(j)
            continue

        # Classify using per-EV direct reasons / modes.
        cats = []
        for ev_a in analyses.values():
            reason = ev_a.direct_customer_reasons.get(int(j), "no_recharge_chain")
            cats.append(_classify_reason(reason))
        if "battery" in cats and all(c in {"battery", "other", "no_safe_continuation"} for c in cats):
            if all(c == "battery" for c in cats):
                batt_ids.append(j)
                continue
        if all(c == "no_safe_continuation" for c in cats):
            cont_ids.append(j)
            continue
        if "battery" in cats:
            batt_ids.append(j)
        elif "no_safe_continuation" in cats:
            cont_ids.append(j)
        else:
            other_unsvc.append(j)

    unserviceable = sorted(set(remaining) - serviceable)

    gs.globally_serviceable_customers_count = len(serviceable)
    gs.serviceable_customer_ids = sorted(serviceable)
    gs.time_window_expired_customer_ids = sorted(set(tw_expired))
    gs.battery_unserviceable_customer_ids = sorted(set(batt_ids))
    gs.capacity_unserviceable_customer_ids = sorted(set(cap_ids))
    gs.no_safe_continuation_customer_ids = sorted(set(cont_ids))
    gs.globally_unserviceable_customer_ids = unserviceable
    gs.globally_expired_customers_count = len(gs.time_window_expired_customer_ids)
    gs.globally_unreachable_battery_count = len(gs.battery_unserviceable_customer_ids)
    gs.globally_unreachable_capacity_count = len(gs.capacity_unserviceable_customer_ids)
    gs.globally_no_safe_continuation_count = len(gs.no_safe_continuation_customer_ids)
    gs.globally_unserviceable_customers_count = len(unserviceable)
    gs.no_service_possible = len(serviceable) == 0
    # Attach analyses for downstream reuse when called without env.
    gs._ev_analyses = analyses  # type: ignore[attr-defined]
    return gs
