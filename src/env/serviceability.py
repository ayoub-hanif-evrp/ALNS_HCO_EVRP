"""Recharge-aware per-customer serviceability with TW expiration categories.

Categories (mutually refined — TW expiration is NOT a synonym for unserviceable):
* serviceable — direct or recharge-chain path exists for some unfinished EV
* time_window_expired — no unfinished EV can begin service within due time,
  even after a valid bounded recharge chain
* battery / capacity / no_safe_continuation — other structural unserviceability
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
            # Cheap TW gate before full feasibility.
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
) -> CustomerServiceabilityResult:
    """Whether ``ev`` can serve ``customer`` via direct or recharge-chain path."""
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    cfg = recharge_config or RechargeSearchConfig()
    if served_mask[customer] or reserved_mask[customer]:
        return CustomerServiceabilityResult(
            False, "unserviceable", None, None, None, None, False,
            "already_served", category="other",
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
    # Also check capacity: if ANY EV has capacity for demand, capacity is not the blocker.
    # If no EV could reach within due regardless of capacity → TW expired.
    return True


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
):
    """Global serviceability using recharge-chain analysis + separate TW categories."""
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

    # Phase 1: cheap direct feasibility.
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

    # Phase 2: one-hop + multi-hop recharge chain.
    fail_category: dict[int, str] = {}
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

    # Phase 3: categorize remaining unserviceable — TW expired only if truly TW.
    for j in remaining:
        if j in serviceable:
            continue
        # Prefer capacity / battery classification from reasons; override with
        # explicit TW expiration test when that is the root cause.
        cat = fail_category.get(j, "other")
        demand = float(instance.nodes[j].demand)
        any_capacity = any(
            ev.remaining_capacity + EPS >= demand for ev in active_evs
        )
        if not any_capacity:
            cap_ids.append(j)
            continue

        # Structural TW expiration: cannot begin service within due on any path.
        if _is_time_window_expired_global(
            instance, active_evs, j, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
            config=cfg,
        ):
            # Confirm it is not solely a capacity issue already handled.
            tw_expired.append(j)
            continue

        if cat == "battery":
            batt_ids.append(j)
        elif cat == "no_safe_continuation":
            cont_ids.append(j)
        else:
            # Still unserviceable for other structural reasons.
            if cat == "capacity":
                cap_ids.append(j)
            else:
                other_unsvc.append(j)
                if cat == "battery":
                    batt_ids.append(j)

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
