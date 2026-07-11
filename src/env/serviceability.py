"""Recharge-aware per-customer serviceability analysis."""
from __future__ import annotations

from dataclasses import dataclass

from .masks import customer_feasible
from .charge_masks import feasible_charge_portions, safe_depot_return_from
from .ev_state import EVState


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
) -> CustomerServiceabilityResult:
    """Whether ``ev`` can serve ``customer`` via direct or recharge-mediated path."""
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    if served_mask[customer] or reserved_mask[customer]:
        return CustomerServiceabilityResult(False, "unserviceable", None, None, None, None, False, "already_served")

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
                float(ss), True, "direct",
            )

    if allow_station_recharge:
        from .masks import station_feasible
        for station in instance.charging_station_indices:
            ok_s, sr = station_feasible(
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
                sim_ev = _simulate_charge(instance, ev, station, p, portions)
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
                sim_ev = _simulate_charge(instance, ev, depot, p, portions)
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
                    )

    return CustomerServiceabilityResult(False, "unserviceable", None, None, None, None, False, reason)


def _simulate_charge(instance, ev: EVState, node: int, portion: float, portions: list[float]) -> EVState:
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
    )


def compute_global_serviceability_recharge_aware(
    instance,
    evs: list[EVState],
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
):
    """Global serviceability using recharge-aware per-customer analysis."""
    from .action_space import GlobalServiceability

    remaining = [j for j in instance.customer_indices if not served_mask[j]]
    gs = GlobalServiceability(remaining_customers_count=len(remaining))
    if not remaining:
        gs.no_service_possible = False
        return gs

    active_evs = [ev for ev in evs if not ev.finished]
    serviceable: set[int] = set()
    recharge_pending: list[int] = []

    # Phase 1: cheap direct feasibility before recharge simulation.
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

    # Phase 2: recharge-aware paths only for customers not directly reachable.
    for j in recharge_pending:
        for ev in active_evs:
            res = compute_ev_customer_serviceability(
                instance, ev, j, served_mask, reserved_mask,
                skip_direct=True,
                use_horizon_check=use_horizon_check,
                use_depot_charging=use_depot_charging,
                charging_portions=charging_portions,
            )
            if res.serviceable:
                serviceable.add(j)
                break

    gs.globally_serviceable_customers_count = len(serviceable)
    gs.globally_expired_customers_count = len(remaining) - len(serviceable)
    gs.serviceable_customer_ids = sorted(serviceable)
    gs.no_service_possible = len(serviceable) == 0
    return gs
