"""Bounded label-setting charging-chain reachability over rechargeable nodes.

This is a computational search bound over depot + stations — not a dataset
modification. Edge u→v exists when the EV can travel after a feasible charge
at u and reach v with non-negative battery.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from heapq import heappop, heappush

import numpy as np

from .charge_masks import simulate_recharge_departure
from .ev_state import EVState
from .masks import EPS

LOGGER = logging.getLogger(__name__)

# Process-wide counters for hop-limit hits (exposed in timing / reports).
_HOP_LIMIT_HITS = 0


def hop_limit_hit_count() -> int:
    return _HOP_LIMIT_HITS


def reset_hop_limit_hits() -> None:
    global _HOP_LIMIT_HITS
    _HOP_LIMIT_HITS = 0


@dataclass(frozen=True)
class RechargeSearchConfig:
    max_station_hops: int = 3
    use_dominance_pruning: bool = True

    @classmethod
    def from_env_config(cls, cfg: dict | None) -> "RechargeSearchConfig":
        raw = (cfg or {}).get("recharge_search") or {}
        return cls(
            max_station_hops=int(raw.get("max_station_hops", 3)),
            use_dominance_pruning=bool(raw.get("use_dominance_pruning", True)),
        )


@dataclass(order=True)
class _HeapItem:
    departure_time: float
    neg_battery: float
    charge_time: float
    hops: int
    label_id: int


@dataclass
class RechargeLabel:
    node: int
    arrival_time: float
    departure_time: float
    battery: float
    cumulative_charging_time: float
    predecessor: int | None
    charge_portion: float | None
    hops: int = 0
    # Optional vectorized customer results attached by EV-state-first analysis.
    basic_customer_mask: object = None
    safe_customer_mask: object = None
    enabled_customer_ids: list | None = None
    enabled_customer_count: int = 0
    enables_depot_return: bool = False

    @property
    def battery_after_charge(self) -> float:
        return float(self.battery)


@dataclass
class RechargeReachabilityResult:
    """Labels reaching each rechargeable node, plus hop-limit diagnostics."""

    labels_by_node: dict[int, list[RechargeLabel]] = field(default_factory=dict)
    hop_limit_hits: int = 0
    expanded: int = 0


def _dominates(a: RechargeLabel, b: RechargeLabel) -> bool:
    """A dominates B if departure no later, battery no lower, hops no greater, charge time no greater."""
    weakly = (
        a.departure_time <= b.departure_time + EPS
        and a.battery + EPS >= b.battery
        and a.hops <= b.hops
        and a.cumulative_charging_time <= b.cumulative_charging_time + EPS
    )
    if not weakly:
        return False
    return (
        a.departure_time < b.departure_time - EPS
        or a.battery > b.battery + EPS
        or a.hops < b.hops
        or a.cumulative_charging_time < b.cumulative_charging_time - EPS
    )


def _is_dominated(label: RechargeLabel, existing: list[RechargeLabel]) -> bool:
    for other in existing:
        if (
            other.departure_time <= label.departure_time + EPS
            and other.battery + EPS >= label.battery
            and other.hops <= label.hops
            and other.cumulative_charging_time <= label.cumulative_charging_time + EPS
        ):
            return True
    return False


def _insert_nondominated(label: RechargeLabel, existing: list[RechargeLabel]) -> bool:
    if _is_dominated(label, existing):
        return False
    kept: list[RechargeLabel] = []
    changed = False
    for x in existing:
        if _dominates(label, x):
            changed = True
        else:
            kept.append(x)
    if changed:
        existing[:] = kept
    existing.append(label)
    return True


def _charging_rate(instance, node: int) -> float:
    if instance.station_charging_rate is not None:
        rate = float(instance.station_charging_rate[node])
        if rate > 1e-9 and not np.isnan(rate):
            return rate
    vp = instance.vehicle
    if vp.inverse_charging_rate > 1e-9:
        return 1.0 / vp.inverse_charging_rate
    return 1.0


def simulate_charge_at_node(
    instance,
    ev: EVState,
    node: int,
    portion: float,
) -> tuple[float, float, float]:
    """Return (departure_time, battery_after, charging_time)."""
    free = instance.nodes[node].is_depot
    departure, batt, charge_time, _ = simulate_recharge_departure(
        instance, ev, node, portion, free=free,
    )
    return departure, batt, charge_time


def rechargeable_nodes(instance) -> list[int]:
    nodes = list(getattr(instance, "rechargeable_indices", None) or [])
    if nodes:
        return nodes
    out = [instance.depot_index]
    out.extend(instance.charging_station_indices)
    return out


def bounded_recharge_reachability(
    instance,
    ev: EVState,
    *,
    portions: list[float] | None = None,
    use_horizon_check: bool = True,
    config: RechargeSearchConfig | None = None,
    start_as_already_charged: bool = False,
) -> RechargeReachabilityResult:
    """Label-setting search over rechargeable nodes with hop bound + dominance.

    If ``start_as_already_charged`` is False, the search begins by traveling from
    the EV's current node to each reachable rechargeable node and charging there
    (0 hops of further recharging after that). Subsequent station→station travels
    increment the hop count.
    """
    global _HOP_LIMIT_HITS
    cfg = config or RechargeSearchConfig()
    p_list = list(portions or [0.25, 0.5, 0.75, 1.0])
    nodes = rechargeable_nodes(instance)
    energy = instance.energy_matrix
    travel = instance.travel_time_matrix
    horizon = float(instance.planning_horizon)
    batt_cap = float(instance.vehicle.battery_capacity)

    result = RechargeReachabilityResult()
    labels: dict[int, list[RechargeLabel]] = {n: [] for n in nodes}
    heap: list[_HeapItem] = []
    label_store: list[RechargeLabel] = []
    live_ids: set[int] = set()
    label_obj_ids: dict[int, int] = {}  # id(label) -> lid
    next_id = 0

    def _push(label: RechargeLabel) -> None:
        nonlocal next_id
        node_labels = labels[label.node]
        if cfg.use_dominance_pruning:
            before = {id(x) for x in node_labels}
            if not _insert_nondominated(label, node_labels):
                return
            # Drop stale heap entries for labels removed by dominance.
            for rid in before - {id(x) for x in node_labels}:
                stale_lid = label_obj_ids.pop(rid, None)
                if stale_lid is not None:
                    live_ids.discard(stale_lid)
        else:
            node_labels.append(label)
        lid = next_id
        next_id += 1
        label_store.append(label)
        live_ids.add(lid)
        label_obj_ids[id(label)] = lid
        heappush(
            heap,
            _HeapItem(
                departure_time=label.departure_time,
                neg_battery=-label.battery,
                charge_time=label.cumulative_charging_time,
                hops=label.hops,
                label_id=lid,
            ),
        )

    # Seed: travel from current position to each rechargeable node + charge.
    for node in nodes:
        if node == ev.current_node and start_as_already_charged:
            seed = RechargeLabel(
                node=node,
                arrival_time=ev.current_time,
                departure_time=ev.current_time,
                battery=ev.remaining_battery,
                cumulative_charging_time=0.0,
                predecessor=None,
                charge_portion=None,
                hops=0,
            )
            _push(seed)
            continue
        e_to = float(energy[ev.current_node, node])
        if ev.remaining_battery + EPS < e_to:
            continue
        arrival = float(ev.current_time + travel[ev.current_node, node])
        if use_horizon_check and arrival > horizon + EPS:
            continue
        temp = EVState(
            ev_id=ev.ev_id,
            current_node=ev.current_node,
            current_time=ev.current_time,
            remaining_battery=ev.remaining_battery,
            remaining_capacity=ev.remaining_capacity,
        )
        for p in p_list:
            free = instance.nodes[node].is_depot
            departure, batt, charge_time, _ = simulate_recharge_departure(
                instance, temp, node, p, free=free,
            )
            if use_horizon_check and departure > horizon + EPS:
                continue
            if batt < -EPS or batt > batt_cap + EPS:
                continue
            _push(RechargeLabel(
                node=node,
                arrival_time=arrival,
                departure_time=float(departure),
                battery=float(batt),
                cumulative_charging_time=float(charge_time),
                predecessor=None,
                charge_portion=float(p),
                hops=0,
            ))

    while heap:
        item = heappop(heap)
        # O(1) stale check after dominance pruning (avoids O(#labels) list scan).
        if cfg.use_dominance_pruning and item.label_id not in live_ids:
            continue
        label = label_store[item.label_id]
        result.expanded += 1
        if label.hops >= cfg.max_station_hops:
            result.hop_limit_hits += 1
            _HOP_LIMIT_HITS += 1
            LOGGER.debug(
                "recharge_search hop limit (%d) reached at node=%s dep=%.3f batt=%.3f",
                cfg.max_station_hops, label.node, label.departure_time, label.battery,
            )
            continue
        # Reuse one temp EVState across portions for this expansion.
        temp = EVState(
            ev_id=ev.ev_id,
            current_node=label.node,
            current_time=label.departure_time,
            remaining_battery=label.battery,
            remaining_capacity=ev.remaining_capacity,
        )
        for nxt in nodes:
            if nxt == label.node:
                continue
            e_to = float(energy[label.node, nxt])
            if label.battery + EPS < e_to:
                continue
            arrival = float(label.departure_time + travel[label.node, nxt])
            if use_horizon_check and arrival > horizon + EPS:
                continue
            for p in p_list:
                free = instance.nodes[nxt].is_depot
                departure, batt, charge_time, _ = simulate_recharge_departure(
                    instance, temp, nxt, p, free=free,
                )
                if use_horizon_check and departure > horizon + EPS:
                    continue
                _push(RechargeLabel(
                    node=nxt,
                    arrival_time=arrival,
                    departure_time=float(departure),
                    battery=float(batt),
                    cumulative_charging_time=float(
                        label.cumulative_charging_time + charge_time
                    ),
                    predecessor=label.node,
                    charge_portion=float(p),
                    hops=label.hops + 1,
                ))

    result.labels_by_node = {n: list(ls) for n, ls in labels.items() if ls}
    return result


def customer_reachable_via_recharge_chain(
    instance,
    ev: EVState,
    customer: int,
    served_mask,
    reserved_mask,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    config: RechargeSearchConfig | None = None,
) -> tuple[bool, str, RechargeLabel | None]:
    """True if any recharge-chain label enables a feasible visit to ``customer``."""
    from .masks import customer_feasible

    if served_mask[customer] or reserved_mask[customer]:
        return False, "already_served_or_reserved", None

    reach = bounded_recharge_reachability(
        instance, ev,
        portions=charging_portions,
        use_horizon_check=use_horizon_check,
        config=config,
    )
    for node, labs in reach.labels_by_node.items():
        if instance.nodes[node].is_depot and not use_depot_charging:
            continue
        for lab in labs:
            sim = EVState(
                ev_id=ev.ev_id,
                current_node=lab.node,
                current_time=lab.departure_time,
                remaining_battery=lab.battery,
                remaining_capacity=ev.remaining_capacity,
            )
            ok, reason = customer_feasible(
                instance, sim, customer, served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, charging_portions,
            )
            if ok:
                return True, "via_recharge_chain", lab
    return False, "no_recharge_chain", None


def vectorized_customers_from_label(
    instance,
    label: RechargeLabel,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> np.ndarray:
    """Boolean mask over customer indices reachable via a simple prefilter + exact checks.

    Returns a length-``num_nodes`` bool array; only customer slots can be True.
    Uses cheap vectorized prefilter then exact ``customer_feasible`` for survivors.
    """
    from .masks import customer_feasible

    n = instance.num_nodes
    out = np.zeros(n, dtype=bool)
    cust = np.asarray(instance.customer_indices, dtype=np.int64)
    if cust.size == 0:
        return out

    demands = getattr(instance, "_demand_cache", None)
    ready = getattr(instance, "_ready_cache", None)
    due = getattr(instance, "_due_cache", None)
    svc = getattr(instance, "_service_cache", None)
    if demands is None:
        demands = instance.demand_array()
    if ready is None:
        ready = instance.ready_time_array()
    if due is None:
        due = instance.due_time_array()
    if svc is None:
        svc = instance.service_time_array()

    r = label.node
    travel_t = instance.travel_time_matrix[r, cust]
    energy = instance.energy_matrix[r, cust]
    arrival = label.departure_time + travel_t
    service_start = np.maximum(arrival, ready[cust])
    not_served = ~served_mask[cust]
    not_reserved = ~reserved_mask[cust]
    capacity_ok = demands[cust] <= ev.remaining_capacity + EPS
    battery_ok = energy <= label.battery + EPS
    tw_ok = service_start <= due[cust] + EPS
    basic = not_served & not_reserved & capacity_ok & battery_ok & tw_ok
    sim = EVState(
        ev_id=ev.ev_id,
        current_node=label.node,
        current_time=label.departure_time,
        remaining_battery=label.battery,
        remaining_capacity=ev.remaining_capacity,
    )
    for j in cust[basic]:
        ok, _ = customer_feasible(
            instance, sim, int(j), served_mask, reserved_mask,
            use_horizon_check, use_depot_charging, charging_portions,
        )
        out[int(j)] = ok
    return out
