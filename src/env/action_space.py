"""MDP action-space logic: usefulness filters, global serviceability, loop detection."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ev_state import EVState
from .masks import (
    EPS,
    customer_feasible,
    depot_finish_feasible,
    depot_recharge_feasible,
    station_feasible,
)

LOOP_WINDOW = 20
NEAR_ZERO_TRAVEL = 1e-4
BATTERY_PROGRESS_EPS = 0.5


@dataclass
class StationUsefulness:
    useful: bool
    reason: str
    enabled_customers_count: int = 0
    enables_depot_return: bool = False
    required_for_battery_safety: bool = False
    best_charge_portion: float | None = None
    selected_charge_portion_usefulness_score: float = 0.0
    enabled_customers_by_charge_portion: dict[float, int] = field(default_factory=dict)
    depot_return_feasible_by_charge_portion: dict[float, bool] = field(default_factory=dict)


@dataclass
class ChargePortionEvaluation:
    portion: float
    enabled_customers: int
    enables_depot_return: bool
    charging_time: float
    feasible: bool
    usefulness_score: float


@dataclass
class ChargePortionAnalysis:
    best: ChargePortionEvaluation | None
    all_feasible: list[ChargePortionEvaluation]

    def to_log_dict(self) -> dict:
        feasible_portions = [float(e.portion) for e in self.all_feasible]
        return {
            "feasible_charge_portions": feasible_portions,
            "best_useful_charge_portion": (
                float(self.best.portion) if self.best else None
            ),
            "best_charge_portion_for_station_usefulness": (
                float(self.best.portion) if self.best else None
            ),
            "enabled_customers_by_charge_portion": {
                float(e.portion): int(e.enabled_customers) for e in self.all_feasible
            },
            "safe_depot_return_by_charge_portion": {
                float(e.portion): bool(e.enables_depot_return) for e in self.all_feasible
            },
            "depot_return_feasible_by_charge_portion": {
                float(e.portion): bool(e.enables_depot_return) for e in self.all_feasible
            },
            "usefulness_score_by_charge_portion": {
                float(e.portion): float(e.usefulness_score) for e in self.all_feasible
            },
            "selected_charge_portion_usefulness_score": (
                float(self.best.usefulness_score) if self.best else 0.0
            ),
        }


@dataclass
class GlobalServiceability:
    remaining_customers_count: int = 0
    globally_serviceable_customers_count: int = 0
    globally_expired_customers_count: int = 0
    globally_unreachable_battery_count: int = 0
    globally_unreachable_capacity_count: int = 0
    globally_no_safe_continuation_count: int = 0
    globally_unserviceable_customers_count: int = 0
    no_service_possible: bool = False
    serviceable_customer_ids: list[int] = field(default_factory=list)
    time_window_expired_customer_ids: list[int] = field(default_factory=list)
    battery_unserviceable_customer_ids: list[int] = field(default_factory=list)
    capacity_unserviceable_customer_ids: list[int] = field(default_factory=list)
    no_safe_continuation_customer_ids: list[int] = field(default_factory=list)
    globally_unserviceable_customer_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, float | bool]:
        return {
            "remaining_customers_count": float(self.remaining_customers_count),
            "globally_serviceable_customers_count": float(self.globally_serviceable_customers_count),
            "globally_expired_customers_count": float(self.globally_expired_customers_count),
            "globally_unreachable_battery_count": float(self.globally_unreachable_battery_count),
            "globally_unreachable_capacity_count": float(self.globally_unreachable_capacity_count),
            "globally_no_safe_continuation_count": float(self.globally_no_safe_continuation_count),
            "globally_unserviceable_customers_count": float(self.globally_unserviceable_customers_count),
            "no_service_possible": bool(self.no_service_possible),
        }


@dataclass
class FeasibleActionSummary:
    has_customer: bool = False
    has_useful_station: bool = False
    has_useful_depot_recharge: bool = False
    has_depot_finish: bool = False
    has_any: bool = False

    @classmethod
    def from_mask(
        cls,
        instance,
        feasible: np.ndarray,
        reasons: dict[int, str],
        *,
        all_done: bool,
        depot_mode_mask: np.ndarray | None = None,
        depot_mode_reasons: list[str] | None = None,
    ) -> "FeasibleActionSummary":
        has_c = any(feasible[j] for j in instance.customer_indices)
        has_s = any(feasible[s] for s in instance.charging_station_indices)
        depot = instance.depot_index
        has_d = bool(feasible[depot])
        if depot_mode_mask is not None:
            has_recharge = bool(depot_mode_mask[0])
            has_finish = bool(depot_mode_mask[1])
        else:
            dep_reason = reasons.get(depot, "")
            has_recharge = has_d and "recharge" in dep_reason
            has_finish = has_d and "finish" in dep_reason
        if all_done and has_d and not has_finish:
            has_finish = True
        return cls(
            has_customer=has_c,
            has_useful_station=has_s,
            has_useful_depot_recharge=has_recharge,
            has_depot_finish=has_finish,
            has_any=bool(feasible.any()),
        )


@dataclass
class ActionRecord:
    ev_id: int
    action_type: str
    from_node: int
    to_node: int
    customer_served: bool = False
    battery_before: float = 0.0
    battery_after: float = 0.0
    time_before: float = 0.0
    time_after: float = 0.0
    enabled_customers: int = 0


@dataclass
class ActionProgressTracker:
    """Rolling action history for no-progress and loop detection."""

    loop_window: int = LOOP_WINDOW
    history: list[ActionRecord] = field(default_factory=list)
    steps_since_last_customer_service: int = 0
    noop_action_count: int = 0
    depot_station_loop_count: int = 0
    repeated_node_loop_count: int = 0
    no_progress_station_actions: int = 0
    no_progress_depot_recharge_actions: int = 0
    loop_detected: bool = False
    loop_pattern: str = ""
    loop_length: int = 0
    loop_actions: list[str] = field(default_factory=list)
    no_progress_segments: int = 0
    _in_no_progress_segment: bool = False
    _last_station_battery: dict[tuple[int, int], float] = field(default_factory=dict)

    def record(self, info: dict, ev: EVState) -> None:
        atype = str(info.get("type", "unknown"))
        customer_served = bool(info.get("customer_served", False))
        rec = ActionRecord(
            ev_id=int(info.get("ev_id", ev.ev_id)),
            action_type=atype,
            from_node=int(info.get("previous_node", ev.current_node)),
            to_node=int(info.get("node", ev.current_node)),
            customer_served=customer_served,
            battery_before=float(info.get("battery_before", ev.remaining_battery)),
            battery_after=float(info.get("battery_after", ev.remaining_battery)),
            time_before=float(info.get("time_before", ev.current_time)),
            time_after=float(ev.current_time),
            enabled_customers=int(info.get("enabled_customers", 0)),
        )
        self.history.append(rec)
        if len(self.history) > max(self.loop_window * 3, 60):
            self.history = self.history[-max(self.loop_window * 3, 60) :]

        battery_gain = rec.battery_after > rec.battery_before + BATTERY_PROGRESS_EPS
        is_progress = (
            customer_served
            or atype in ("depot_finish", "route_close")
            or info.get("valid_depot_finish")
            or info.get("useful_station")
            or info.get("useful_depot_recharge")
            or info.get("usefulness_before_action")
            or rec.enabled_customers > 0
            or info.get("ev_finished")
            # Fast training path may omit usefulness diagnostics; battery gain is a cheap proxy.
            or (atype in ("station", "depot_recharge") and battery_gain)
        )
        if customer_served or atype in ("depot_finish", "route_close"):
            self.steps_since_last_customer_service = 0
            if self._in_no_progress_segment:
                self.no_progress_segments += 1
                self._in_no_progress_segment = False
        elif (
            info.get("useful_station")
            or info.get("useful_depot_recharge")
            or info.get("usefulness_before_action")
            or (atype in ("station", "depot_recharge") and battery_gain)
        ):
            self.steps_since_last_customer_service = 0
            if self._in_no_progress_segment:
                self.no_progress_segments += 1
                self._in_no_progress_segment = False
        elif atype in ("station", "depot_recharge", "depot"):
            if not is_progress:
                self.noop_action_count += 1
                if atype in ("station",):
                    self.no_progress_station_actions += 1
                else:
                    self.no_progress_depot_recharge_actions += 1
                if not self._in_no_progress_segment:
                    self._in_no_progress_segment = True
            self.steps_since_last_customer_service += 1
        else:
            self.steps_since_last_customer_service += 1

        self._update_loop_flags(rec)
        if rec.action_type == "station":
            key = (rec.ev_id, rec.to_node)
            prev_b = self._last_station_battery.get(key, -1.0)
            if prev_b >= 0 and rec.battery_after <= prev_b + BATTERY_PROGRESS_EPS:
                self.repeated_node_loop_count += 1
            self._last_station_battery[key] = rec.battery_after

    def _update_loop_flags(self, rec: ActionRecord) -> None:
        recent = self.history[-self.loop_window :]
        if len(recent) < 4:
            return
        sigs = [(r.action_type, r.from_node, r.to_node) for r in recent]
        for plen in (2, 3, 4):
            if len(sigs) < plen * 2:
                continue
            tail = sigs[-plen * 2 :]
            if tail[:plen] == tail[plen:]:
                non_customer = all(s[0] in ("station", "depot_recharge", "depot") for s in tail[:plen])
                if non_customer:
                    self.loop_detected = True
                    self.loop_length = plen
                    self.loop_pattern = "->".join(f"{a}@{f}->{t}" for a, f, t in tail[:plen])
                    self.loop_actions = [f"{a}" for a, _, _ in tail[:plen]]
                    if any(a in ("depot_recharge", "depot") for a, _, _ in tail[:plen]):
                        self.depot_station_loop_count += 1
                    return

    def to_dict(self) -> dict[str, float | bool | str | list]:
        return {
            "steps_since_last_customer_service": float(self.steps_since_last_customer_service),
            "noop_action_count": float(self.noop_action_count),
            "depot_station_loop_count": float(self.depot_station_loop_count),
            "repeated_node_loop_count": float(self.repeated_node_loop_count),
            "no_progress_station_actions": float(self.no_progress_station_actions),
            "no_progress_depot_recharge_actions": float(self.no_progress_depot_recharge_actions),
            "loop_detected": bool(self.loop_detected),
            "loop_pattern": self.loop_pattern,
            "loop_length": float(self.loop_length),
            "no_progress_segments": float(self.no_progress_segments),
        }


def _feasible_customer_indices(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> list[int]:
    from .feasibility_scratch import ev_state_key, get_scratch
    from .masks import customer_feasible

    scratch = get_scratch()
    key = ev_state_key(ev)
    if scratch is not None:
        cached = scratch.feasible_indices.get(key)
        if cached is not None:
            return list(cached)

    out: list[int] = []
    for j in instance.customer_indices:
        ok, _ = customer_feasible(
            instance, ev, j, served_mask, reserved_mask,
            use_horizon_check, use_depot_charging, charging_portions,
        )
        if ok:
            out.append(j)
    if scratch is not None:
        scratch.feasible_indices[key] = list(out)
        scratch.feasible_sets[key] = frozenset(out)
        scratch.has_feasible[key] = bool(out)
    return out


def _has_feasible_customer(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> bool:
    """Early-exit feasibility probe with per-decision memoization."""
    from .feasibility_scratch import ev_state_key, get_scratch
    from .masks import customer_feasible

    scratch = get_scratch()
    key = ev_state_key(ev)
    if scratch is not None:
        if key in scratch.has_feasible:
            return scratch.has_feasible[key]
        cached = scratch.feasible_indices.get(key)
        if cached is not None:
            return bool(cached)

    for j in instance.customer_indices:
        ok, _ = customer_feasible(
            instance, ev, j, served_mask, reserved_mask,
            use_horizon_check, use_depot_charging, charging_portions,
        )
        if ok:
            if scratch is not None:
                scratch.has_feasible[key] = True
            return True
    if scratch is not None:
        scratch.has_feasible[key] = False
        scratch.feasible_indices[key] = []
        scratch.feasible_sets[key] = frozenset()
    return False


def _count_newly_enabled_customers(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    feasible_before: frozenset[int] | set[int],
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> int:
    """Count customers feasible after ``ev`` that were not feasible before."""
    from .masks import customer_feasible

    count = 0
    for j in instance.customer_indices:
        if j in feasible_before:
            continue
        ok, _ = customer_feasible(
            instance, ev, j, served_mask, reserved_mask,
            use_horizon_check, use_depot_charging, charging_portions,
        )
        if ok:
            count += 1
    return count


def evaluate_charge_portions_for_node(
    instance,
    ev: EVState,
    node: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    feasible_now_indices: list[int] | None = None,
    precomputed_charge_mask: np.ndarray | None = None,
) -> ChargePortionAnalysis:
    """Evaluate all feasible charge portions; pick best by usefulness score."""
    from .charge_masks import feasible_charge_portions, safe_depot_return_from, simulate_recharge_departure
    from .feasibility_scratch import ev_state_key, get_scratch

    scratch = get_scratch()
    cache_key = None
    if scratch is not None:
        cache_key = (*ev_state_key(ev), int(node))
        cached = scratch.charge_analysis.get(cache_key)
        if cached is not None:
            return cached

    p_list = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    if precomputed_charge_mask is not None:
        mask = precomputed_charge_mask
    else:
        mask, _ = feasible_charge_portions(
            instance, ev, node, served_mask, reserved_mask,
            portions=p_list,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
        )
    if feasible_now_indices is None:
        feasible_now_indices = _feasible_customer_indices(
            instance, ev, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
    feasible_now = len(feasible_now_indices)
    feasible_before = frozenset(feasible_now_indices)
    free = instance.nodes[node].is_depot
    evals: list[ChargePortionEvaluation] = []
    for idx, p in enumerate(p_list):
        if not mask[idx]:
            continue
        departure, batt, charge_time, _ = simulate_recharge_departure(
            instance, ev, node, p, free=free,
        )
        sim = EVState(
            ev_id=ev.ev_id,
            current_node=node,
            current_time=departure,
            remaining_battery=batt,
            remaining_capacity=ev.remaining_capacity,
        )
        enables = _count_newly_enabled_customers(
            instance, sim, served_mask, reserved_mask, feasible_before,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
        ok_dep, _ = safe_depot_return_from(
            instance, node, departure, batt, use_horizon_check=use_horizon_check,
        )
        safe_cont = bool(ok_dep or enables > 0)
        score = float(enables) * 1000.0 + float(safe_cont) * 10.0 - float(charge_time) * 0.01
        evals.append(ChargePortionEvaluation(
            portion=float(p),
            enabled_customers=int(enables),
            enables_depot_return=bool(ok_dep),
            charging_time=float(charge_time),
            feasible=True,
            usefulness_score=score,
        ))
    if not evals:
        result = ChargePortionAnalysis(best=None, all_feasible=[])
    else:
        best = max(
            evals,
            key=lambda e: (e.enabled_customers, int(e.enables_depot_return), -e.charging_time),
        )
        result = ChargePortionAnalysis(best=best, all_feasible=evals)
    if scratch is not None and cache_key is not None:
        scratch.charge_analysis[cache_key] = result
    return result


def _count_feasible_after_station_recharge(
    instance,
    ev: EVState,
    station: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    analysis: ChargePortionAnalysis | None = None,
    feasible_now_count: int | None = None,
) -> tuple[int, ChargePortionAnalysis]:
    """Return (total feasible customers after best charge, analysis)."""
    if analysis is None:
        analysis = evaluate_charge_portions_for_node(
            instance, ev, station, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
    if analysis.best is None:
        return 0, analysis
    if feasible_now_count is None:
        feasible_now_count = len(_feasible_customer_indices(
            instance, ev, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        ))
    return feasible_now_count + analysis.best.enabled_customers, analysis


def _simulate_best_charge_departure(
    instance,
    ev: EVState,
    station: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    portions: list[float] | None,
    use_horizon_check: bool,
    use_depot_charging: bool,
    analysis: ChargePortionAnalysis | None = None,
) -> tuple[float, float, ChargePortionAnalysis]:
    from .charge_masks import simulate_recharge_departure

    if analysis is None:
        analysis = evaluate_charge_portions_for_node(
            instance, ev, station, served_mask, reserved_mask,
            charging_portions=portions,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
        )
    if analysis.best is None:
        return ev.current_time, ev.remaining_battery, analysis
    dep_time, batt, _, _ = simulate_recharge_departure(
        instance, ev, station, analysis.best.portion,
        free=instance.nodes[station].is_depot,
    )
    return dep_time, batt, analysis


def compute_global_serviceability(
    instance,
    evs: list[EVState],
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> GlobalServiceability:
    """Whether any remaining customer can still be served by any active EV."""
    from .charge_masks import safe_depot_return_from

    remaining = [j for j in instance.customer_indices if not served_mask[j]]
    gs = GlobalServiceability(remaining_customers_count=len(remaining))
    if not remaining:
        gs.no_service_possible = False
        return gs

    serviceable: set[int] = set()
    for j in remaining:
        any_ok = False
        expired_only = True
        battery_block = True
        cap_block = True
        for ev in evs:
            if ev.finished:
                continue
            ok, reason = customer_feasible(
                instance, ev, j, served_mask, reserved_mask,
                use_horizon_check, use_depot_charging, charging_portions,
            )
            if ok:
                any_ok = True
                serviceable.add(j)
                break
            if reason != "customer_time_window_infeasible" and reason != "time_window_late":
                expired_only = False
            if reason not in (
                "customer_battery_unreachable", "battery_cannot_reach",
                "customer_infeasible_after_service_battery",
                "customer_no_safe_continuation", "customer_no_safe_direct_depot",
                "customer_no_reachable_station", "customer_station_has_no_feasible_charge",
                "customer_infeasible_no_safe_continuation",
                "customer_infeasible_no_reachable_station_or_depot",
            ):
                battery_block = False
            if reason != "customer_capacity_infeasible" and reason != "capacity_exceeded":
                cap_block = False
        if not any_ok:
            if expired_only:
                gs.globally_expired_customers_count += 1
            elif battery_block:
                gs.globally_unreachable_battery_count += 1
            elif cap_block:
                gs.globally_unreachable_capacity_count += 1

    gs.globally_serviceable_customers_count = len(serviceable)
    gs.no_service_possible = len(serviceable) == 0
    _ = safe_depot_return_from  # noqa: kept for future depot-return checks
    return gs


def compute_station_usefulness(
    instance,
    ev: EVState,
    station: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    all_customers_served: bool,
    global_svc: GlobalServiceability | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    feasible_now: list[int] | None = None,
    ok_dep_now: bool | None = None,
    precomputed_charge_mask: np.ndarray | None = None,
) -> StationUsefulness:
    """Selected-station-specific usefulness for mask / penalty / logging."""
    from .charge_masks import safe_depot_return_from

    if ok_dep_now is None:
        ok_dep_now, _ = safe_depot_return_from(
            instance, ev.current_node, ev.current_time, ev.remaining_battery,
            use_horizon_check=use_horizon_check,
        )
    if feasible_now is None:
        feasible_now = _feasible_customer_indices(
            instance, ev, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
    feasible_now_count = len(feasible_now)

    if all_customers_served:
        if ok_dep_now:
            return StationUsefulness(False, "station_unnecessary_direct_depot_return")
        analysis = evaluate_charge_portions_for_node(
            instance, ev, station, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
            feasible_now_indices=feasible_now,
            precomputed_charge_mask=precomputed_charge_mask,
        )
        dep_time, dep_batt, _ = _simulate_best_charge_departure(
            instance, ev, station, served_mask, reserved_mask,
            charging_portions, use_horizon_check, use_depot_charging,
            analysis=analysis,
        )
        ok_dep_after, _ = safe_depot_return_from(
            instance, station, dep_time, dep_batt, use_horizon_check=use_horizon_check,
        )
        if ok_dep_after:
            u = StationUsefulness(
                True, "station_required_for_depot_return", enables_depot_return=True,
            )
            u.best_charge_portion = analysis.best.portion if analysis.best else None
            return u
        return StationUsefulness(False, "station_cannot_enable_depot_return")

    if feasible_now and ok_dep_now:
        # Proactive charging is strategically valid. Do not spend O(stations ×
        # portions × customers) to prove "no new customers" — that analysis is
        # only required when the unsafe prune_station_when_customer_available
        # heuristic is explicitly enabled by the caller.
        return StationUsefulness(True, "station_proactive_allowed")

    analysis = evaluate_charge_portions_for_node(
        instance, ev, station, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
        feasible_now_indices=feasible_now,
        precomputed_charge_mask=precomputed_charge_mask,
    )
    if analysis.best is None:
        after_count = 0
    else:
        after_count = feasible_now_count + analysis.best.enabled_customers
    enables_customers = max(0, after_count - feasible_now_count)

    dep_time, dep_batt, _ = _simulate_best_charge_departure(
        instance, ev, station, served_mask, reserved_mask,
        charging_portions, use_horizon_check, use_depot_charging,
        analysis=analysis,
    )
    ok_dep_after, _ = safe_depot_return_from(
        instance, station, dep_time, dep_batt, use_horizon_check=use_horizon_check,
    )
    enables_depot_return = (not ok_dep_now) and ok_dep_after
    cp_log = analysis.to_log_dict()

    def _with_cp(u: StationUsefulness) -> StationUsefulness:
        u.best_charge_portion = analysis.best.portion if analysis.best else None
        u.selected_charge_portion_usefulness_score = cp_log["selected_charge_portion_usefulness_score"]
        u.enabled_customers_by_charge_portion = cp_log["enabled_customers_by_charge_portion"]
        u.depot_return_feasible_by_charge_portion = cp_log["depot_return_feasible_by_charge_portion"]
        return u

    if enables_customers > 0:
        return _with_cp(StationUsefulness(
            True, "station_unlocks_service", enabled_customers_count=enables_customers,
        ))
    if enables_depot_return:
        return _with_cp(StationUsefulness(True, "station_enables_depot_return", enables_depot_return=True))

    if global_svc and global_svc.no_service_possible and ok_dep_after:
        return _with_cp(StationUsefulness(
            True, "station_for_route_closure", enables_depot_return=True,
        ))

    if not ok_dep_now and ok_dep_after:
        return _with_cp(StationUsefulness(
            True, "station_battery_safety", required_for_battery_safety=True,
            enables_depot_return=True,
        ))

    return _with_cp(StationUsefulness(False, "station_not_useful"))


def compute_depot_recharge_usefulness(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    all_customers_served: bool,
    global_svc: GlobalServiceability | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    feasible_now_count: int | None = None,
    feasible_now_indices: list[int] | None = None,
    precomputed_charge_mask: np.ndarray | None = None,
) -> tuple[bool, str, int]:
    """Depot recharge is useful only if it enables future service."""
    if all_customers_served:
        return False, "depot_recharge_all_served", 0

    depot = instance.depot_index
    analysis = evaluate_charge_portions_for_node(
        instance, ev, depot, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
        feasible_now_indices=feasible_now_indices,
        precomputed_charge_mask=precomputed_charge_mask,
    )
    after_count, _ = _count_feasible_after_station_recharge(
        instance, ev, depot, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
        analysis=analysis,
        feasible_now_count=feasible_now_count,
    )
    if after_count > 0:
        return True, "depot_recharge_enables_service", after_count

    if global_svc and global_svc.no_service_possible:
        return False, "depot_recharge_no_global_service", 0

    return False, "depot_recharge_no_future_service", 0


@dataclass
class DepotFinishValidity:
    valid: bool
    reason: str
    category: str  # valid | early | forced | invalid
    useful_station_exists: bool = False
    depot_recharge_could_enable_service: bool = False
    finish_masked_due_to_future_service: bool = False
    finish_would_reduce_global_serviceability: bool = False


def exists_useful_recharge_action(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    global_svc: GlobalServiceability | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> bool:
    """True if useful depot recharge or station action exists for this EV."""
    can, _, _ = ev_future_service_possible(
        instance, ev, served_mask, reserved_mask,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    return can


def exists_future_service_after_recharge(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    global_svc: GlobalServiceability | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> bool:
    """True if recharge-mediated (depot or station) service becomes possible."""
    useful_r, _, enables_r = compute_depot_recharge_usefulness(
        instance, ev, served_mask, reserved_mask,
        all_customers_served=False,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    if useful_r and enables_r > 0:
        return True
    for s in instance.charging_station_indices:
        u = compute_station_usefulness(
            instance, ev, s, served_mask, reserved_mask,
            all_customers_served=False,
            global_svc=global_svc,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
        if u.useful and u.enabled_customers_count > 0:
            return True
    return False


def global_serviceability_after_finish(
    instance,
    evs: list[EVState],
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    finishing_ev: EVState,
    *,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> dict:
    """Global serviceability snapshot with finish-impact diagnostics."""
    gs = compute_global_serviceability(
        instance, evs, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    future_svc = exists_future_service_after_recharge(
        instance, finishing_ev, served_mask, reserved_mask,
        global_svc=gs,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    out = gs.to_dict()
    out["future_service_after_recharge_exists"] = bool(future_svc)
    out["finish_would_reduce_global_serviceability"] = bool(
        future_svc and gs.globally_serviceable_customers_count > 0
    )
    return out


def ev_future_service_possible(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    global_svc: GlobalServiceability | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, str, dict]:
    """True if this EV can still serve a remaining customer (recharge-aware)."""
    from .serviceability import compute_ev_customer_serviceability

    diag: dict = {
        "useful_station_exists_before_finish": False,
        "depot_recharge_could_enable_service": False,
    }
    for j in instance.customer_indices:
        if served_mask[j]:
            continue
        res = compute_ev_customer_serviceability(
            instance, ev, j, served_mask, reserved_mask,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
        if not res.serviceable:
            continue
        if res.mode == "direct":
            return True, "finish_invalid_customer_available", diag
        if res.mode == "via_station":
            diag["useful_station_exists_before_finish"] = True
            return True, "finish_invalid_station_unlocks_service", diag
        if res.mode == "via_depot_recharge":
            diag["depot_recharge_could_enable_service"] = True
            return True, "finish_invalid_depot_recharge_unlocks_service", diag
    return False, "", diag


def compute_depot_finish_validity(
    instance,
    ev: EVState,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    num_served_customers: int,
    *,
    global_svc: GlobalServiceability | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    forced: bool = False,
) -> DepotFinishValidity:
    ok, reason = depot_finish_feasible(instance, ev, use_horizon_check)
    if not ok:
        return DepotFinishValidity(False, reason, "invalid")

    all_done = num_served_customers >= instance.num_customers
    if all_done:
        return DepotFinishValidity(True, "finish_all_customers_served", "valid")

    if forced:
        return DepotFinishValidity(True, "finish_forced_safe_closure", "forced")

    can_service, svc_reason, svc_diag = ev_future_service_possible(
        instance, ev, served_mask, reserved_mask,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    )
    if can_service:
        gs_finish = global_serviceability_after_finish(
            instance, [ev], served_mask, reserved_mask, ev,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
        return DepotFinishValidity(
            False, svc_reason, "invalid",
            useful_station_exists=bool(svc_diag.get("useful_station_exists_before_finish")),
            depot_recharge_could_enable_service=bool(svc_diag.get("depot_recharge_could_enable_service")),
            finish_masked_due_to_future_service=True,
            finish_would_reduce_global_serviceability=bool(
                gs_finish.get("finish_would_reduce_global_serviceability"),
            ),
        )

    if global_svc and global_svc.no_service_possible:
        return DepotFinishValidity(True, "finish_global_no_service_possible", "valid")

    if exists_useful_recharge_action(
        instance, ev, served_mask, reserved_mask,
        global_svc=global_svc,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    ):
        return DepotFinishValidity(
            False, "finish_invalid_depot_recharge_unlocks_service", "invalid",
            finish_masked_due_to_future_service=True,
            finish_would_reduce_global_serviceability=True,
        )

    return DepotFinishValidity(True, "finish_ev_truly_exhausted", "valid")


def is_near_zero_move(instance, from_node: int, to_node: int) -> bool:
    t = float(instance.travel_time_matrix[from_node, to_node])
    d = float(instance.distance_matrix[from_node, to_node])
    return t <= NEAR_ZERO_TRAVEL and d <= NEAR_ZERO_TRAVEL


def would_be_no_progress_action(
    instance,
    ev: EVState,
    target_node: int,
    served_mask: np.ndarray,
    reserved_mask: np.ndarray,
    *,
    action_kind: str,
    depot_finish: bool,
    all_customers_served: bool,
    global_svc: GlobalServiceability | None,
    progress: ActionProgressTracker | None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
) -> tuple[bool, str]:
    """True if the action would not make meaningful progress."""
    if action_kind == "customer":
        return False, ""
    if action_kind == "depot_finish":
        return False, ""
    if depot_finish:
        return False, ""

    node = instance.nodes[target_node]
    if node.is_customer:
        return False, ""

    from .charge_masks import safe_depot_return_from

    ok_dep_before, _ = safe_depot_return_from(
        instance, ev.current_node, ev.current_time, ev.remaining_battery,
        use_horizon_check=use_horizon_check,
    )
    batt_before = ev.remaining_battery
    feasible_before = len(_feasible_customer_indices(
        instance, ev, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    ))

    if node.is_charging_station:
        usefulness = compute_station_usefulness(
            instance, ev, target_node, served_mask, reserved_mask,
            all_customers_served=all_customers_served,
            global_svc=global_svc,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
        if not usefulness.useful:
            return True, "no_progress_useless_station"
        dep_time, dep_batt, _ = _simulate_best_charge_departure(
            instance, ev, target_node, served_mask, reserved_mask,
            charging_portions, use_horizon_check, use_depot_charging,
        )
    elif node.is_depot:
        useful, reason, _ = compute_depot_recharge_usefulness(
            instance, ev, served_mask, reserved_mask,
            all_customers_served=all_customers_served,
            global_svc=global_svc,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=charging_portions,
        )
        if not useful:
            return True, reason
        dep_time, dep_batt, _ = _simulate_best_charge_departure(
            instance, ev, target_node, served_mask, reserved_mask,
            charging_portions, use_horizon_check, use_depot_charging,
        )
    else:
        return False, ""

    ok_dep_after, _ = safe_depot_return_from(
        instance, target_node, dep_time, dep_batt, use_horizon_check=use_horizon_check,
    )
    sim = EVState(
        ev_id=ev.ev_id,
        current_node=target_node,
        current_time=dep_time,
        remaining_battery=dep_batt,
        remaining_capacity=ev.remaining_capacity,
    )
    feasible_after = len(_feasible_customer_indices(
        instance, sim, served_mask, reserved_mask,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=charging_portions,
    ))
    batt_gain = dep_batt - batt_before
    depot_improved = (not ok_dep_before) and ok_dep_after
    customer_improved = feasible_after > feasible_before

    if customer_improved or depot_improved or all_customers_served:
        return False, ""

    if batt_gain <= BATTERY_PROGRESS_EPS and feasible_after <= feasible_before:
        near_zero = is_near_zero_move(instance, ev.current_node, target_node)
        if near_zero or (progress and progress.loop_detected):
            return True, "no_progress_near_zero_loop"
        return True, "no_progress_no_improvement"

    return False, ""


def compute_full_feasible_actions(
    instance,
    env_state,
    ev: EVState,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    charging_portions: list[float] | None = None,
    global_svc: GlobalServiceability | None = None,
) -> tuple[np.ndarray, dict[int, str], FeasibleActionSummary, GlobalServiceability]:
    """Layered exact mask: hard feasibility ∧ optional pruning."""
    from .mask_layers import (
        ActionPruningConfig,
        combine_mask_layers,
        compute_hard_feasibility_mask,
        compute_optional_pruning_mask,
    )
    from .serviceability import compute_global_serviceability_recharge_aware

    served = env_state.served_mask
    reserved = env_state.reserved_mask
    all_done = env_state.num_served_customers >= instance.num_customers
    portions = list(charging_portions or [0.25, 0.5, 0.75, 1.0])
    evs = getattr(env_state, "evs", [ev])

    pruning_cfg = ActionPruningConfig.from_env_config(getattr(env_state, "environment_config", None))
    hard, hard_reasons, hard_charge, hard_charge_reasons, hard_depot_mode, hard_depot_reasons = (
        compute_hard_feasibility_mask(
            instance, env_state, ev,
            use_horizon_check=use_horizon_check,
            use_depot_charging=use_depot_charging,
            charging_portions=portions,
        )
    )

    if global_svc is None:
        if pruning_cfg.enabled:
            if hasattr(env_state, "get_global_serviceability"):
                global_svc = env_state.get_global_serviceability()
            else:
                global_svc = compute_global_serviceability_recharge_aware(
                    instance, evs, served, reserved,
                    use_horizon_check=use_horizon_check,
                    use_depot_charging=use_depot_charging,
                    charging_portions=portions,
                    environment_config=getattr(env_state, "environment_config", None),
                )
        else:
            global_svc = GlobalServiceability(
                remaining_customers_count=int(np.sum(~served[instance.customer_indices])),
            )

    prune_layer, prune_reasons, prune_depot_mode, prune_depot_reasons, _, _ = compute_optional_pruning_mask(
        instance, env_state, ev, hard, hard_reasons,
        hard_depot_mode=hard_depot_mode,
        global_svc=global_svc,
        pruning=pruning_cfg,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        charging_portions=portions,
    )
    feasible, reasons = combine_mask_layers(hard, prune_layer, hard_reasons, prune_reasons)
    depot_mode_mask = hard_depot_mode & prune_depot_mode
    depot_mode_reasons = [
        hard_depot_reasons[0] if not depot_mode_mask[0] else "depot_recharge_feasible",
        hard_depot_reasons[1] if not depot_mode_mask[1] else hard_depot_reasons[1],
    ]

    env_state._hard_charge = hard_charge
    env_state._hard_charge_reasons = hard_charge_reasons
    summary = FeasibleActionSummary.from_mask(
        instance, feasible, reasons, all_done=all_done,
        depot_mode_mask=depot_mode_mask,
        depot_mode_reasons=depot_mode_reasons,
    )
    env_state._last_hard_node_mask = hard
    env_state._last_hard_reasons = hard_reasons
    env_state._last_pruning_mask = prune_layer
    env_state._last_prune_reasons = prune_reasons
    env_state._last_hard_counts = int(hard.sum())
    env_state._last_depot_mode_mask = depot_mode_mask
    env_state._last_depot_mode_reasons = depot_mode_reasons
    return feasible, reasons, summary, global_svc


def should_auto_close(
    summary: FeasibleActionSummary,
    *,
    global_svc: GlobalServiceability,
) -> bool:
    """Auto-close only when no meaningful policy action remains."""
    if summary.has_customer:
        return False
    if summary.has_useful_station:
        return False
    if summary.has_useful_depot_recharge:
        return False
    if summary.has_depot_finish:
        return False
    if summary.has_any:
        return False
    return True


def termination_diagnostics(env) -> dict:
    """Episode-end diagnostic bundle for logging / dashboard."""
    pt = getattr(env, "progress_tracker", None)
    gs = getattr(env, "last_global_serviceability", None)
    base = {
        "termination_reason": env.get_done_reason(),
        "max_steps_reached": bool(getattr(env, "max_steps_reached", False)),
        "success_all_served_all_returned": env.get_done_reason() == "all_served_returned",
        "terminal_no_service_possible": bool(gs.no_service_possible) if gs else False,
        "terminal_no_safe_return": bool(getattr(env, "terminal_no_safe_return", False)),
        "terminal_max_steps": env.get_done_reason() == "max_steps",
    }
    if pt:
        base.update(pt.to_dict())
    if gs:
        base.update({f"end_{k}": v for k, v in gs.to_dict().items()})
    if getattr(env, "max_steps_reached", False):
        hist = pt.history[-20:] if pt else []
        base["last_action_sequence"] = [
            {"type": r.action_type, "from": r.from_node, "to": r.to_node} for r in hist
        ]
        base["last_action_types"] = [r.action_type for r in hist]
        base["last_nodes"] = [r.to_node for r in hist]
        base["repeated_loop_detected"] = bool(pt.loop_detected) if pt else False
    return base
