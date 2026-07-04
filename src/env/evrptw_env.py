"""Asynchronous event-driven E-VRPTW environment.

Each EV keeps its own clock. EVs do not act in fixed global rounds: at every
decision event the EV with the smallest ``current_time`` acts next (ties broken
by route length, then ev_id). Because all EVs start at the depot at time 0 and
acting advances a clock, the first ``m`` decisions naturally form the sequential
initial dispatch described in the specification.

The environment never lets the policy apply an infeasible action: callers must
restrict choices to ``compute_feasible_mask``. The environment still tracks any
violation quantities defensively so the cost function can penalise them.
"""
from __future__ import annotations

import numpy as np

from .cost import CostWeights, SolutionSummary, compute_cost, is_feasible
from .ev_state import EVState
from .event_queue import select_next_ev
from .masks import compute_feasible_mask

EPS = 1e-6


class EVRPTWEnv:
    def __init__(
        self,
        instance,
        m: int,
        charging_portions: list[float] | None = None,
        use_horizon_check: bool = True,
        use_depot_charging: bool = True,
        max_steps_factor: int = 6,
        *,
        hard_time_windows: bool = True,
    ):
        self.instance = instance
        self.m = m
        self.charging_portions = charging_portions or [0.25, 0.50, 0.75, 1.0]
        self.use_horizon_check = use_horizon_check
        self.use_depot_charging = bool(use_depot_charging)
        self.hard_time_windows = bool(hard_time_windows)
        self.max_steps = max(50, max_steps_factor * max(instance.num_customers, 1))
        self.reset()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        inst = self.instance
        vp = inst.vehicle
        self.evs: list[EVState] = [
            EVState.initial(k, inst.depot_index, vp.battery_capacity, vp.load_capacity)
            for k in range(self.m)
        ]
        n = inst.num_nodes
        self.served_mask = np.zeros(n, dtype=bool)
        self.reserved_mask = np.zeros(n, dtype=bool)
        self.reserved_by = np.full(n, -1, dtype=np.int64)
        self.num_served_customers = 0
        self.current_event_time = 0.0
        self.total_distance = 0.0
        self.total_waiting_time = 0.0
        self.total_charging_time = 0.0
        self.total_charging_price = 0.0
        self.total_lateness = 0.0
        self.customer_lateness = 0.0
        self.depot_lateness = 0.0
        self.total_battery_violation = 0.0
        self.total_capacity_violation = 0.0
        self.depot_return_violation = 0
        self.duplicate_customer_visits = 0
        # Event counts (distinct from magnitude sums above).
        self.time_window_violation_count = 0
        self.customer_tw_violation_count = 0
        self.depot_tw_violation_count = 0
        self.battery_violation_count = 0
        self.customer_move_battery_violation = 0.0
        self.station_move_battery_violation = 0.0
        self.depot_return_battery_violation = 0.0
        self.auto_close_battery_violation = 0.0
        self.forced_auto_close_count = 0
        self.depot_recharge_count = 0
        self.depot_finish_count = 0
        self.capacity_violation_count = 0
        self.infeasible = False
        self.infeasibility_reasons: list[str] = []
        self.step_count = 0
        self.done_reason: str | None = None

    def get_done_reason(self) -> str:
        return self.done_reason or "in_progress"

    # ------------------------------------------------------------------ #
    # Event selection and feasibility
    # ------------------------------------------------------------------ #
    def select_acting_ev(self) -> EVState | None:
        return select_next_ev(self.evs)

    def feasible_mask(self, ev: EVState) -> tuple[np.ndarray, dict[int, str]]:
        return compute_feasible_mask(
            self.instance,
            self,
            ev,
            self.use_horizon_check,
            self.use_depot_charging,
            self.charging_portions,
        )

    def compute_mask_info(self, ev: EVState) -> tuple[np.ndarray, dict[int, str]]:
        """Pre-action feasible mask and per-node infeasibility reasons."""
        return self.feasible_mask(ev)

    def snapshot_decision_context(self, ev: EVState) -> dict:
        """Immutable decision context before ``apply_action``."""
        inst = self.instance
        return {
            "time_before": float(ev.current_time),
            "battery_before": float(ev.remaining_battery),
            "capacity_before": float(ev.remaining_capacity),
            "from_node": int(ev.current_node),
            "num_served": int(self.num_served_customers),
        }

    def compute_action_features(
        self,
        ev: EVState,
        selected_node: int,
        charge_portion: float | None = None,
        *,
        depot_finish: bool = False,
    ) -> dict:
        """Pre-action features for the selected node/mode."""
        from src.env.tw_diagnostics import customer_service_start

        inst = self.instance
        node = inst.nodes[selected_node]
        out: dict = {
            "to_node": int(selected_node),
            "selected_charge_portion": charge_portion,
            "depot_finish": bool(depot_finish),
        }
        if node.is_customer:
            arr, wait, ss, slack = customer_service_start(inst, ev, selected_node)
            due = float(node.due_time)
            e = float(inst.energy_matrix[ev.current_node, selected_node])
            out.update({
                "action_type": "customer",
                "arrival_time": arr,
                "service_start": ss,
                "due_time": due,
                "waiting_time": wait,
                "selected_customer_slack": slack,
                "battery_after_est": float(ev.remaining_battery - e),
            })
        elif node.is_charging_station:
            out["action_type"] = "station"
        elif node.is_depot:
            out["action_type"] = "depot_finish" if depot_finish else "depot_recharge"
        else:
            out["action_type"] = "unknown"
        return out

    def assert_pre_action_consistency(
        self,
        ev: EVState,
        selected_node: int,
        charge_portion: float | None,
        depot_finish: bool,
        feasible_mask: np.ndarray,
        mask_reasons: dict[int, str],
        *,
        global_epoch: int = 0,
    ) -> None:
        """Hard-mode checks that the selected action matches the feasible mask."""
        from src.env.charge_masks import feasible_charge_portions

        inst = self.instance
        if not feasible_mask[selected_node]:
            raise AssertionError(
                f"Selected infeasible node {selected_node}: {mask_reasons.get(selected_node, '?')}"
            )
        node = inst.nodes[selected_node]
        if node.is_customer:
            from src.env.tw_diagnostics import customer_service_start

            _, _, service_start, _ = customer_service_start(inst, ev, selected_node)
            due = float(node.due_time)
            assert service_start <= due + EPS, (
                f"customer TW: service_start={service_start} > due={due}"
            )
            e = float(inst.energy_matrix[ev.current_node, selected_node])
            battery_after = ev.remaining_battery - e
            assert battery_after >= -EPS, f"battery after customer would be {battery_after}"
        if node.is_charging_station or (node.is_depot and not depot_finish):
            mask, reasons = feasible_charge_portions(
                inst, ev, selected_node, self.served_mask, self.reserved_mask,
                portions=self.charging_portions,
                use_horizon_check=self.use_horizon_check,
                use_depot_charging=self.use_depot_charging,
            )
            if charge_portion is not None:
                idx = self.charging_portions.index(charge_portion)
                assert mask[idx], f"charge portion infeasible: {reasons[idx]}"
        _ = global_epoch

    def _record_battery_shortfall(self, ev: EVState, shortfall: float, category: str) -> None:
        if shortfall <= EPS:
            return
        self.total_battery_violation += shortfall
        self.battery_violation_count += 1
        if category == "customer":
            self.customer_move_battery_violation += shortfall
        elif category == "station":
            self.station_move_battery_violation += shortfall
        elif category == "auto_close":
            self.auto_close_battery_violation += shortfall
        else:
            self.depot_return_battery_violation += shortfall

    def _record_customer_lateness(
        self, ev: EVState, j: int, lateness: float, *, travel: float, arrival: float,
        ready: float, due: float, service_start: float,
    ) -> None:
        if lateness <= EPS:
            return
        self.customer_lateness += lateness
        self.total_lateness += lateness
        self.time_window_violation_count += 1
        self.customer_tw_violation_count += 1
        if self.hard_time_windows:
            self.infeasible = True
            self.infeasibility_reasons.append(
                f"hard_tw_violation ev={ev.ev_id} node={j} t={ev.current_time:.2f} "
                f"travel={travel:.2f} arrival={arrival:.2f} ready={ready:.2f} due={due:.2f} "
                f"service_start={service_start:.2f}"
            )

    def _record_depot_lateness(self, lateness: float) -> None:
        if lateness <= EPS:
            return
        self.depot_lateness += lateness
        self.total_lateness += lateness
        self.time_window_violation_count += 1
        self.depot_tw_violation_count += 1

    def should_close_route(self, ev: EVState) -> bool:
        """True when the full feasible mask has no selectable action."""
        mask, _ = self.feasible_mask(ev)
        return not mask.any()

    def all_customers_served(self) -> bool:
        return self.num_served_customers >= self.instance.num_customers

    def is_done(self) -> bool:
        if self.done_reason is not None:
            return True
        if self.step_count >= self.max_steps:
            self.done_reason = "max_steps"
            return True
        # Done when every EV is finished.
        if all(ev.finished for ev in self.evs):
            self.done_reason = "all_evs_finished"
            return True
        # Done when all customers are served and no EV still needs to move.
        if self.all_customers_served():
            # EVs away from depot still need to return; otherwise done.
            if all(ev.finished or ev.current_node == self.instance.depot_index for ev in self.evs):
                self.done_reason = "all_served_returned"
                return True
        return False

    # ------------------------------------------------------------------ #
    # Reservation
    # ------------------------------------------------------------------ #
    def reserve_customer(self, ev: EVState, j: int) -> None:
        """Reserve customer j for ev immediately upon selection."""
        self.reserved_mask[j] = True
        self.reserved_by[j] = ev.ev_id
        ev.current_reserved_customer = j

    def _release_reservation(self, j: int) -> None:
        self.reserved_mask[j] = False
        self.reserved_by[j] = -1

    # ------------------------------------------------------------------ #
    # Action application (physics)
    # ------------------------------------------------------------------ #
    def apply_action(
        self,
        ev: EVState,
        node: int,
        charge_portion: float | None = None,
        *,
        depot_finish: bool = False,
    ) -> dict:
        """Apply the EV's chosen next node and return an info dict.

        For customers, the reservation is taken (if not already) and resolved to
        a completed service within the same atomic event, because the EV clock
        advances to the service departure time.
        """
        inst = self.instance
        node_obj = inst.nodes[node]
        self.step_count += 1
        previous_node = ev.current_node
        d_move, t_move, e_move = self._move_energy_time(ev, node)

        if node_obj.is_customer:
            info = self._apply_customer(ev, node)
        elif node_obj.is_charging_station:
            info = self._apply_recharge(ev, node, charge_portion, free=False)
        elif node_obj.is_depot:
            if depot_finish:
                info = self.apply_depot_finish(ev)
                self.depot_finish_count += 1
            else:
                if not (self.use_depot_charging and inst.depot_has_charging):
                    raise ValueError("Depot recharge requires use_depot_charging and depot_has_charging")
                info = self._apply_recharge(ev, node, charge_portion, free=True)
                self.depot_recharge_count += 1
        else:  # pragma: no cover - node types are exhaustive
            raise ValueError(f"Unknown node type for node {node}")

        info["ev_id"] = ev.ev_id
        info["previous_node"] = previous_node
        info["delta_travel_time"] = t_move
        info["delta_energy_used"] = e_move
        info["customer_served"] = bool(info.get("type") == "customer" and info.get("node") == node)
        self.current_event_time = max(self.current_event_time, ev.current_time)
        return info

    def _move_energy_time(self, ev: EVState, target: int) -> tuple[float, float, float]:
        inst = self.instance
        d = float(inst.distance_matrix[ev.current_node, target])
        t = float(inst.travel_time_matrix[ev.current_node, target])
        e = float(inst.energy_matrix[ev.current_node, target])
        return d, t, e

    def _apply_customer(self, ev: EVState, j: int) -> dict:
        inst = self.instance
        node = inst.nodes[j]
        d, t, e = self._move_energy_time(ev, j)

        if self.served_mask[j]:
            self.duplicate_customer_visits += 1

        if self.reserved_by[j] not in (-1, ev.ev_id):
            # Selecting a customer reserved by another EV should never happen
            # under correct masking; record as infeasible if it does.
            self.infeasible = True
            self.infeasibility_reasons.append(f"customer_{j}_reserved_conflict")
        if not self.reserved_mask[j]:
            self.reserve_customer(ev, j)

        arrival = ev.current_time + t
        ev.remaining_battery -= e
        self._record_battery_shortfall(ev, max(0.0, -ev.remaining_battery), "customer")

        ready = node.ready_time
        due = node.due_time
        waiting = max(0.0, ready - arrival)
        service_start = max(arrival, ready)
        departure = service_start + node.service_time
        lateness = max(0.0, service_start - due)
        if self.hard_time_windows and lateness > EPS:
            raise RuntimeError(
                f"Hard TW violation on customer {j}: service_start={service_start:.4f} > due={due:.4f}"
            )
        self._record_customer_lateness(
            ev, j, lateness, travel=t, arrival=arrival, ready=ready, due=due, service_start=service_start,
        )

        ev.remaining_capacity -= node.demand
        if ev.remaining_capacity < -EPS:
            self.total_capacity_violation += -ev.remaining_capacity
            self.capacity_violation_count += 1

        ev.current_time = departure
        ev.route_distance += d
        ev.total_waiting_time += waiting
        self.total_distance += d
        self.total_waiting_time += waiting

        ev.route_nodes.append(j)
        ev.arrival_times.append(arrival)
        ev.departure_times.append(departure)
        ev.battery_trace.append(ev.remaining_battery)
        ev.capacity_trace.append(ev.remaining_capacity)
        ev.current_node = j  # EV physically moves to the served customer
        ev.active = True
        ev.busy = False

        # Resolve reservation -> served.
        self._release_reservation(j)
        ev.current_reserved_customer = None
        if not self.served_mask[j]:
            self.served_mask[j] = True
            self.num_served_customers += 1

        return {
            "type": "customer",
            "node": j,
            "arrival": arrival,
            "departure": departure,
            "waiting": waiting,
            "lateness": lateness,
            "customer_lateness": lateness,
            "depot_lateness": 0.0,
            "service_start": service_start,
            "service_end": departure,
        }

    def _charging_rate_at(self, node: int) -> float:
        """Energy recharge rate at a node (per-station override, else vehicle default)."""
        inst = self.instance
        if inst.station_charging_rate is not None:
            rate = float(inst.station_charging_rate[node])
            if rate > 1e-9 and not np.isnan(rate):
                return rate
        vp = inst.vehicle
        if vp.inverse_charging_rate > 1e-9:
            return 1.0 / vp.inverse_charging_rate
        return 1.0

    def _price_per_energy_at(self, node: int) -> float:
        """Expected price per unit charged energy at a node (0 if unknown or depot)."""
        inst = self.instance
        if node == inst.depot_index:
            return 0.0
        if inst.station_price_per_energy is not None:
            price = float(inst.station_price_per_energy[node])
            if price > 1e-9 and not np.isnan(price):
                return price
        return 0.0

    def _apply_recharge(
        self,
        ev: EVState,
        node: int,
        charge_portion: float | None,
        *,
        free: bool,
    ) -> dict:
        inst = self.instance
        vp = inst.vehicle
        d, t, e = self._move_energy_time(ev, node)
        portion = 1.0 if charge_portion is None else float(charge_portion)

        arrival = ev.current_time + t
        ev.remaining_battery -= e
        cat = "depot" if inst.nodes[node].is_depot else "station"
        self._record_battery_shortfall(ev, max(0.0, -ev.remaining_battery), cat)
        battery_after_arrival = ev.remaining_battery

        missing = max(0.0, vp.battery_capacity - battery_after_arrival)
        charged_energy = portion * missing
        new_battery = min(vp.battery_capacity, battery_after_arrival + charged_energy)
        charging_time = charged_energy / self._charging_rate_at(node)

        if inst.nodes[node].is_depot:
            waiting = max(0.0, inst.nodes[node].ready_time - arrival)
            lateness = max(0.0, arrival - inst.nodes[node].due_time)
            self._record_depot_lateness(lateness)
            self.total_waiting_time += waiting
            ev.total_waiting_time += waiting
            charge_start = max(arrival, inst.nodes[node].ready_time)
        else:
            waiting = 0.0
            lateness = 0.0
            charge_start = arrival

        charge_price = 0.0 if free else self._price_per_energy_at(node) * charged_energy

        ev.remaining_battery = new_battery
        ev.current_time = charge_start + charging_time
        ev.route_distance += d
        ev.total_charging_time += charging_time
        ev.total_charging_price += charge_price
        self.total_distance += d
        self.total_charging_time += charging_time
        self.total_charging_price += charge_price

        ev.route_nodes.append(node)
        ev.arrival_times.append(arrival)
        ev.departure_times.append(ev.current_time)
        ev.battery_trace.append(ev.remaining_battery)
        ev.capacity_trace.append(ev.remaining_capacity)
        ev.current_node = node
        ev.charging_actions.append({
            "node": node,
            "portion": portion,
            "charged_energy": charged_energy,
            "charging_time": charging_time,
            "charging_price": charge_price,
        })
        ev.active = True
        ev.busy = False

        node_type = "depot_recharge" if inst.nodes[node].is_depot else "station"
        return {
            "type": node_type,
            "node": node,
            "arrival": arrival,
            "charged_energy": charged_energy,
            "charging_time": charging_time,
            "charging_price": charge_price,
            "waiting": waiting,
            "lateness": lateness,
            "customer_lateness": 0.0,
            "depot_lateness": lateness if inst.nodes[node].is_depot else 0.0,
            "delta_distance": d,
            "charge_portion": portion,
        }

    def _finalize_depot_arrival(
        self,
        ev: EVState,
        arrival: float,
        distance: float,
    ) -> tuple[float, float, float]:
        """Apply depot closing window, distance, waiting; return (waiting, lateness, departure)."""
        depot = self.instance.depot_index
        node = self.instance.nodes[depot]
        waiting = max(0.0, node.ready_time - arrival)
        lateness = max(0.0, arrival - node.due_time)
        self._record_depot_lateness(lateness)
        self.total_waiting_time += waiting
        ev.total_waiting_time += waiting
        service_start = max(arrival, node.ready_time)
        departure = service_start + node.service_time
        ev.current_time = departure
        ev.route_distance += distance
        self.total_distance += distance
        return waiting, lateness, departure

    def apply_depot_finish(self, ev: EVState) -> dict:
        """Policy-chosen route finish: return to depot if needed and mark EV finished."""
        return self.auto_close_route(
            ev, forced=False, return_info=True, action_type="depot_finish", count_step=False,
        )

    def auto_close_route(
        self,
        ev: EVState,
        *,
        forced: bool = True,
        return_info: bool = True,
        action_type: str = "auto_close_depot",
        count_step: bool = True,
    ) -> dict:
        """End the EV route with a mandatory return to the depot when required."""
        inst = self.instance
        depot = inst.depot_index
        if count_step:
            self.step_count += 1
        previous_node = ev.current_node
        battery_before = ev.remaining_battery
        time_before = ev.current_time
        if forced:
            self.forced_auto_close_count += 1

        if ev.current_node == depot:
            ev.finished = True
            ev.active = False
            return self._close_info_dict(
                ev, previous_node, depot, action_type,
                delta_distance=0.0, delta_travel_time=0.0, delta_energy=0.0,
                arrival=ev.current_time, waiting=0.0, lateness=0.0,
                battery_before=battery_before, battery_after=ev.remaining_battery,
                route_closed=True, forced=forced,
            )

        d, t, e = self._move_energy_time(ev, depot)
        arrival = ev.current_time + t
        ev.remaining_battery -= e
        shortfall = max(0.0, -ev.remaining_battery)
        self._record_battery_shortfall(ev, shortfall, "auto_close")
        if shortfall > EPS:
            self.depot_return_violation += 1
            self.infeasible = True
            self.infeasibility_reasons.append(
                f"ev_{ev.ev_id}_cannot_return_to_depot_from_{ev.current_node}"
            )

        waiting, lateness, departure = self._finalize_depot_arrival(ev, arrival, d)

        ev.route_nodes.append(depot)
        ev.arrival_times.append(arrival)
        ev.departure_times.append(departure)
        ev.battery_trace.append(ev.remaining_battery)
        ev.capacity_trace.append(ev.remaining_capacity)
        ev.current_node = depot
        ev.finished = True
        ev.active = False
        self.current_event_time = max(self.current_event_time, ev.current_time)
        return self._close_info_dict(
            ev, previous_node, depot, action_type,
            delta_distance=d, delta_travel_time=t, delta_energy=e,
            arrival=arrival, waiting=waiting, lateness=lateness,
            battery_before=battery_before, battery_after=ev.remaining_battery,
            route_closed=True, forced=forced, departure=departure, time_before=time_before,
        )

    def _close_info_dict(
        self, ev, from_node, to_node, action_type, *, delta_distance, delta_travel_time,
        delta_energy, arrival, waiting, lateness, battery_before, battery_after,
        route_closed, forced, departure=None, time_before=None,
    ) -> dict:
        inst = self.instance
        depot = inst.depot_index
        due = float(inst.nodes[depot].due_time)
        return {
            "type": "route_close" if action_type == "auto_close_depot" else action_type,
            "action_type": action_type,
            "node": to_node,
            "from_node": from_node,
            "to_node": to_node,
            "arrival": arrival,
            "departure": departure if departure is not None else arrival,
            "waiting": waiting,
            "lateness": lateness,
            "customer_lateness": 0.0,
            "depot_lateness": lateness,
            "route_closed": route_closed,
            "forced_auto_close": forced,
            "ev_id": ev.ev_id,
            "previous_node": from_node,
            "delta_travel_time": delta_travel_time,
            "delta_energy_used": delta_energy,
            "delta_distance": delta_distance,
            "delta_duration": (ev.current_time - time_before) if time_before is not None else delta_travel_time,
            "delta_waiting": waiting,
            "delta_lateness": lateness,
            "battery_before": battery_before,
            "battery_after": battery_after,
            "depot_due_time": due,
            "customer_served": False,
            "charging_time": 0.0,
            "charged_energy": 0.0,
        }

    def try_auto_close_route(self, ev: EVState) -> dict | None:
        """Close the route only when the full feasible mask is empty."""
        mask, _ = self.feasible_mask(ev)
        if mask.any():
            return None
        return self.auto_close_route(ev, forced=True, return_info=True)

    def finish_idle_ev(self, ev: EVState) -> None:
        """Deprecated alias: always close with mandatory depot return."""
        self.auto_close_route(ev)

    # ------------------------------------------------------------------ #
    # Feature extraction for the policy
    # ------------------------------------------------------------------ #
    def ev_dynamic_features(self, ev: EVState) -> np.ndarray:
        inst = self.instance
        vp = inst.vehicle
        horizon = inst.planning_horizon if inst.planning_horizon > 0 else 1.0
        max_dist = inst.distance_matrix.max() * max(inst.num_nodes, 1) or 1.0
        battery_frac = ev.remaining_battery / vp.battery_capacity if vp.battery_capacity else 0.0
        cap_frac = ev.remaining_capacity / vp.load_capacity if vp.load_capacity else 0.0
        return np.array([
            ev.current_time / horizon,
            battery_frac,
            cap_frac,
            ev.route_distance / max_dist,
            float(ev.active),
            float(ev.finished),
            1.0 - battery_frac,
            1.0 - cap_frac,
        ], dtype=np.float32)

    def ev_token_features(self, ev: EVState) -> np.ndarray:
        """Compact per-EV features for the communication module."""
        inst = self.instance
        vp = inst.vehicle
        horizon = inst.planning_horizon if inst.planning_horizon > 0 else 1.0
        max_dist = inst.distance_matrix.max() * max(inst.num_nodes, 1) or 1.0
        return np.array([
            ev.current_time / horizon,
            ev.remaining_battery / vp.battery_capacity if vp.battery_capacity else 0.0,
            ev.remaining_capacity / vp.load_capacity if vp.load_capacity else 0.0,
            float(ev.active),
            float(ev.finished),
            ev.route_distance / max_dist,
        ], dtype=np.float32)

    def global_features(self) -> np.ndarray:
        inst = self.instance
        horizon = inst.planning_horizon if inst.planning_horizon > 0 else 1.0
        num_c = max(inst.num_customers, 1)
        total_demand = inst.total_demand or 1.0
        remaining_demand = sum(
            inst.nodes[c].demand for c in inst.customer_indices if not self.served_mask[c]
        )
        max_dist = inst.distance_matrix.max() * max(inst.num_nodes, 1) or 1.0
        active = sum(1 for ev in self.evs if ev.active and not ev.finished)
        finished = sum(1 for ev in self.evs if ev.finished)
        reserved = int(self.reserved_mask[inst.customer_indices].sum()) if inst.customer_indices else 0
        return np.array([
            self.num_served_customers / num_c,
            reserved / num_c,
            active / self.m,
            finished / self.m,
            self.total_distance / max_dist,
            self.total_charging_time / horizon,
            self.current_event_time / horizon,
            self.m / num_c,
            remaining_demand / total_demand,
        ], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Scoring
    # ------------------------------------------------------------------ #
    def used_evs(self) -> int:
        return sum(1 for ev in self.evs if ev.used())

    def build_summary(self) -> SolutionSummary:
        inst = self.instance
        unserved = inst.num_customers - self.num_served_customers
        unresolved_reserved = int(self.reserved_mask[inst.customer_indices].sum()) if inst.customer_indices else 0
        # Route duration: sum of each used EV's final clock.
        total_duration = sum(ev.current_time for ev in self.evs if ev.used())
        infeasible_flag = int(self.infeasible or unserved > 0)
        return SolutionSummary(
            used_evs=self.used_evs(),
            total_distance=self.total_distance,
            total_duration=total_duration,
            total_charging_time=self.total_charging_time,
            total_charging_price=self.total_charging_price,
            total_waiting_time=self.total_waiting_time,
            unserved_customers=unserved,
            duplicate_customer_visits=self.duplicate_customer_visits,
            total_lateness=self.total_lateness,
            customer_lateness=self.customer_lateness,
            depot_lateness=self.depot_lateness,
            customer_tw_violations=self.customer_tw_violation_count,
            depot_tw_violations=self.depot_tw_violation_count,
            total_battery_violation=self.total_battery_violation,
            total_capacity_violation=self.total_capacity_violation,
            depot_return_violation=self.depot_return_violation,
            unresolved_reserved_customers=unresolved_reserved,
            infeasible_flag=infeasible_flag,
        )

    def score(self, weights: CostWeights) -> tuple[float, dict[str, float], SolutionSummary]:
        summary = self.build_summary()
        j, breakdown = compute_cost(summary, weights)
        return j, breakdown, summary

    def feasible_solution(self) -> bool:
        return is_feasible(self.build_summary())
