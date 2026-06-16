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
        max_steps_factor: int = 6,
    ):
        self.instance = instance
        self.m = m
        self.charging_portions = charging_portions or [0.25, 0.50, 0.75, 1.0]
        self.use_horizon_check = use_horizon_check
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
        self.total_battery_violation = 0.0
        self.total_capacity_violation = 0.0
        self.depot_return_violation = 0
        self.duplicate_customer_visits = 0
        # Event counts (distinct from magnitude sums above).
        self.time_window_violation_count = 0
        self.battery_violation_count = 0
        self.capacity_violation_count = 0
        self.infeasible = False
        self.infeasibility_reasons: list[str] = []
        self.step_count = 0

    # ------------------------------------------------------------------ #
    # Event selection and feasibility
    # ------------------------------------------------------------------ #
    def select_acting_ev(self) -> EVState | None:
        return select_next_ev(self.evs)

    def feasible_mask(self, ev: EVState) -> tuple[np.ndarray, dict[int, str]]:
        return compute_feasible_mask(self.instance, self, ev, self.use_horizon_check)

    def all_customers_served(self) -> bool:
        return self.num_served_customers >= self.instance.num_customers

    def is_done(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        # Done when every EV is finished.
        if all(ev.finished for ev in self.evs):
            return True
        # Done when all customers are served and no EV still needs to move.
        if self.all_customers_served():
            # EVs away from depot still need to return; otherwise done.
            return all(ev.finished or ev.current_node == self.instance.depot_index for ev in self.evs)
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
    def apply_action(self, ev: EVState, node: int, charge_portion: float | None = None) -> dict:
        """Apply the EV's chosen next node and return an info dict.

        For customers, the reservation is taken (if not already) and resolved to
        a completed service within the same atomic event, because the EV clock
        advances to the service departure time.
        """
        inst = self.instance
        node_obj = inst.nodes[node]
        self.step_count += 1

        if node_obj.is_customer:
            info = self._apply_customer(ev, node)
        elif node_obj.is_charging_station:
            info = self._apply_station(ev, node, charge_portion)
        elif node_obj.is_depot:
            info = self._apply_depot(ev, node)
        else:  # pragma: no cover - node types are exhaustive
            raise ValueError(f"Unknown node type for node {node}")

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
        if ev.remaining_battery < -EPS:
            self.total_battery_violation += -ev.remaining_battery
            self.battery_violation_count += 1

        ready = node.ready_time
        due = node.due_time
        waiting = max(0.0, ready - arrival)
        service_start = max(arrival, ready)
        departure = service_start + node.service_time
        lateness = max(0.0, arrival - due)
        self.total_lateness += lateness
        if lateness > EPS:
            self.time_window_violation_count += 1

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
        }

    def _charging_rate_at(self, node: int) -> float:
        """Energy recharge rate at a node (Hanif CSV per station, else vehicle default)."""
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
        """Expected Hanif price per unit charged energy at a node (0 if unknown)."""
        inst = self.instance
        if inst.station_price_per_energy is not None:
            price = float(inst.station_price_per_energy[node])
            if price > 1e-9 and not np.isnan(price):
                return price
        return 0.0

    def _apply_station(self, ev: EVState, s: int, charge_portion: float | None) -> dict:
        inst = self.instance
        vp = inst.vehicle
        d, t, e = self._move_energy_time(ev, s)
        portion = 1.0 if charge_portion is None else float(charge_portion)

        arrival = ev.current_time + t
        ev.remaining_battery -= e
        if ev.remaining_battery < -EPS:
            self.total_battery_violation += -ev.remaining_battery
            self.battery_violation_count += 1
        battery_after_arrival = ev.remaining_battery

        missing = max(0.0, vp.battery_capacity - battery_after_arrival)
        charged_energy = portion * missing
        new_battery = min(vp.battery_capacity, battery_after_arrival + charged_energy)
        charging_time = charged_energy / self._charging_rate_at(s)
        charge_price = self._price_per_energy_at(s) * charged_energy

        ev.remaining_battery = new_battery
        ev.current_time = arrival + charging_time
        ev.route_distance += d
        ev.total_charging_time += charging_time
        ev.total_charging_price += charge_price
        self.total_distance += d
        self.total_charging_time += charging_time
        self.total_charging_price += charge_price

        ev.route_nodes.append(s)
        ev.arrival_times.append(arrival)
        ev.departure_times.append(ev.current_time)
        ev.battery_trace.append(ev.remaining_battery)
        ev.capacity_trace.append(ev.remaining_capacity)
        ev.current_node = s  # EV physically moves to the charging station
        ev.charging_actions.append({
            "node": s,
            "portion": portion,
            "charged_energy": charged_energy,
            "charging_time": charging_time,
            "charging_price": charge_price,
        })
        ev.active = True
        ev.busy = False

        return {
            "type": "station",
            "node": s,
            "arrival": arrival,
            "charged_energy": charged_energy,
            "charging_time": charging_time,
            "charging_price": charge_price,
        }

    def _apply_depot(self, ev: EVState, depot: int) -> dict:
        d, t, e = self._move_energy_time(ev, depot)
        arrival = ev.current_time + t
        ev.remaining_battery -= e
        if ev.remaining_battery < -EPS:
            self.total_battery_violation += -ev.remaining_battery
            self.battery_violation_count += 1
            self.depot_return_violation += 1
        ev.current_time = arrival
        ev.route_distance += d
        self.total_distance += d

        ev.route_nodes.append(depot)
        ev.arrival_times.append(arrival)
        ev.departure_times.append(arrival)
        ev.battery_trace.append(ev.remaining_battery)
        ev.capacity_trace.append(ev.remaining_capacity)
        ev.current_node = depot  # EV physically returns to the depot
        ev.finished = True
        ev.active = False
        return {"type": "depot", "node": depot, "arrival": arrival}

    def finish_idle_ev(self, ev: EVState) -> None:
        """Mark an EV finished when it has no feasible action.

        If the EV is away from the depot this is a depot-return violation (it is
        stranded); if it is at the depot it simply remains unused.
        """
        if ev.current_node != self.instance.depot_index:
            self.depot_return_violation += 1
            self.infeasible = True
            self.infeasibility_reasons.append(f"ev_{ev.ev_id}_stranded_at_{ev.current_node}")
        ev.finished = True
        ev.active = False

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
