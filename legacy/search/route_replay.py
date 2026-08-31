"""Exact route / solution replay for TG-FPS (no live-env mutation before accept)."""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.evaluation.paper_objective import (
    MODE_CLASSICAL,
    charging_rate_s,
    expected_price_s,
    expected_wait_s,
    objective_spec,
    phi_from_replay,
)
from src.search.score import SearchScore


def _arc_energy(instance, i: int, j: int) -> float:
    return float(instance.energy_matrix[int(i), int(j)])


def _arc_travel_time(instance, i: int, j: int) -> float:
    return float(instance.travel_time_matrix[int(i), int(j)])


@dataclass
class RouteEvent:
    position: int
    node: int
    arrival: float
    departure: float
    waiting: float
    service_start: float
    battery_before: float
    battery_after: float
    charged_energy: float
    charge_portion: float | None
    load_after: float
    charging_time: float = 0.0


@dataclass
class RouteReplayResult:
    feasible: bool
    first_failure: str | None
    first_failure_position: int | None
    served_customers: list[int]
    distance: float
    duration: float
    charging_time: float
    waiting_time: float
    end_time: float
    route_events: list[RouteEvent] = field(default_factory=list)
    hard_violation_count: int = 0
    customer_early_waiting_time: float = 0.0
    station_waiting_time: float = 0.0
    energy_consumed: float = 0.0
    travel_time: float = 0.0
    service_time: float = 0.0


@dataclass
class SolutionReplayResult:
    feasible: bool
    first_failure: str | None
    served_customers: list[int]
    unserved_customers: int
    used_evs: int
    distance: float
    duration: float
    charging_time: float
    waiting_time: float
    objective: float
    hard_violation_count: int
    duplicate_customers: int
    route_results: list[RouteReplayResult] = field(default_factory=list)
    score: SearchScore | None = None
    customer_early_waiting_time: float = 0.0
    station_waiting_time: float = 0.0
    phi_breakdown: dict[str, Any] | None = None
    battery_metrics: dict[str, Any] = field(default_factory=dict)
    station_metrics: dict[str, Any] = field(default_factory=dict)
    late_customers: int = 0
    depot_return_violations: int = 0
    time_window_violations: int = 0
    energy_consumed: float = 0.0
    travel_time: float = 0.0
    service_time: float = 0.0


def _normalize_route(
    route_nodes: list[int],
    charge_portions: list[float | None] | None,
    depot_index: int,
) -> tuple[list[int], list[float | None]]:
    nodes = [int(n) for n in route_nodes]
    if not nodes:
        nodes = [depot_index, depot_index]
    if nodes[0] != depot_index:
        nodes = [depot_index] + nodes
    if nodes[-1] != depot_index:
        nodes = nodes + [depot_index]
    n = len(nodes)
    if charge_portions is None:
        portions: list[float | None] = [None] * n
    else:
        portions = list(charge_portions)
        if len(portions) < n:
            portions = portions + [None] * (n - len(portions))
        portions = portions[:n]
    return nodes, portions


def replay_route(
    instance,
    proposed_route: list[int],
    charging_labels: list[float | None] | None = None,
    *,
    config: dict | None = None,
    charging_portions: list[float] | None = None,
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    hard_time_windows: bool = True,
) -> RouteReplayResult:
    """Replay one EV route from depot through visits and back.

    Does not mutate any caller-owned environment. Uses a disposable single-EV env.
    """
    cfg = config or {}
    problem = cfg.get("problem") or {}
    portions = list(
        charging_portions
        or problem.get("charging_portions")
        or [0.25, 0.5, 0.75, 1.0]
    )
    depot = int(instance.depot_index)
    nodes, labels = _normalize_route(proposed_route, charging_labels, depot)

    env = EVRPTWEnv(
        instance,
        m=1,
        charging_portions=portions,
        use_horizon_check=bool(problem.get("use_horizon_check", use_horizon_check)),
        use_depot_charging=bool(problem.get("use_depot_charging", use_depot_charging)),
        hard_time_windows=hard_time_windows,
    )
    ev = env.evs[0]
    events: list[RouteEvent] = []
    served: list[int] = []
    first_failure = None
    first_pos = None
    energy_consumed = 0.0
    travel_time = 0.0
    service_time = 0.0

    # Skip the initial depot occupancy (already at depot).
    for pos in range(1, len(nodes)):
        node = int(nodes[pos])
        portion = labels[pos]
        node_obj = instance.nodes[node]
        prev = int(nodes[pos - 1])
        energy_consumed += _arc_energy(instance, prev, node)
        travel_time += _arc_travel_time(instance, prev, node)
        if node_obj.is_customer:
            service_time += float(node_obj.service_time)
        battery_before = float(ev.remaining_battery)
        try:
            if node_obj.is_depot and pos == len(nodes) - 1:
                info = env.apply_depot_finish(ev)
            elif node_obj.is_depot:
                info = env.apply_action(ev, node, portion if portion is not None else 1.0, depot_finish=False)
            elif node_obj.is_charging_station:
                info = env.apply_action(ev, node, 1.0 if portion is None else float(portion), depot_finish=False)
            else:
                info = env.apply_action(ev, node, None, depot_finish=False)
                if node_obj.is_customer:
                    served.append(node)
        except Exception as exc:  # hard TW / illegal transition
            first_failure = f"pos={pos} node={node}: {exc}"
            first_pos = pos
            break

        events.append(
            RouteEvent(
                position=pos,
                node=node,
                arrival=float(info.get("arrival", ev.arrival_times[-1] if ev.arrival_times else 0.0)),
                departure=float(ev.departure_times[-1] if ev.departure_times else ev.current_time),
                waiting=float(info.get("waiting", 0.0) or 0.0),
                service_start=float(info.get("service_start", info.get("arrival", 0.0)) or 0.0),
                battery_before=battery_before,
                battery_after=float(ev.remaining_battery),
                charged_energy=float(info.get("charged_energy", 0.0) or 0.0),
                charge_portion=None if portion is None else float(portion),
                load_after=float(ev.remaining_capacity),
                charging_time=float(info.get("charging_time", 0.0) or 0.0),
            )
        )
        hard = (
            int(env.battery_violation_count)
            + int(env.capacity_violation_count)
            + int(env.time_window_violation_count)
            + int(env.depot_return_violation)
        )
        if hard > 0 or env.infeasible:
            first_failure = first_failure or (
                f"pos={pos} node={node}: hard_violation "
                f"batt={env.battery_violation_count} cap={env.capacity_violation_count} "
                f"tw={env.time_window_violation_count} depot={env.depot_return_violation} "
                f"reasons={list(env.infeasibility_reasons)[:3]}"
            )
            first_pos = pos if first_pos is None else first_pos
            break

    hard_count = (
        int(env.battery_violation_count)
        + int(env.capacity_violation_count)
        + int(env.time_window_violation_count)
        + int(env.depot_return_violation)
        + int(bool(env.infeasible))
    )
    feasible = first_failure is None and hard_count == 0 and bool(ev.finished or nodes[-1] == depot)
    if first_failure is None and not ev.finished and nodes[-1] == depot:
        # Depot finish may have been applied; treat as ok if no violations.
        feasible = hard_count == 0

    return RouteReplayResult(
        feasible=bool(feasible),
        first_failure=first_failure,
        first_failure_position=first_pos,
        served_customers=served,
        distance=float(ev.route_distance),
        duration=float(ev.current_time),
        charging_time=float(ev.total_charging_time),
        waiting_time=float(ev.total_waiting_time),
        end_time=float(ev.current_time),
        route_events=events,
        hard_violation_count=hard_count,
        customer_early_waiting_time=float(ev.customer_early_waiting_time),
        station_waiting_time=float(ev.station_waiting_time),
        energy_consumed=float(energy_consumed),
        travel_time=float(travel_time),
        service_time=float(service_time),
    )


def extract_routes_from_env(env: EVRPTWEnv) -> list[tuple[list[int], list[float | None]]]:
    """Serialize used EV routes + charge labels from a live env (for replay tests)."""
    out: list[tuple[list[int], list[float | None]]] = []
    for ev in env.evs:
        if not ev.used() and len(ev.route_nodes) <= 1:
            continue
        nodes = list(ev.route_nodes)
        labels: list[float | None] = [None] * len(nodes)
        # Match charging_actions in visit order for station/depot-recharge nodes.
        charge_iter = iter(ev.charging_actions)
        for i, node in enumerate(nodes):
            if i == 0:
                continue
            node_obj = env.instance.nodes[node]
            if node_obj.is_charging_station or (node_obj.is_depot and i < len(nodes) - 1):
                try:
                    ca = next(charge_iter)
                    labels[i] = float(ca.get("portion", 1.0))
                except StopIteration:
                    labels[i] = 1.0
        if nodes and nodes[-1] != env.instance.depot_index:
            nodes = nodes + [env.instance.depot_index]
            labels = labels + [None]
        out.append((nodes, labels))
    return out


def replay_solution(
    instance,
    routes: list[list[int]] | list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    charging_portions: list[float] | None = None,
    hard_time_windows: bool = True,
    stop_on_hard: bool | None = None,
) -> SolutionReplayResult:
    """Replay a multi-route solution jointly; detect duplicates and unserved."""
    cfg = config or {}
    problem = cfg.get("problem") or {}
    portions = list(
        charging_portions
        or problem.get("charging_portions")
        or [0.25, 0.5, 0.75, 1.0]
    )
    stop = bool(hard_time_windows) if stop_on_hard is None else bool(stop_on_hard)
    if weights is not None:
        w = weights
    elif cfg:
        w = CostWeights.from_eval_config(cfg)
    else:
        w = CostWeights()

    normalized: list[tuple[list[int], list[float | None]]] = []
    for r in routes:
        if isinstance(r, tuple):
            normalized.append((list(r[0]), list(r[1]) if r[1] is not None else []))
        else:
            normalized.append((list(r), []))

    fleet = int(m if m is not None else max(len(normalized), 1))
    env = EVRPTWEnv(
        instance,
        m=fleet,
        charging_portions=portions,
        use_horizon_check=bool(problem.get("use_horizon_check", True)),
        use_depot_charging=bool(problem.get("use_depot_charging", True)),
        hard_time_windows=bool(hard_time_windows),
    )

    route_results: list[RouteReplayResult] = []
    served_set: set[int] = set()
    duplicates = 0
    first_failure = None

    for ri, (nodes, labels) in enumerate(normalized):
        if ri >= fleet:
            first_failure = first_failure or f"route {ri}: exceeds fleet m={fleet}"
            break
        # Replay this route onto env.evs[ri] by applying actions.
        depot = instance.depot_index
        nodes_n, labels_n = _normalize_route(nodes, labels if labels else None, depot)
        ev = env.evs[ri]
        events: list[RouteEvent] = []
        route_served: list[int] = []
        route_fail = None
        route_fail_pos = None
        route_energy = 0.0
        route_travel = 0.0
        route_service = 0.0
        for pos in range(1, len(nodes_n)):
            node = int(nodes_n[pos])
            portion = labels_n[pos]
            node_obj = instance.nodes[node]
            prev = int(nodes_n[pos - 1])
            route_energy += _arc_energy(instance, prev, node)
            route_travel += _arc_travel_time(instance, prev, node)
            if node_obj.is_customer:
                route_service += float(node_obj.service_time)
            battery_before = float(ev.remaining_battery)
            try:
                if node_obj.is_depot and pos == len(nodes_n) - 1:
                    info = env.apply_depot_finish(ev)
                elif node_obj.is_depot:
                    info = env.apply_action(
                        ev, node, portion if portion is not None else 1.0, depot_finish=False
                    )
                elif node_obj.is_charging_station:
                    info = env.apply_action(
                        ev, node, 1.0 if portion is None else float(portion), depot_finish=False
                    )
                else:
                    if env.served_mask[node]:
                        duplicates += 1
                    info = env.apply_action(ev, node, None, depot_finish=False)
                    if node_obj.is_customer:
                        route_served.append(node)
                        if node in served_set:
                            duplicates += 1
                        served_set.add(node)
            except Exception as exc:
                route_fail = f"route={ri} pos={pos} node={node}: {exc}"
                route_fail_pos = pos
                first_failure = first_failure or route_fail
                break
            events.append(
                RouteEvent(
                    position=pos,
                    node=node,
                    arrival=float(info.get("arrival", 0.0) or 0.0),
                    departure=float(ev.departure_times[-1] if ev.departure_times else ev.current_time),
                    waiting=float(info.get("waiting", 0.0) or 0.0),
                    service_start=float(info.get("service_start", info.get("arrival", 0.0)) or 0.0),
                    battery_before=battery_before,
                    battery_after=float(ev.remaining_battery),
                    charged_energy=float(info.get("charged_energy", 0.0) or 0.0),
                    charge_portion=None if portion is None else float(portion),
                    load_after=float(ev.remaining_capacity),
                    charging_time=float(info.get("charging_time", 0.0) or 0.0),
                )
            )
            hard = (
                int(env.battery_violation_count)
                + int(env.capacity_violation_count)
                + int(env.time_window_violation_count)
                + int(env.depot_return_violation)
            )
            if (hard > 0 or env.infeasible) and stop:
                route_fail = route_fail or (
                    f"route={ri} pos={pos} node={node}: hard_violation"
                )
                route_fail_pos = pos if route_fail_pos is None else route_fail_pos
                first_failure = first_failure or route_fail
                break

        hard_count = (
            int(env.battery_violation_count)
            + int(env.capacity_violation_count)
            + int(env.time_window_violation_count)
            + int(env.depot_return_violation)
            + int(bool(env.infeasible))
        )
        route_results.append(
            RouteReplayResult(
                feasible=route_fail is None and hard_count == 0,
                first_failure=route_fail,
                first_failure_position=route_fail_pos,
                served_customers=route_served,
                distance=float(ev.route_distance),
                duration=float(ev.current_time),
                charging_time=float(ev.total_charging_time),
                waiting_time=float(ev.total_waiting_time),
                end_time=float(ev.current_time),
                route_events=events,
                hard_violation_count=hard_count,
                customer_early_waiting_time=float(ev.customer_early_waiting_time),
                station_waiting_time=float(ev.station_waiting_time),
                energy_consumed=float(route_energy),
                travel_time=float(route_travel),
                service_time=float(route_service),
            )
        )

    # Close remaining open EVs for a complete score.
    for ev in env.evs:
        if ev.used() and not ev.finished:
            try:
                env.apply_depot_finish(ev)
            except Exception as exc:
                first_failure = first_failure or f"close ev={ev.ev_id}: {exc}"

    objective, _bd, summary = env.score(w)
    hard = (
        int(env.battery_violation_count)
        + int(env.capacity_violation_count)
        + int(env.time_window_violation_count)
        + int(env.depot_return_violation)
        + int(bool(env.infeasible))
        + int(duplicates > 0)
    )
    n_cust = int(instance.num_customers)
    unserved = max(0, n_cust - len(served_set))
    late_customers = 0
    for route in route_results:
        for ev in route.route_events:
            nd = instance.nodes[int(ev.node)]
            if nd.is_customer and float(ev.arrival) > float(nd.due_time) + 1e-6:
                late_customers += 1
    feasible = (
        first_failure is None
        and hard == 0
        and unserved == 0
        and bool(env.feasible_solution())
    )
    result = SolutionReplayResult(
        feasible=feasible,
        first_failure=first_failure,
        served_customers=sorted(served_set),
        unserved_customers=unserved,
        used_evs=int(summary.used_evs),
        distance=float(summary.total_distance),
        duration=float(summary.total_duration),
        charging_time=float(summary.total_charging_time),
        waiting_time=float(summary.total_waiting_time),
        objective=float(objective),
        hard_violation_count=hard,
        duplicate_customers=int(duplicates),
        route_results=route_results,
        late_customers=int(late_customers),
        depot_return_violations=int(env.depot_return_violation),
        time_window_violations=int(env.time_window_violation_count),
        energy_consumed=sum(float(r.energy_consumed) for r in route_results),
        travel_time=sum(float(r.travel_time) for r in route_results),
        service_time=sum(float(r.service_time) for r in route_results),
    )
    result.customer_early_waiting_time = float(summary.customer_early_waiting_time)
    result.station_waiting_time = float(summary.station_waiting_time)
    phi_info = phi_from_replay(instance, result, cfg)
    spec = objective_spec(cfg)
    result.phi_breakdown = phi_info
    result.battery_metrics = _battery_metrics(instance, result)
    result.station_metrics = _station_metrics(instance, result, phi_info)
    operational = (
        float(summary.total_distance)
        if spec["mode"] == MODE_CLASSICAL
        else float(phi_info["Phi"])
    )
    result.score = SearchScore(
        hard_violation_count=hard,
        unserved_customer_count=unserved,
        used_evs=int(summary.used_evs),
        total_distance=float(summary.total_distance),
        total_charging_time=float(summary.total_charging_time),
        total_duration=float(summary.total_duration),
        paper_objective=operational,
        phi=float(phi_info["Phi"]),
        total_station_time=float(phi_info["tstation"]),
        objective_mode=str(spec["mode"]),
    )
    return result


def _battery_metrics(instance, result: SolutionReplayResult) -> dict[str, Any]:
    q = float(instance.vehicle.battery_capacity)
    energy_consumed = 0.0
    energy_charged = 0.0
    mins: list[float] = []
    route_mins: list[float] = []
    n_recharge_routes = 0
    n_used = 0
    charging_visits = 0
    charging_events: list[dict[str, Any]] = []
    for route in result.route_results:
        if not route.route_events:
            continue
        n_used += 1
        rmin = q
        charged_here = False
        prev_batt = q
        for ev in route.route_events:
            charged = float(ev.charged_energy or 0.0)
            # Traction energy on the incoming arc: drop before charge, restored after charge.
            energy_consumed += max(0.0, float(ev.battery_before) - float(ev.battery_after) + charged)
            energy_charged += charged
            rmin = min(rmin, float(ev.battery_before), float(ev.battery_after))
            if charged > 1e-12:
                charged_here = True
                charging_visits += 1
                price = expected_price_s(instance, int(ev.node))
                wait = expected_wait_s(instance, int(ev.node))
                rate = charging_rate_s(instance, int(ev.node))
                chg_t = float(ev.charging_time or 0.0)
                if chg_t <= 1e-12 and rate > 1e-12:
                    chg_t = charged / rate
                charging_events.append({
                    "station_id": int(ev.node),
                    "arrival_time": float(ev.arrival),
                    "battery_before_charging": float(ev.battery_before),
                    "energy_added": charged,
                    "battery_after_charging": float(ev.battery_after),
                    "charging_time": chg_t,
                    "expected_station_wait": wait,
                    "expected_price": price,
                    "charging_expense": charged * price,
                })
            prev_batt = float(ev.battery_after)
        mins.append(rmin)
        route_mins.append(rmin / max(q, 1e-9))
        if charged_here:
            n_recharge_routes += 1
    matrix_energy = float(getattr(result, "energy_consumed", 0.0) or 0.0)
    reported_energy = matrix_energy if matrix_energy > 1e-9 else energy_consumed
    extra = {
        "total_energy_consumed": reported_energy,
        "energy_consumed_battery_trace": energy_consumed,
        "total_energy_charged": energy_charged,
        "routes_requiring_charging": n_recharge_routes,
        "fraction_routes_requiring_recharge": n_recharge_routes / max(n_used, 1),
        "charging_visits": charging_visits,
        "charging_visits_per_used_ev": charging_visits / max(n_used, 1),
        "energy_charged_per_used_ev": energy_charged / max(n_used, 1),
        "battery_violations": int(result.hard_violation_count),
        "charging_events": charging_events,
    }
    if not mins:
        return {
            "minimum_absolute_battery": q,
            "minimum_soc_fraction": 1.0,
            "mean_route_minimum_soc": 1.0,
            "median_route_minimum_soc": 1.0,
            **extra,
            "total_energy_consumed": reported_energy,
            "total_energy_charged": 0.0,
            "charging_events": [],
        }
    return {
        "minimum_absolute_battery": min(mins),
        "minimum_soc_fraction": min(mins) / max(q, 1e-9),
        "mean_route_minimum_soc": sum(route_mins) / len(route_mins),
        "median_route_minimum_soc": float(statistics.median(route_mins)),
        **extra,
    }


def _station_metrics(instance, result: SolutionReplayResult, phi_info: dict[str, Any]) -> dict[str, Any]:
    visits = list(phi_info.get("charging_visit_records") or [])
    stations = [int(v["station_id"]) for v in visits]
    unique = sorted(set(stations))
    rates, prices, waits = [], [], []
    for sid in stations:
        from src.evaluation.paper_objective import charging_rate_s, expected_price_s, expected_wait_s
        rates.append(charging_rate_s(instance, sid))
        prices.append(expected_price_s(instance, sid))
        waits.append(expected_wait_s(instance, sid))
    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0
    return {
        "charging_visits": len(visits),
        "unique_stations_used": len(unique),
        "total_charged_energy": float(result.battery_metrics.get("total_energy_charged", 0.0)) if result.battery_metrics else sum(float(v["energy_added"]) for v in visits),
        "total_charging_time": float(result.charging_time),
        "expected_charging_cost": float(phi_info.get("expected_charging_cost") or 0.0),
        "risk_adjusted_charging_cost": float(phi_info.get("crisk") or 0.0),
        "CVaR95_charging_cost_exposure": float(phi_info.get("cvar95_charging_cost_exposure") or 0.0),
        "station_waiting_time_expected": float(phi_info.get("station_waiting_expected") or 0.0),
        "risk_adjusted_station_waiting": float(phi_info.get("wrisk") or 0.0),
        "CVaR95_wait_exposure": float(phi_info.get("cvar95_wait_exposure") or 0.0),
        "mean_charging_rate_visited": _mean(rates),
        "mean_expected_price_visited": _mean(prices),
        "mean_expected_wait_visited": _mean(waits),
    }


def replay_env_solution(
    env: EVRPTWEnv,
    weights: CostWeights | None = None,
    config: dict | None = None,
) -> SolutionReplayResult:
    """Extract routes from env and re-validate via disposable replay."""
    routes = extract_routes_from_env(env)
    cfg = dict(config or {})
    return replay_solution(
        env.instance,
        routes,
        config=cfg,
        m=env.m,
        charging_portions=list(env.charging_portions),
        weights=weights or CostWeights(),
    )
