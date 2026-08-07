"""Exact route / solution replay for TG-FPS (no live-env mutation before accept)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.search.score import SearchScore


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

    # Skip the initial depot occupancy (already at depot).
    for pos in range(1, len(nodes)):
        node = int(nodes[pos])
        portion = labels[pos]
        node_obj = instance.nodes[node]
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
) -> SolutionReplayResult:
    """Replay a multi-route solution jointly; detect duplicates and unserved."""
    cfg = config or {}
    problem = cfg.get("problem") or {}
    portions = list(
        charging_portions
        or problem.get("charging_portions")
        or [0.25, 0.5, 0.75, 1.0]
    )
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
        hard_time_windows=True,
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
        for pos in range(1, len(nodes_n)):
            node = int(nodes_n[pos])
            portion = labels_n[pos]
            node_obj = instance.nodes[node]
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
                )
            )
            hard = (
                int(env.battery_violation_count)
                + int(env.capacity_violation_count)
                + int(env.time_window_violation_count)
                + int(env.depot_return_violation)
            )
            if hard > 0 or env.infeasible:
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
    )
    result.score = SearchScore(
        hard_violation_count=hard,
        unserved_customer_count=unserved,
        used_evs=int(summary.used_evs),
        total_distance=float(summary.total_distance),
        total_charging_time=float(summary.total_charging_time),
        total_duration=float(summary.total_duration),
        paper_objective=float(objective),
    )
    return result


def replay_env_solution(env: EVRPTWEnv, weights: CostWeights | None = None) -> SolutionReplayResult:
    """Extract routes from env and re-validate via disposable replay."""
    routes = extract_routes_from_env(env)
    return replay_solution(
        env.instance,
        routes,
        m=env.m,
        charging_portions=list(env.charging_portions),
        weights=weights or CostWeights(),
    )
