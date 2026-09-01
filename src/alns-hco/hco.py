"""Heterogeneous Charging Oracle (HCO).

A deterministic route-level charging heuristic. It is not an exact solver and
does not claim global optimality.

For a fixed customer sequence the oracle may insert, replace, or drop stations
and set a just-enough partial charge. Station choice uses rate r_s, expected
wait W̄_s and expected price P̄_s.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.hanif.model import charging_rate, expected_price, expected_wait
from src.hanif.replay import customer_sequence, replay_route, replay_solution

EPS = 1e-9


@dataclass
class OraclePlan:
    feasible: bool
    nodes: list[int]
    labels: list[float | None]
    distance: float
    duration: float
    charging_expense: float
    charging_time: float
    station_waiting: float
    charged_energy: float
    charging_visits: int
    failure: str | None = None


@dataclass
class OracleCache:
    plans: dict = field(default_factory=dict)
    hits: int = 0
    calls: int = 0


def _empty(reason: str | None = None) -> OraclePlan:
    return OraclePlan(
        feasible=False, nodes=[], labels=[], distance=1e18, duration=1e18,
        charging_expense=1e18, charging_time=0.0, station_waiting=0.0,
        charged_energy=0.0, charging_visits=0, failure=reason,
    )


def _from_replay(plan) -> OraclePlan:
    if plan is None or not plan.feasible:
        return _empty(getattr(plan, "failure", None))
    return OraclePlan(
        feasible=True,
        nodes=list(plan.nodes),
        labels=list(plan.labels),
        distance=float(plan.distance),
        duration=float(plan.duration),
        charging_expense=float(plan.charging_expense),
        charging_time=float(plan.charging_time),
        station_waiting=float(plan.station_waiting),
        charged_energy=float(plan.charged_energy),
        charging_visits=int(plan.charging_visits),
    )


def just_enough_energy(instance, station: int, batt_arr: float, next_node: int, slack: float = 1.0) -> float:
    batt_cap = float(instance.vehicle.battery_capacity)
    missing = max(0.0, batt_cap - float(batt_arr))
    if missing <= EPS:
        return 0.0
    e_next = float(instance.energy_matrix[station, next_node])
    depot = int(instance.depot_index)
    onward = [float(instance.energy_matrix[next_node, depot])]
    for s in instance.charging_station_indices:
        onward.append(float(instance.energy_matrix[next_node, int(s)]))
    need = e_next + min(onward) + float(slack)
    return min(missing, max(0.0, need - float(batt_arr)))


def just_enough_alpha(instance, station: int, batt_arr: float, next_node: int) -> float:
    batt_cap = float(instance.vehicle.battery_capacity)
    missing = max(0.0, batt_cap - float(batt_arr))
    if missing <= EPS:
        return 0.0
    e = just_enough_energy(instance, station, batt_arr, next_node)
    return min(1.0, max(0.0, e / missing))


def just_enough_labels(instance, seq: list[int]) -> list[float | None]:
    labs: list[float | None] = [None] * len(seq)
    batt_cap = float(instance.vehicle.battery_capacity)
    energy = instance.energy_matrix
    batt = batt_cap
    depot = int(instance.depot_index)
    for i in range(1, len(seq)):
        batt -= float(energy[seq[i - 1], seq[i]])
        if not instance.nodes[seq[i]].is_charging_station:
            continue
        nxt = next(
            (n for n in seq[i + 1:] if instance.nodes[n].is_customer or n == depot),
            depot,
        )
        alpha = just_enough_alpha(instance, seq[i], max(0.0, batt), nxt)
        labs[i] = min(1.0, max(float(alpha), 0.20))
        missing = max(0.0, batt_cap - batt)
        batt = min(batt_cap, batt + labs[i] * missing)
    return labs


def station_candidates(instance, node: int, k: int = 4) -> list[int]:
    dist = instance.distance_matrix
    stations = list(instance.charging_station_indices)
    if not stations:
        return []
    near = sorted(stations, key=lambda s: float(dist[node, s]))[: max(1, k)]
    cheap = sorted(stations, key=lambda s: expected_price(instance, s))[:2]
    fast = sorted(stations, key=lambda s: -charging_rate(instance, s))[:2]
    quiet = sorted(stations, key=lambda s: expected_wait(instance, s))[:2]
    out, seen = [], set()
    for s in near + cheap + fast + quiet:
        if s not in seen:
            seen.add(s)
            out.append(int(s))
    return out[: max(k + 2, 6)]


def _battery_before(instance, nodes: list[int], pos: int) -> float:
    batt = float(instance.vehicle.battery_capacity)
    energy = instance.energy_matrix
    for i in range(1, pos):
        batt -= float(energy[nodes[i - 1], nodes[i]])
        if instance.nodes[nodes[i]].is_charging_station:
            return max(0.0, batt)
    if pos > 0:
        batt -= float(energy[nodes[pos - 1], nodes[pos]])
    return batt


def plan_min_charge_stations(instance, seq: list[int], station_pos: int, next_node: int):
    labs = [None] * len(seq)
    batt = _battery_before(instance, seq, station_pos)
    s = seq[station_pos]
    labs[station_pos] = just_enough_alpha(instance, s, batt, next_node)
    if labs[station_pos] < EPS:
        labs[station_pos] = 0.25
    plan = replay_route(instance, seq, labs)
    if plan.feasible:
        return plan
    labs[station_pos] = 0.5
    plan = replay_route(instance, seq, labs)
    if plan.feasible:
        return plan
    labs[station_pos] = 1.0
    return replay_route(instance, seq, labs)


def _seed_stations(instance, customers: list[int], *, k_stations: int = 6) -> OraclePlan:
    depot = int(instance.depot_index)
    energy = instance.energy_matrix
    batt_cap = float(instance.vehicle.battery_capacity)
    custs = [int(c) for c in customers if instance.nodes[int(c)].is_customer]
    nodes = [depot] + custs + [depot]
    labels: list[float | None] = [None] * len(nodes)
    plan = replay_route(instance, nodes, labels)
    if plan.feasible:
        return _from_replay(plan)
    for _ in range(6):
        batt = batt_cap
        placed = False
        for i in range(1, len(nodes)):
            prev, nxt = nodes[i - 1], nodes[i]
            need = float(energy[prev, nxt])
            if batt + EPS < need and not instance.nodes[prev].is_charging_station:
                best = None
                for s in station_candidates(instance, prev, k_stations):
                    trial = nodes[:i] + [s] + nodes[i:]
                    cand = plan_min_charge_stations(instance, trial, i, nxt)
                    if not cand.feasible:
                        labs = [None] * len(trial)
                        labs[i] = 1.0
                        cand = replay_route(instance, trial, labs)
                    if cand.feasible and (best is None or cand.charging_expense < best.charging_expense):
                        best = cand
                if best is None:
                    return _empty("hco_insert_failed")
                nodes, labels = list(best.nodes), list(best.labels)
                placed = True
                break
            batt -= need
            if instance.nodes[nxt].is_charging_station:
                missing = max(0.0, batt_cap - batt)
                alpha = float(labels[i] or 0.0) if i < len(labels) else 0.0
                batt = min(batt_cap, batt + alpha * missing)
        plan = replay_route(instance, nodes, labels)
        if plan.feasible:
            return _from_replay(plan)
        if not placed:
            return _empty(plan.failure)
    return _empty("hco_budget")


def _replace_stations(instance, nodes: list[int], labels: list[float | None]) -> OraclePlan:
    best = replay_route(instance, nodes, labels)
    best = _from_replay(best) if best is not None and best.feasible else None
    station_pos = [i for i, n in enumerate(nodes) if instance.nodes[n].is_charging_station]
    for pos in station_pos:
        nxt = next(
            (n for n in nodes[pos + 1:] if instance.nodes[n].is_customer or n == instance.depot_index),
            instance.depot_index,
        )
        dropped = nodes[:pos] + nodes[pos + 1:]
        drop_labs = labels[:pos] + labels[pos + 1:]
        cand = _from_replay(replay_route(instance, dropped, drop_labs))
        if cand.feasible and (best is None or cand.charging_expense < best.charging_expense - EPS):
            best = cand
        for s in station_candidates(instance, nodes[pos - 1] if pos else nodes[0], 4):
            trial = list(nodes)
            trial[pos] = s
            plan = plan_min_charge_stations(instance, trial, pos, nxt)
            cand = _from_replay(plan)
            if cand.feasible and (best is None or cand.charging_expense < best.charging_expense - EPS):
                best = cand
    return best if best is not None else _empty("hco_no_plan")


def hco(instance, customers: list[int], cache: OracleCache | None = None) -> OraclePlan:
    """Deterministic charging plan for a fixed customer sequence."""
    key = tuple(int(c) for c in customers)
    if cache is not None:
        cache.calls += 1
        hit = cache.plans.get(key)
        if hit is not None:
            cache.hits += 1
            return hit
    seed = _seed_stations(instance, list(key), k_stations=6)
    if not seed.feasible:
        plan = seed
    else:
        improved = _replace_stations(instance, seed.nodes, seed.labels)
        plan = improved if improved.feasible else seed
    if cache is not None:
        cache.plans[key] = plan
    return plan


def apply_hco_to_routes(instance, routes: list, cache: OracleCache | None = None) -> list:
    out = []
    for nodes, _labs in routes:
        custs = list(customer_sequence(instance, nodes))
        if not custs:
            continue
        plan = hco(instance, custs, cache)
        if not plan.feasible:
            return []
        out.append((list(plan.nodes), list(plan.labels)))
    return out


def solution_from_hco(instance, routes: list, cache: OracleCache | None = None):
    planned = apply_hco_to_routes(instance, routes, cache=cache)
    if not planned:
        return None, replay_solution(instance, [])
    return planned, replay_solution(instance, planned)


# Compatibility names used by archived tests; there is only one HCO.
heterogeneous_charging_oracle = hco
