"""Heterogeneous Charging Oracle: fast screening and selective exact evaluation.

Labels stored on routes are charge portions of the residual battery capacity,
the same convention as ``src.hanif.replay``. Continuous energy is recovered as
e_s = alpha * (B^max - B_arr).
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
    exact: bool = False
    failure: str | None = None


@dataclass
class OracleCache:
    fast: dict = field(default_factory=dict)
    exact: dict = field(default_factory=dict)
    fast_hits: int = 0
    exact_hits: int = 0
    fast_calls: int = 0
    exact_calls: int = 0


def _empty_plan(reason: str | None = None) -> OraclePlan:
    return OraclePlan(
        feasible=False, nodes=[], labels=[], distance=1e18, duration=1e18,
        charging_expense=1e18, charging_time=0.0, station_waiting=0.0,
        charged_energy=0.0, charging_visits=0, failure=reason,
    )


def _from_replay(plan, *, exact: bool = False) -> OraclePlan:
    if plan is None or not plan.feasible:
        return _empty_plan(getattr(plan, "failure", None))
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
        exact=exact,
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
    """Closed-form just-enough portion at every station already in the sequence."""
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


def plan_min_charge_stations(instance, seq: list[int], station_pos: int, next_node: int) -> list:
    """Return (nodes, labels) using the minimum feasible portion at one station."""
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


def greedy_min_charge_route(instance, customers: list[int], *, k_stations: int = 4):
    """Insert nearby min-charge stations until the customer sequence is feasible."""
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
                    return _empty_plan("greedy_insert_failed")
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
            return _empty_plan(plan.failure)
    return _empty_plan("greedy_budget")


def fast_hco(instance, customers: list[int], cache: OracleCache | None = None) -> OraclePlan:
    key = tuple(int(c) for c in customers)
    if cache is not None:
        cache.fast_calls += 1
        hit = cache.fast.get(key)
        if hit is not None:
            cache.fast_hits += 1
            return hit
    plan = greedy_min_charge_route(instance, list(key))
    if cache is not None:
        cache.fast[key] = plan
    return plan


def _replace_stations(instance, nodes: list[int], labels: list[float | None]) -> OraclePlan:
    """Try dropping or swapping each station; keep the cheapest feasible plan."""
    best = replay_route(instance, nodes, labels)
    if not best.feasible:
        best = None
    else:
        best = _from_replay(best, exact=True)
    station_pos = [i for i, n in enumerate(nodes) if instance.nodes[n].is_charging_station]
    for pos in station_pos:
        nxt = next((n for n in nodes[pos + 1:] if instance.nodes[n].is_customer or n == instance.depot_index), instance.depot_index)
        dropped = nodes[:pos] + nodes[pos + 1:]
        drop_labs = labels[:pos] + labels[pos + 1:]
        drop = replay_route(instance, dropped, drop_labs)
        cand = _from_replay(drop, exact=True)
        if cand.feasible and (best is None or cand.charging_expense < best.charging_expense - EPS):
            best = cand
        for s in station_candidates(instance, nodes[pos - 1] if pos else nodes[0], 4):
            trial = list(nodes)
            trial[pos] = s
            plan = plan_min_charge_stations(instance, trial, pos, nxt)
            cand = _from_replay(plan, exact=True)
            if cand.feasible and (best is None or cand.charging_expense < best.charging_expense - EPS):
                best = cand
            labs = list(labels)
            labs[pos] = 1.0
            full = replay_route(instance, trial, labs)
            cand = _from_replay(full, exact=True)
            if cand.feasible and (best is None or cand.charging_expense < best.charging_expense - EPS):
                best = cand
    return best if best is not None else _empty_plan("exact_no_plan")


def exact_hco(instance, customers: list[int], cache: OracleCache | None = None) -> OraclePlan:
    key = tuple(int(c) for c in customers)
    if cache is not None:
        cache.exact_calls += 1
        hit = cache.exact.get(key)
        if hit is not None:
            cache.exact_hits += 1
            return hit
    seed = greedy_min_charge_route(instance, list(key), k_stations=6)
    if not seed.feasible:
        plan = seed
    else:
        plan = _replace_stations(instance, seed.nodes, seed.labels)
        if not plan.feasible:
            plan = seed
        plan.exact = True
    if cache is not None:
        cache.exact[key] = plan
    return plan


def apply_oracle_to_routes(instance, routes: list, *, exact: bool, cache: OracleCache | None = None) -> list:
    out = []
    for nodes, _labs in routes:
        custs = list(customer_sequence(instance, nodes))
        if not custs:
            continue
        plan = exact_hco(instance, custs, cache) if exact else fast_hco(instance, custs, cache)
        if not plan.feasible:
            return []
        out.append((list(plan.nodes), list(plan.labels)))
    return out


def solution_from_oracle(instance, routes: list, *, exact: bool, cache: OracleCache | None = None):
    planned = apply_oracle_to_routes(instance, routes, exact=exact, cache=cache)
    if not planned:
        return None, replay_solution(instance, [])
    return planned, replay_solution(instance, planned)


def promising_for_exact(hat_cost: float, best_cost: float, theta: float) -> bool:
    return hat_cost < (1.0 + float(theta)) * max(best_cost, EPS)
