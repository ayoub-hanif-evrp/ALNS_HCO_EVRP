"""Schneider-scale fleet and charging boosts (100 customers, 21 stations).

Does not change the Hanif 1000-customer paper path. These helpers exist because
the first Schneider run started at 24 vehicles and skipped intensification
(n < 400), so type-2 instances finished about one EV above published BKS.

Type-2 detection uses the planning horizon (Solomon C2/R2/RC2 ≈ 3000+;
type-1 ≈ 1236). Just-enough charging follows Keskin & Çatay (2016): charge
only what is needed to reach the next customer and the nearest recharge.
"""
from __future__ import annotations

from typing import Any

from src.search.charging_dp import optimize_charging_plan
from src.search.repair import _clone_routes
from src.search.route_replay import replay_solution
from src.search.score import SearchScore


TYPE2_HORIZON = 2000.0


def is_long_horizon(instance) -> bool:
    """True for Solomon/Schneider type-2 windows (long horizon)."""
    horizon = float(getattr(instance, "planning_horizon", 0.0) or 0.0)
    if horizon >= TYPE2_HORIZON:
        return True
    dues = [float(n.due_time) for n in instance.nodes if n.is_customer]
    return bool(dues) and max(dues) >= TYPE2_HORIZON


def is_schneider_scale(instance) -> bool:
    n = int(getattr(instance, "num_customers", 0) or 0)
    n_st = len(getattr(instance, "charging_station_indices", []) or [])
    return 40 <= n <= 200 and 5 <= n_st <= 40


def schneider_start_upper(lo: int, hi: int, instance) -> int:
    """First feasible probe. Type-2 must not start near 20–24 vehicles."""
    lo = int(lo)
    hi = int(hi)
    if hi < lo:
        return lo
    if is_long_horizon(instance):
        return int(min(hi, max(lo + 2, min(8, int(round(2.2 * lo)) + 1))))
    n = int(getattr(instance, "num_customers", 0) or 0)
    type1 = max(lo + 2, int((n + 6) // 7))  # ~15 on n=100
    return int(min(hi, max(type1, min(20, lo + 10))))


def just_enough_charge(
    instance,
    station: int,
    battery_on_arrival: float,
    next_customer: int,
    *,
    slack: float = 1.0,
) -> float:
    """Minimum energy to serve next_customer and still reach a recharge/depot."""
    batt_cap = float(instance.vehicle.battery_capacity)
    missing = max(0.0, batt_cap - float(battery_on_arrival))
    if missing <= 1e-12:
        return 0.0
    e_next = float(instance.energy_matrix[station, next_customer])
    depot = int(instance.depot_index)
    onward = [float(instance.energy_matrix[next_customer, depot])]
    for s in getattr(instance, "charging_station_indices", []) or []:
        onward.append(float(instance.energy_matrix[next_customer, int(s)]))
    need = e_next + min(onward) + float(slack)
    charge = min(missing, max(0.0, need - float(battery_on_arrival)))
    return float(charge)


def _route_customers(instance, nodes: list[int]) -> list[int]:
    return [n for n in nodes if instance.nodes[n].is_customer]


def _sequence_variants(instance, ca: list[int], cb: list[int]) -> list[list[int]]:
    """Cheap customer orders for a merged route (concat, TW sort, 2-opt)."""
    seqs = [ca + cb, cb + ca]
    by_ready = sorted(ca + cb, key=lambda n: (float(instance.nodes[n].ready_time), n))
    seqs.append(by_ready)
    depot = int(instance.depot_index)

    def _len(seq: list[int]) -> float:
        if not seq:
            return 0.0
        d = float(instance.distance_matrix[depot, seq[0]])
        for i in range(len(seq) - 1):
            d += float(instance.distance_matrix[seq[i], seq[i + 1]])
        return d + float(instance.distance_matrix[seq[-1], depot])

    best = list(by_ready)
    best_d = _len(best)
    improved = True
    guard = 0
    while improved and guard < 40:
        improved = False
        guard += 1
        for i in range(len(best)):
            for j in range(i + 1, len(best)):
                trial = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                d = _len(trial)
                if d + 1e-6 < best_d:
                    best, best_d, improved = trial, d, True
    seqs.append(best)
    uniq: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for s in seqs:
        key = tuple(s)
        if key not in seen:
            seen.add(key)
            uniq.append(s)
    return uniq


def try_merge_two_routes(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights=None,
    deadline=None,
    max_pairs: int = 12,
) -> tuple[list, SearchScore | None, dict[str, Any]]:
    """Try concatenating two short routes and re-planning charges (type-2 lever)."""
    meta: dict[str, Any] = {"tried": 0, "merged": False}
    cfg = config or {}
    scored: list[tuple[int, int]] = []
    for i, (nodes, _) in enumerate(routes):
        n_c = len(_route_customers(instance, nodes))
        if n_c:
            scored.append((n_c, i))
    scored.sort()
    if len(scored) < 2:
        return routes, None, meta

    pairs: list[tuple[int, int]] = []
    idxs = [i for _, i in scored[:8]]
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            pairs.append((idxs[a], idxs[b]))
    pairs = pairs[: max(1, int(max_pairs))]

    best_routes = None
    best_score: SearchScore | None = None
    for ia, ib in pairs:
        if deadline is not None and deadline.expired():
            break
        meta["tried"] += 1
        ca = _route_customers(instance, routes[ia][0])
        cb = _route_customers(instance, routes[ib][0])
        if not ca or not cb:
            continue
        for seq in _sequence_variants(instance, ca, cb):
            if deadline is not None and deadline.expired():
                break
            plan = optimize_charging_plan(instance, seq, config=cfg, deadline=deadline)
            if not plan.feasible or not plan.nodes:
                continue
            trial = [r for k, r in enumerate(_clone_routes(routes)) if k not in (ia, ib)]
            trial.append((list(plan.nodes), list(plan.labels)))
            used = sum(1 for nodes, _ in trial if _route_customers(instance, nodes))
            check = replay_solution(instance, trial, config=cfg, weights=weights, m=used)
            if check.feasible and check.score is not None:
                if best_score is None or check.score.dominates(best_score):
                    best_routes, best_score = trial, check.score
                    meta["merged"] = True
    if best_routes is None:
        return routes, None, meta
    return best_routes, best_score, meta


def compress_by_merges(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights=None,
    deadline=None,
    max_rounds: int = 6,
    lower_bound: int = 1,
) -> tuple[list, SearchScore | None, dict[str, Any]]:
    """Repeat pairwise merges until no merge is feasible or the LB is hit."""
    current = _clone_routes(routes)
    score: SearchScore | None = None
    log = {"rounds": 0, "merges": 0, "tried": 0}
    for _ in range(max(1, int(max_rounds))):
        used = sum(1 for nodes, _ in current if _route_customers(instance, nodes))
        if used <= int(lower_bound):
            break
        if deadline is not None and deadline.remaining() < 2.0:
            break
        new_routes, new_score, meta = try_merge_two_routes(
            instance, current, config=config, weights=weights, deadline=deadline,
        )
        log["tried"] += int(meta.get("tried") or 0)
        log["rounds"] += 1
        if not meta.get("merged") or new_score is None:
            break
        current, score = new_routes, new_score
        log["merges"] += 1
    return current, score, log


def _tw_optimistic_ok(instance, seq: list[int]) -> bool:
    """Cheap TW prune that ignores charging time (optimistic)."""
    depot = int(instance.depot_index)
    t = 0.0
    prev = depot
    tt = instance.travel_time_matrix
    for c in seq:
        t += float(tt[prev, c])
        node = instance.nodes[c]
        t = max(t, float(node.ready_time))
        if t > float(node.due_time) + 1e-9:
            return False
        t += float(node.service_time)
        prev = c
    horizon = float(getattr(instance, "planning_horizon", 1e9) or 1e9)
    return t + float(tt[prev, depot]) <= horizon + 1e-9


def _seq_load(instance, seq: list[int]) -> float:
    return sum(float(instance.nodes[n].demand) for n in seq)


def try_absorb_shortest_route(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights=None,
    deadline=None,
    max_hosts: int = 8,
    max_victims: int = 8,
) -> tuple[list, SearchScore | None, dict[str, Any]]:
    """Move every customer of the shortest route onto other routes (type-1 lever)."""
    meta: dict[str, Any] = {"tried": 0, "placed": 0, "absorbed": False}
    cfg = config or {}
    indexed = [
        (i, _route_customers(instance, nodes))
        for i, (nodes, _) in enumerate(routes)
    ]
    used = [(i, custs) for i, custs in indexed if custs]
    if len(used) < 2:
        return routes, None, meta
    used.sort(key=lambda item: (len(item[1]), item[0]))
    victim_i, victims = used[0]
    if not victims or len(victims) > int(max_victims):
        return routes, None, meta
    hosts = used[1 : 1 + max(1, int(max_hosts))]
    host_seqs = {i: list(custs) for i, custs in hosts}
    cap = float(instance.vehicle.load_capacity or 1e18)

    for cust in victims:
        if deadline is not None and deadline.expired():
            return routes, None, meta
        meta["tried"] += 1
        demand = float(instance.nodes[cust].demand)
        best: tuple[int, list[int], float] | None = None
        for hi, _orig in hosts:
            seq = host_seqs[hi]
            if _seq_load(instance, seq) + demand > cap + 1e-9:
                continue
            for pos in range(len(seq) + 1):
                if deadline is not None and deadline.expired():
                    return routes, None, meta
                trial_seq = seq[:pos] + [cust] + seq[pos:]
                if not _tw_optimistic_ok(instance, trial_seq):
                    continue
                plan = optimize_charging_plan(instance, trial_seq, config=cfg, deadline=deadline)
                if not plan.feasible or not plan.nodes:
                    continue
                key = float(plan.route_distance)
                if best is None or key + 1e-9 < best[2]:
                    best = (hi, trial_seq, key)
        if best is None:
            return routes, None, meta
        host_seqs[best[0]] = best[1]
        meta["placed"] += 1

    trial = []
    for i, (nodes, labels) in enumerate(routes):
        if i == victim_i:
            continue
        if i in host_seqs:
            plan = optimize_charging_plan(instance, host_seqs[i], config=cfg, deadline=deadline)
            if not plan.feasible or not plan.nodes:
                return routes, None, meta
            trial.append((list(plan.nodes), list(plan.labels)))
        else:
            trial.append((list(nodes), list(labels)))
    used_n = sum(1 for nodes, _ in trial if _route_customers(instance, nodes))
    check = replay_solution(instance, trial, config=cfg, weights=weights, m=used_n)
    if check.feasible and check.score is not None:
        meta["absorbed"] = True
        return trial, check.score, meta
    return routes, None, meta


def compress_by_absorb(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights=None,
    deadline=None,
    max_rounds: int = 4,
    lower_bound: int = 1,
) -> tuple[list, SearchScore | None, dict[str, Any]]:
    """Repeat shortest-route absorption until stuck or the LB is hit."""
    current = _clone_routes(routes)
    score: SearchScore | None = None
    log = {"rounds": 0, "absorbed": 0, "tried": 0, "placed": 0}
    for _ in range(max(1, int(max_rounds))):
        used = sum(1 for nodes, _ in current if _route_customers(instance, nodes))
        if used <= int(lower_bound):
            break
        if deadline is not None and deadline.remaining() < 3.0:
            break
        new_routes, new_score, meta = try_absorb_shortest_route(
            instance, current, config=config, weights=weights, deadline=deadline,
        )
        log["tried"] += int(meta.get("tried") or 0)
        log["placed"] += int(meta.get("placed") or 0)
        log["rounds"] += 1
        if not meta.get("absorbed") or new_score is None:
            break
        current, score = new_routes, new_score
        log["absorbed"] += 1
    return current, score, log


def squeeze_fleet(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights=None,
    deadline=None,
    lower_bound: int = 1,
    allow_merge: bool = True,
) -> tuple[list, SearchScore | None, dict[str, Any]]:
    """Absorb short routes, then merge, until neither operator shrinks K."""
    current = _clone_routes(routes)
    score: SearchScore | None = None
    log: dict[str, Any] = {"absorb": {}, "merge": {}}
    current, score, log["absorb"] = compress_by_absorb(
        instance, current, config=config, weights=weights, deadline=deadline,
        max_rounds=5, lower_bound=lower_bound,
    )
    if allow_merge:
        merged, mscore, mlog = compress_by_merges(
            instance, current, config=config, weights=weights, deadline=deadline,
            max_rounds=6, lower_bound=lower_bound,
        )
        log["merge"] = mlog
        if mscore is not None:
            current, score = merged, mscore
    return current, score, log
