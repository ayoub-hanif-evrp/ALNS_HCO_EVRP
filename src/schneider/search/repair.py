"""Exact feasible insertion repair for TG-FPS."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

from src.schneider.env.cost import CostWeights
from src.schneider.search.route_replay import replay_solution
from src.schneider.search.score import SearchScore


@dataclass
class InsertionCandidate:
    route_index: int
    position: int
    customer: int
    station_before: int | None
    station_after: int | None
    charge_before: float | None
    charge_after: float | None
    score: SearchScore
    routes: list[tuple[list[int], list[float | None]]]


def _customer_nodes(instance) -> list[int]:
    return [i for i, n in enumerate(instance.nodes) if n.is_customer]


def _station_nodes(instance) -> list[int]:
    return [i for i, n in enumerate(instance.nodes) if n.is_charging_station]


def _nearest_stations(instance, customer: int, k: int = 5) -> list[int]:
    stations = _station_nodes(instance)
    if not stations:
        return []
    dist = instance.distance_matrix
    ranked = sorted(stations, key=lambda s: float(dist[customer, s]) + float(dist[s, customer]))
    return ranked[: max(1, k)]


def _served_in_routes(instance, routes: list[tuple[list[int], list[float | None]]]) -> set[int]:
    served: set[int] = set()
    for nodes, _ in routes:
        for n in nodes:
            if instance.nodes[n].is_customer:
                served.add(int(n))
    return served


def _clone_routes(
    routes: list[tuple[list[int], list[float | None]]],
) -> list[tuple[list[int], list[float | None]]]:
    return [(list(nodes), list(labels)) for nodes, labels in routes]


def _insert_into_route(
    nodes: list[int],
    labels: list[float | None],
    position: int,
    customer: int,
    *,
    station_before: int | None = None,
    station_after: int | None = None,
    charge_before: float | None = 1.0,
    charge_after: float | None = 1.0,
) -> tuple[list[int], list[float | None]]:
    """Insert customer (and optional stations) at position (index in nodes list)."""
    new_nodes = list(nodes)
    new_labels = list(labels) if labels else [None] * len(nodes)
    if len(new_labels) < len(new_nodes):
        new_labels = new_labels + [None] * (len(new_nodes) - len(new_labels))
    insert_at = max(1, min(position, len(new_nodes) - 1))
    chunk_nodes: list[int] = []
    chunk_labels: list[float | None] = []
    if station_before is not None:
        chunk_nodes.append(int(station_before))
        chunk_labels.append(float(charge_before) if charge_before is not None else 1.0)
    chunk_nodes.append(int(customer))
    chunk_labels.append(None)
    if station_after is not None:
        chunk_nodes.append(int(station_after))
        chunk_labels.append(float(charge_after) if charge_after is not None else 1.0)
    new_nodes[insert_at:insert_at] = chunk_nodes
    new_labels[insert_at:insert_at] = chunk_labels
    return new_nodes, new_labels


def evaluate_insertion(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    route_index: int,
    position: int,
    customer: int,
    *,
    station_before: int | None = None,
    station_after: int | None = None,
    charge_before: float | None = 1.0,
    charge_after: float | None = 1.0,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    deadline=None,
    fast_route_only: bool = False,
) -> InsertionCandidate | None:
    from src.schneider.search.incremental import capacity_precheck, energy_precheck
    from src.schneider.search.route_replay import replay_route

    if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
        return None

    trial = _clone_routes(routes)
    if route_index >= len(trial):
        depot = instance.depot_index
        trial.append(([depot, depot], [None, None]))
        route_index = len(trial) - 1
    nodes, labels = trial[route_index]
    # Cheap necessary filters before any replay.
    if not capacity_precheck(instance, nodes, customer):
        return None
    prev = nodes[position - 1] if position - 1 < len(nodes) else instance.depot_index
    nxt = nodes[position] if position < len(nodes) else instance.depot_index
    if station_before is not None:
        if not energy_precheck(instance, prev, station_before):
            return None
        if not energy_precheck(instance, station_before, customer):
            return None
    elif not energy_precheck(instance, prev, customer):
        return None
    if station_after is not None:
        if not energy_precheck(instance, customer, station_after):
            return None
        if not energy_precheck(instance, station_after, nxt):
            return None
    elif not energy_precheck(instance, customer, nxt):
        return None

    new_nodes, new_labels = _insert_into_route(
        nodes,
        labels,
        position,
        customer,
        station_before=station_before,
        station_after=station_after,
        charge_before=charge_before,
        charge_after=charge_after,
    )
    # Fast path: exact affected-route replay first (orders of magnitude cheaper).
    route_res = replay_route(instance, new_nodes, new_labels, config=config)
    if not route_res.feasible or customer not in route_res.served_customers:
        return None
    trial[route_index] = (new_nodes, new_labels)

    if fast_route_only:
        # Build a provisional score from route metrics + unchanged siblings (approx).
        # Full solution replay is deferred to caller acceptance.
        from src.schneider.search.score import SearchScore
        dist = float(route_res.distance)
        dur = float(route_res.duration)
        chg = float(route_res.charging_time)
        served = set()
        for nds, _ in trial:
            for n in nds:
                if instance.nodes[n].is_customer:
                    served.add(int(n))
        n_cust = int(instance.num_customers)
        unserved = max(0, n_cust - len(served))
        used = sum(1 for nds, _ in trial if any(instance.nodes[n].is_customer for n in nds))
        from src.schneider.evaluation.paper_objective import MODE_CLASSICAL, objective_spec, phi_from_replay
        spec = objective_spec(config)
        fake = type("R", (), {"route_results": [route_res], "distance": dist})()
        phi_info = phi_from_replay(instance, fake, config)
        score = SearchScore(
            hard_violation_count=0,
            unserved_customer_count=unserved,
            used_evs=used,
            total_distance=dist,
            total_charging_time=chg,
            total_duration=dur,
            paper_objective=float(phi_info["Phi"]) if spec["mode"] != MODE_CLASSICAL else dist,
            phi=float(phi_info["Phi"]),
            total_station_time=float(phi_info["tstation"]),
            objective_mode=str(spec["mode"]),
        )
        return InsertionCandidate(
            route_index=route_index,
            position=position,
            customer=customer,
            station_before=station_before,
            station_after=station_after,
            charge_before=charge_before,
            charge_after=charge_after,
            score=score,
            routes=trial,
        )

    if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
        return None
    result = replay_solution(instance, trial, config=config, weights=weights, m=m or max(len(trial), 1))
    if result.score is None:
        return None
    if customer not in result.served_customers:
        return None
    return InsertionCandidate(
        route_index=route_index,
        position=position,
        customer=customer,
        station_before=station_before,
        station_after=station_after,
        charge_before=charge_before,
        charge_after=charge_after,
        score=result.score,
        routes=trial,
    )


def enumerate_insertions_for_customer(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    customer: int,
    *,
    station_candidates: int = 5,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    allow_new_route: bool = True,
    deadline=None,
    max_full_replays: int = 8,
) -> list[InsertionCandidate]:
    """Enumerate bounded insertion candidates; screen with route replay, full-replay top-K."""
    cands_fast: list[InsertionCandidate] = []
    rem = float("inf") if deadline is None else float(deadline.remaining())
    # Tighten search under time pressure.
    st_k = int(station_candidates)
    if rem < 8:
        st_k = 0
    elif rem < 20:
        st_k = min(st_k, 1)
    elif rem < 40:
        st_k = min(st_k, 2)

    stations = _nearest_stations(instance, customer, k=max(0, st_k)) if st_k > 0 else []
    route_list = list(routes)
    if allow_new_route:
        depot = instance.depot_index
        route_list = route_list + [([depot, depot], [None, None])]

    # Prefer routes with residual capacity / nearer centroid first.
    cx = float(instance.nodes[customer].x)
    cy = float(instance.nodes[customer].y)
    ranked_routes: list[tuple[float, int, list[int], list]] = []
    for ri, (nodes, labels) in enumerate(route_list):
        load = sum(float(instance.nodes[n].demand) for n in nodes if instance.nodes[n].is_customer)
        cap = float(instance.vehicle.load_capacity)
        if float(instance.nodes[customer].demand) > cap - load + 1e-9 and ri < len(routes):
            continue
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        if custs:
            rcx = sum(float(instance.nodes[n].x) for n in custs) / len(custs)
            rcy = sum(float(instance.nodes[n].y) for n in custs) / len(custs)
            dist = (rcx - cx) ** 2 + (rcy - cy) ** 2
        else:
            dist = 0.0
        ranked_routes.append((dist, ri, nodes, labels))
    ranked_routes.sort(key=lambda t: t[0])
    max_routes = len(ranked_routes)
    if rem < 15:
        max_routes = min(max_routes, 6)
    elif rem < 40:
        max_routes = min(max_routes, 12)
    ranked_routes = ranked_routes[:max_routes]

    station_options: list[tuple[int | None, int | None]] = [(None, None)]
    for s in stations[:st_k]:
        station_options.append((s, None))
        station_options.append((None, s))

    evals = 0
    max_evals = 400
    if rem < 10:
        max_evals = 40
    elif rem < 25:
        max_evals = 120
    elif rem < 50:
        max_evals = 220

    for _dist, ri, nodes, _labels in ranked_routes:
        if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
            break
        if evals >= max_evals:
            break
        positions = list(range(1, max(len(nodes), 2)))
        # Cap positions under pressure: try ends + a few mid samples
        if rem < 20 and len(positions) > 6:
            positions = [positions[0], positions[len(positions) // 2], positions[-1]] + positions[1:4]
            positions = sorted(set(positions))
        for pos in positions:
            if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
                break
            if evals >= max_evals:
                break
            for sb, sa in station_options:
                evals += 1
                cand = evaluate_insertion(
                    instance,
                    routes if ri < len(routes) else routes,
                    ri if ri < len(routes) else len(routes),
                    pos,
                    customer,
                    station_before=sb,
                    station_after=sa,
                    config=config,
                    weights=weights,
                    m=m,
                    deadline=deadline,
                    fast_route_only=True,
                )
                if cand is not None:
                    cands_fast.append(cand)

    # Promote top fast candidates with full solution replay
    cands_fast.sort(key=lambda c: c.score.as_tuple())
    promoted: list[InsertionCandidate] = []
    for cand in cands_fast[: max(1, max_full_replays)]:
        if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
            break
        full = evaluate_insertion(
            instance,
            routes,
            cand.route_index,
            cand.position,
            customer,
            station_before=cand.station_before,
            station_after=cand.station_after,
            charge_before=cand.charge_before,
            charge_after=cand.charge_after,
            config=config,
            weights=weights,
            m=m,
            deadline=deadline,
            fast_route_only=False,
        )
        if full is not None:
            promoted.append(full)
    if not promoted and cands_fast:
        # Fallback: keep best fast candidate (caller will full-replay on accept via repair loop)
        promoted = [cands_fast[0]]
    return sorted(promoted, key=lambda c: c.score.as_tuple())


def best_insertion(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    customer: int,
    **kwargs,
) -> InsertionCandidate | None:
    cands = enumerate_insertions_for_customer(instance, routes, customer, **kwargs)
    return cands[0] if cands else None


def regret_insertion(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    unserved: list[int],
    *,
    regret_k: int = 2,
    deadline=None,
    **kwargs,
) -> InsertionCandidate | None:
    """Choose customer with largest regret between best and k-th best insertion."""
    best_overall: InsertionCandidate | None = None
    best_regret = -1.0
    for cust in unserved:
        if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
            break
        cands = enumerate_insertions_for_customer(
            instance, routes, cust, deadline=deadline, **kwargs
        )
        if not cands:
            continue
        if len(cands) == 1:
            regret = 0.0
        else:
            k = min(max(regret_k, 2), len(cands))
            best_j = cands[0].score.operational_scalar()
            kth_j = cands[k - 1].score.operational_scalar()
            regret = float(kth_j - best_j)
        if regret > best_regret or (
            abs(regret - best_regret) < 1e-9
            and (best_overall is None or cands[0].score.dominates(best_overall.score))
        ):
            best_regret = regret
            best_overall = cands[0]
    return best_overall


def critical_slack_order(instance, unserved: list[int]) -> list[int]:
    """Order unserved by earliest due time (observable critical slack proxy)."""
    return sorted(unserved, key=lambda j: (float(instance.nodes[j].due_time), j))


def repair_solution(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    max_rounds: int = 10,
    regret_k: int = 2,
    station_candidates: int = 5,
    mode: str = "regret",
    deadline=None,
) -> tuple[list[tuple[list[int], list[float | None]]], SolutionReplayLike]:
    """Iteratively insert unserved customers with exact-replay validation.

    mode: 'best' | 'regret' | 'critical'
    """
    from src.schneider.search.route_replay import SolutionReplayResult

    current = _clone_routes(routes)
    last: SolutionReplayResult | None = None
    for _ in range(max(1, max_rounds)):
        if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
            break
        last = replay_solution(instance, current, config=config, weights=weights, m=m)
        served = set(last.served_customers)
        unserved = [j for j in _customer_nodes(instance) if j not in served]
        if not unserved:
            break
        if mode == "critical":
            ordered = critical_slack_order(instance, unserved)
            pick = best_insertion(
                instance,
                current,
                ordered[0],
                station_candidates=station_candidates,
                config=config,
                weights=weights,
                m=m,
            )
        elif mode == "best":
            # Greedy: try all customers, take globally best insertion.
            pick = None
            for cust in unserved:
                if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
                    break
                cand = best_insertion(
                    instance,
                    current,
                    cust,
                    station_candidates=station_candidates,
                    config=config,
                    weights=weights,
                    m=m,
                )
                if cand is None:
                    continue
                if pick is None or cand.score.dominates(pick.score):
                    pick = cand
        else:
            pick = regret_insertion(
                instance,
                current,
                unserved,
                regret_k=regret_k,
                station_candidates=station_candidates,
                config=config,
                weights=weights,
                m=m,
                deadline=deadline,
            )
        if pick is None:
            break
        # Always confirm acceptance with a full exact replay of the chosen trial.
        confirmed = replay_solution(instance, pick.routes, config=config, weights=weights, m=m)
        if confirmed.score is None or pick.customer not in confirmed.served_customers:
            break
        pick_score = confirmed.score
        if last.score is not None and not (
            pick_score.unserved_customer_count < last.score.unserved_customer_count
            or (
                pick_score.unserved_customer_count == last.score.unserved_customer_count
                and pick_score.dominates(last.score)
            )
        ):
            if pick_score.unserved_customer_count >= last.unserved_customers:
                break
        current = pick.routes
        last = confirmed
    if last is None:
        last = replay_solution(instance, current, config=config, weights=weights, m=m)
    else:
        last = replay_solution(instance, current, config=config, weights=weights, m=m)
    return current, last


# Typing alias for callers
SolutionReplayLike = Any


def remove_station_if_redundant(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
) -> list[tuple[list[int], list[float | None]]]:
    """Try removing each station visit; keep if exact replay score improves."""
    current = _clone_routes(routes)
    base = replay_solution(instance, current, config=config, weights=weights, m=m)
    if base.score is None:
        return current
    improved = True
    while improved:
        improved = False
        for ri, (nodes, labels) in enumerate(current):
            for pos in range(1, len(nodes) - 1):
                if not instance.nodes[nodes[pos]].is_charging_station:
                    continue
                trial = _clone_routes(current)
                n2 = list(trial[ri][0])
                l2 = list(trial[ri][1])
                del n2[pos]
                if pos < len(l2):
                    del l2[pos]
                trial[ri] = (n2, l2)
                res = replay_solution(instance, trial, config=config, weights=weights, m=m)
                if res.score is not None and res.score.dominates(base.score):
                    current = trial
                    base = res
                    improved = True
                    break
            if improved:
                break
    return current


def relabel_charge_portions(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    portions: list[float] | None = None,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
) -> list[tuple[list[int], list[float | None]]]:
    """Try alternate charge portions on station visits; keep lex improvements."""
    opts = portions or [0.25, 0.5, 0.75, 1.0]
    current = _clone_routes(routes)
    base = replay_solution(instance, current, config=config, weights=weights, m=m)
    if base.score is None:
        return current
    for ri, (nodes, labels) in enumerate(current):
        labels = list(labels) if labels else [None] * len(nodes)
        for pos, node in enumerate(nodes):
            if not instance.nodes[node].is_charging_station:
                continue
            for p in opts:
                trial = _clone_routes(current)
                n2, l2 = trial[ri]
                l2 = list(l2) if l2 else [None] * len(n2)
                if len(l2) < len(n2):
                    l2 = l2 + [None] * (len(n2) - len(l2))
                l2[pos] = float(p)
                trial[ri] = (list(n2), l2)
                res = replay_solution(instance, trial, config=config, weights=weights, m=m)
                if res.score is not None and res.score.dominates(base.score):
                    current = trial
                    base = res
                    labels = l2
    return current
