"""Fast construction helpers for budget-limited fleet search."""
from __future__ import annotations

from src.schneider.env.cost import CostWeights
from src.schneider.env.evrptw_env import EVRPTWEnv
from src.schneider.env.policy_action import PolicyAction, apply_policy_action
from src.schneider.search.deadline import SearchDeadline
from src.schneider.search.repair import (
    best_insertion,
    critical_slack_order,
    regret_insertion,
    repair_solution,
)
from src.schneider.search.route_replay import extract_routes_from_env, replay_solution
from src.schneider.search.score import SearchScore


def dense_construct(
    instance,
    fleet_size: int,
    *,
    seed: int = 0,
    config: dict | None = None,
    weights: CostWeights | None = None,
    deadline: SearchDeadline | None = None,
    variant: str = "edf",
) -> tuple[list, SearchScore, bool]:
    """Sequential insertion into a fixed fleet.

    n < 400: exact best-insertion (Schneider-scale).
    n >= 400: Hanif-scale nearest-route fast insert (route-local replay only).
    """
    if int(getattr(instance, "num_customers", 0) or 0) >= 400:
        return _dense_construct_large(
            instance, fleet_size, seed=seed, config=config, weights=weights,
            deadline=deadline, variant=variant,
        )
    return _dense_construct_exact(
        instance, fleet_size, seed=seed, config=config, weights=weights,
        deadline=deadline, variant=variant,
    )


def _recover_unserved_fast(
    instance,
    routes: list,
    unserved: list[int],
    *,
    m: int,
    config: dict,
    weights: CostWeights,
    deadline: SearchDeadline | None = None,
) -> tuple[list, list[int]]:
    """Place leftovers with nearest-station inserts (route-local replay only)."""
    from src.schneider.search.incremental import capacity_precheck
    from src.schneider.search.repair import _nearest_stations, evaluate_insertion

    dist = instance.distance_matrix
    depot = int(instance.depot_index)
    remaining: list[int] = []
    for ui, cust in enumerate(unserved):
        if deadline is not None and deadline.expired():
            remaining.extend(unserved[ui:])
            break
        cx = float(instance.nodes[cust].x)
        cy = float(instance.nodes[cust].y)
        ranked: list[tuple[float, int]] = []
        for ri, (nodes, _) in enumerate(routes):
            if not capacity_precheck(instance, nodes, cust):
                continue
            custs = [n for n in nodes if instance.nodes[n].is_customer]
            if custs:
                rcx = sum(float(instance.nodes[n].x) for n in custs) / len(custs)
                rcy = sum(float(instance.nodes[n].y) for n in custs) / len(custs)
                d = (rcx - cx) ** 2 + (rcy - cy) ** 2
            else:
                d = float(dist[depot, cust])
            ranked.append((d, ri))
        ranked.sort()
        stations = [None] + _nearest_stations(instance, cust, k=3)
        placed = False
        nonempty = [t for t in ranked if any(instance.nodes[n].is_customer for n in routes[t[1]][0])]
        empties = [t for t in ranked if not any(instance.nodes[n].is_customer for n in routes[t[1]][0])]
        # Try packed routes first, then open empty trips for leftovers.
        try_order = nonempty[:8] + empties[:3]
        for _, ri in try_order:
            nodes = routes[ri][0]
            positions = [len(nodes) - 1]
            if len(nodes) > 2:
                best_j = min(
                    range(1, len(nodes) - 1),
                    key=lambda j: float(dist[nodes[j], cust])
                    if instance.nodes[nodes[j]].is_customer
                    else 1e18,
                )
                positions.extend([best_j, min(len(nodes) - 1, best_j + 1)])
            for pos in list(dict.fromkeys(positions)):
                for st in stations:
                    for charge in ((1.0,) if st is not None else (None,)):
                        cand = evaluate_insertion(
                            instance, routes, ri, pos, cust,
                            station_before=st,
                            charge_before=charge,
                            config=config, weights=weights, m=m, deadline=deadline,
                            fast_route_only=True,
                        )
                        if cand is not None:
                            routes = cand.routes
                            placed = True
                            break
                    if placed:
                        break
                    # Also try station after customer (return energy).
                    if st is not None:
                        cand = evaluate_insertion(
                            instance, routes, ri, pos, cust,
                            station_after=st,
                            charge_after=1.0,
                            config=config, weights=weights, m=m, deadline=deadline,
                            fast_route_only=True,
                        )
                        if cand is not None:
                            routes = cand.routes
                            placed = True
                            break
                if placed:
                    break
            if placed:
                break
        if not placed:
            remaining.append(cust)
    return routes, remaining


def _dense_construct_large(
    instance,
    fleet_size: int,
    *,
    seed: int = 0,
    config: dict | None = None,
    weights: CostWeights | None = None,
    deadline: SearchDeadline | None = None,
    variant: str = "edf",
) -> tuple[list, SearchScore, bool]:
    """O(n·k) construction for 1000-customer Hanif instances."""
    from src.schneider.search.incremental import capacity_precheck
    from src.schneider.search.repair import evaluate_insertion

    cfg = config or {}
    w = weights or CostWeights()
    m = max(1, int(fleet_size))
    depot = int(instance.depot_index)
    routes: list = [([depot, depot], [None, None]) for _ in range(m)]
    customers = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    ordered = critical_slack_order(instance, customers)
    salt = int(seed) % 7
    if salt and len(ordered) > 20:
        ordered = ordered[salt:] + ordered[:salt]

    dist = instance.distance_matrix
    unserved: list[int] = []
    for ci, cust in enumerate(ordered):
        if deadline is not None and deadline.expired():
            unserved.extend(ordered[ci:])
            break
        cx = float(instance.nodes[cust].x)
        cy = float(instance.nodes[cust].y)
        # Rank routes: prefer nonempty with residual capacity & nearby centroid.
        ranked: list[tuple[float, int]] = []
        for ri, (nodes, _) in enumerate(routes):
            if not capacity_precheck(instance, nodes, cust):
                continue
            custs = [n for n in nodes if instance.nodes[n].is_customer]
            if custs:
                rcx = sum(float(instance.nodes[n].x) for n in custs) / len(custs)
                rcy = sum(float(instance.nodes[n].y) for n in custs) / len(custs)
                d = (rcx - cx) ** 2 + (rcy - cy) ** 2
            else:
                d = float(dist[depot, cust])  # empty route: open from depot
            ranked.append((d, ri))
        ranked.sort()
        # Mix nearby packed routes with empty slots (need empties early when fleet is open).
        empties = [t for t in ranked if not any(instance.nodes[n].is_customer for n in routes[t[1]][0])]
        nonempty = [t for t in ranked if any(instance.nodes[n].is_customer for n in routes[t[1]][0])]
        n_keep = 8
        n_non = min(6, len(nonempty))
        ranked = nonempty[:n_non] + empties[: max(2, n_keep - n_non)]
        placed = False
        for _, ri in ranked:
            nodes, labels = routes[ri]
            positions = [len(nodes) - 1]
            if len(nodes) > 2:
                best_j = min(
                    range(1, len(nodes) - 1),
                    key=lambda j: float(dist[nodes[j], cust]) if instance.nodes[nodes[j]].is_customer
                    else 1e18,
                )
                positions.append(best_j)
                positions.append(min(len(nodes) - 1, best_j + 1))
            positions = list(dict.fromkeys(positions))
            best_cand = None
            for pos in positions:
                for st_before, st_after in ((None, None),):
                    cand = evaluate_insertion(
                        instance, routes, ri, pos, cust,
                        station_before=st_before, station_after=st_after,
                        config=cfg, weights=w, m=m, deadline=deadline,
                        fast_route_only=True,
                    )
                    if cand is None:
                        continue
                    if best_cand is None or cand.score.dominates(best_cand.score):
                        best_cand = cand
            if best_cand is not None:
                routes = best_cand.routes
                placed = True
                break
        # Energy-blocked: try a few nearest stations before customer.
        if not placed and ranked:
            from src.schneider.search.repair import _nearest_stations
            for near_st in _nearest_stations(instance, cust, k=3):
                for _, ri in ranked[:4]:
                    nodes = routes[ri][0]
                    for pos in (len(nodes) - 1, max(1, len(nodes) // 2)):
                        cand = evaluate_insertion(
                            instance, routes, ri, pos, cust,
                            station_before=near_st, charge_before=1.0,
                            config=cfg, weights=w, m=m, deadline=deadline,
                            fast_route_only=True,
                        )
                        if cand is not None:
                            routes = cand.routes
                            placed = True
                            break
                    if placed:
                        break
                if placed:
                    break
        if not placed:
            unserved.append(cust)

    # Leftover recovery with stations (Hanif energy/TW blockers).
    if unserved and (deadline is None or deadline.remaining() > 3):
        routes, unserved = _recover_unserved_fast(
            instance, routes, unserved, m=m, config=cfg, weights=w, deadline=deadline,
        )

    rounds = 4 if int(getattr(instance, "num_customers", 0) or 0) >= 400 else 6
    st_k = 1 if int(getattr(instance, "num_customers", 0) or 0) >= 400 else 2
    if deadline is not None and deadline.remaining() < 20:
        rounds = 2
        st_k = 0
    if deadline is not None and deadline.remaining() < 8:
        rounds = 1
        st_k = 0
    routes, res = repair_solution(
        instance, routes, config=cfg, weights=w, m=m,
        max_rounds=rounds, mode="regret", regret_k=2, station_candidates=st_k,
        deadline=deadline,
    )
    score = res.score
    if score is None:
        final = replay_solution(instance, routes, config=cfg, weights=w, m=m)
        score = final.score or SearchScore.worst()
        return routes, score, bool(final.feasible)
    return routes, score, bool(res.feasible)


def _dense_construct_exact(
    instance,
    fleet_size: int,
    *,
    seed: int = 0,
    config: dict | None = None,
    weights: CostWeights | None = None,
    deadline: SearchDeadline | None = None,
    variant: str = "edf",
) -> tuple[list, SearchScore, bool]:
    """Exact sequential insertion for ≤~200 customer instances."""
    cfg = config or {}
    w = weights or CostWeights()
    m = max(1, int(fleet_size))
    depot = int(instance.depot_index)
    routes: list = [([depot, depot], [None, None]) for _ in range(m)]

    customers = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    if variant in ("edf", "hybrid", "fleet"):
        ordered = critical_slack_order(instance, customers)
    elif variant == "energy":
        ordered = sorted(
            customers,
            key=lambda j: (
                float(instance.energy_matrix[depot, j]),
                float(instance.nodes[j].due_time),
            ),
        )
    else:
        ordered = sorted(
            customers,
            key=lambda j: (
                float(instance.distance_matrix[depot, j]),
                float(instance.nodes[j].due_time),
            ),
        )
    salt = int(seed) % max(1, min(5, len(ordered) // 10 + 1))
    if salt and len(ordered) > salt + 1:
        ordered = ordered[salt:] + ordered[:salt]

    st_k = 1
    if deadline is not None and deadline.remaining() < 20:
        st_k = 0
    elif deadline is not None and deadline.remaining() > 60:
        st_k = 2

    unserved: list[int] = []
    for idx, cust in enumerate(ordered):
        if deadline is not None and deadline.expired():
            unserved.extend(ordered[idx:])
            break
        cand = best_insertion(
            instance, routes, cust, station_candidates=st_k, config=cfg, weights=w, m=m,
            allow_new_route=False, deadline=deadline, max_full_replays=4,
        )
        if cand is None and st_k > 0:
            cand = best_insertion(
                instance, routes, cust, station_candidates=0, config=cfg, weights=w, m=m,
                allow_new_route=False, deadline=deadline, max_full_replays=4,
            )
        if cand is not None:
            routes = cand.routes
        else:
            unserved.append(cust)

    if unserved and (deadline is None or deadline.remaining() > 3):
        for _ in range(max(2, len(unserved))):
            if deadline is not None and deadline.expired():
                break
            if not unserved:
                break
            pick = regret_insertion(
                instance, routes, unserved, regret_k=2, station_candidates=st_k,
                config=cfg, weights=w, m=m, deadline=deadline,
            )
            if pick is None:
                break
            routes = pick.routes
            unserved = [c for c in unserved if c != pick.customer]

    rounds = 16
    if deadline is not None:
        rem = deadline.remaining()
        if rem < 5:
            rounds = 3
        elif rem < 15:
            rounds = 8
        elif rem < 30:
            rounds = 12

    routes, res = repair_solution(
        instance, routes, config=cfg, weights=w, m=m, max_rounds=rounds,
        mode="regret", regret_k=2, station_candidates=max(1, st_k), deadline=deadline,
    )
    score = res.score
    if score is None:
        final = replay_solution(instance, routes, config=cfg, weights=w, m=m)
        score = final.score or SearchScore.worst()
        return routes, score, bool(final.feasible)
    return routes, score, bool(res.feasible)



def matrix_construct(
    instance,
    fleet_size: int,
    *,
    seed: int = 0,
    config: dict | None = None,
    weights: CostWeights | None = None,
    deadline: SearchDeadline | None = None,
    variant: str = "edf",
) -> tuple[list, SearchScore, bool]:
    """Greedy env construction (EDF/nearest) + short regret repair."""
    cfg = config or {}
    problem = cfg.get("problem") or {}
    portions = list(problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    w = weights or CostWeights()
    env = EVRPTWEnv(
        instance,
        m=int(fleet_size),
        charging_portions=portions,
        use_horizon_check=bool(problem.get("use_horizon_check", True)),
        use_depot_charging=bool(problem.get("use_depot_charging", True)),
        max_steps_factor=3,
        hard_time_windows=True,
    )
    steps = 0
    salt = int(seed) * 17 + hash(variant) % 7

    while not env.is_done() and steps < env.max_steps:
        if deadline is not None and deadline.expired():
            break
        active = [e for e in env.evs if not e.finished]
        if not active:
            break
        ev = active[(steps + salt) % len(active)]
        analysis = env.analyze_decision(ev)
        mask = analysis.final_node_mask
        if mask is None:
            steps += 1
            continue
        depot = instance.depot_index
        custs = [
            j for j, ok in enumerate(mask)
            if ok and instance.nodes[j].is_customer and not env.served_mask[j]
        ]
        stations = [j for j, ok in enumerate(mask) if ok and instance.nodes[j].is_charging_station]
        action = None
        if custs:
            if variant in ("edf", "hybrid", "fleet"):
                custs.sort(key=lambda j: (float(instance.nodes[j].due_time), j))
            elif variant == "energy":
                cur = ev.current_node
                custs.sort(key=lambda j: (float(instance.energy_matrix[cur, j]), j))
            else:
                cur = ev.current_node
                custs.sort(key=lambda j: (float(instance.distance_matrix[cur, j]), j))
            idx = 0 if variant == "edf" else (salt % min(3, len(custs)))
            action = PolicyAction(node=int(custs[idx]))
        elif stations and float(getattr(ev, "battery", instance.vehicle.battery_capacity)) < 0.4 * float(
            instance.vehicle.battery_capacity
        ):
            cur = ev.current_node
            stations.sort(key=lambda j: float(instance.energy_matrix[cur, j]))
            action = PolicyAction(node=int(stations[0]), charge_portion=1.0)
        elif mask[depot]:
            action = PolicyAction(node=int(depot))

        if action is None:
            if mask[depot]:
                try:
                    env.apply_depot_finish(ev)
                except Exception:
                    pass
            steps += 1
            continue
        try:
            apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
        except Exception:
            try:
                if mask[depot]:
                    env.apply_depot_finish(ev)
            except Exception:
                pass
        steps += 1

    for ev in env.evs:
        if ev.used() and not ev.finished:
            try:
                env.apply_depot_finish(ev)
            except Exception:
                pass

    routes = extract_routes_from_env(env)
    rounds = 12
    st_k = 2
    if deadline is not None:
        rem = deadline.remaining()
        if rem < 5:
            rounds, st_k = 2, 1
        elif rem < 15:
            rounds, st_k = 6, 1
        elif rem < 30:
            rounds, st_k = 10, 2
    routes, res = repair_solution(
        instance, routes, config=cfg, weights=w, m=fleet_size,
        max_rounds=rounds, mode="regret", deadline=deadline, station_candidates=st_k,
    )
    score = res.score
    if score is None:
        final = replay_solution(instance, routes, config=cfg, weights=w, m=fleet_size)
        score = final.score or SearchScore.from_env(env, w)
        return routes, score, bool(final.feasible)
    return routes, score, bool(res.feasible)
