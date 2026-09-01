"""Generic critical-customer rescue with detailed attribution (no instance-name branches)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from src.hanif.env.cost import CostWeights
from src.hanif.search.charging_dp import optimize_charging_plan
from src.hanif.search.ejection import ejection_rescue
from src.hanif.search.repair import _clone_routes, best_insertion, enumerate_insertions_for_customer, repair_solution
from src.hanif.search.route_pool import RoutePool
from src.hanif.search.route_replay import replay_route, replay_solution
from src.hanif.search.score import SearchScore
from src.hanif.search.set_partitioning import solve_set_partitioning


@dataclass
class CustomerFailureRecord:
    customer: int
    recoverable_by_insertion: bool
    capacity_blocker: bool
    tw_blocker: bool
    battery_blocker: bool
    candidate_routes: list[int]
    extra_fleet_recoverable: bool
    detail: dict = field(default_factory=dict)


def attribute_blockers(instance, customer: int, routes, *, config=None, m=None) -> CustomerFailureRecord:
    """Test each route/position and report concrete insertion failure causes."""
    node = instance.nodes[int(customer)]
    demand = float(node.demand)
    veh = instance.vehicle
    cap = float(getattr(veh, "load_capacity", 0) or 0)
    batt = float(veh.battery_capacity)
    energy = instance.energy_matrix
    travel = instance.travel_time_matrix
    depot = int(instance.depot_index)

    best = None
    best_reason = "no_candidate"
    best_detail: dict[str, Any] = {}
    candidate_routes: list[int] = []

    for ri, (nodes, labels) in enumerate(routes):
        load = sum(float(instance.nodes[n].demand) for n in nodes if instance.nodes[n].is_customer)
        cap_remain = cap - load
        if demand > cap_remain + 1e-9:
            reason = "capacity_remaining"
        else:
            reason = None
        # Probe positions
        for pos in range(1, max(1, len(nodes))):
            prev = nodes[pos - 1] if pos - 1 < len(nodes) else depot
            # Earliest arrival assuming leave prev at ready of prev (approx via replay of prefix)
            e_to = float(energy[prev, customer])
            t_to = float(travel[prev, customer])
            # Battery path rough check from depot via prev
            batt_ok = e_to <= batt + 1e-9
            arrive_earliest = float(instance.nodes[prev].ready_time) + t_to
            arrive_latest_ok = arrive_earliest <= float(node.due_time) + 1e-9
            tw_slack = float(node.due_time) - arrive_earliest
            depot_e = float(energy[customer, depot])
            depot_ok = depot_e <= batt + 1e-9
            cand = None
            try:
                from src.hanif.search.repair import evaluate_insertion
                cand = evaluate_insertion(
                    instance, routes, ri, pos, int(customer), config=config, m=m,
                )
            except Exception:
                cand = None
            if cand is not None:
                candidate_routes.append(ri)
                best = cand
                best_reason = "insertable"
                best_detail = {
                    "route": ri,
                    "position": pos,
                    "capacity_remaining": cap_remain,
                    "earliest_feasible_arrival": arrive_earliest,
                    "tw_slack": tw_slack,
                    "battery_path_feasible": batt_ok,
                    "depot_return_feasible": depot_ok,
                }
                break
            fail = reason or (
                "battery_path" if not batt_ok else (
                    "tw_arrival" if not arrive_latest_ok else (
                        "depot_return" if not depot_ok else "replay_reject"
                    )
                )
            )
            detail = {
                "route": ri,
                "position": pos,
                "capacity_remaining": cap_remain,
                "earliest_feasible_arrival": arrive_earliest,
                "latest_feasible_arrival": float(node.due_time),
                "tw_slack": tw_slack,
                "battery_path_feasible": batt_ok,
                "depot_return_feasible": depot_ok,
                "best_failed_reason": fail,
            }
            if best is None:
                best_reason = fail
                best_detail = detail
        if best is not None:
            break

    # New-route recoverability
    depot_route = [([depot, depot], [None, None])]
    new_route_ok = best_insertion(
        instance, routes + depot_route, int(customer), config=config, m=(m or len(routes)) + 1,
    ) is not None

    # Ejection recoverability (cheap flag)
    eject_ok = False
    try:
        from src.hanif.search.ejection import try_ejection_chain
        _, _, att = try_ejection_chain(
            instance, routes, int(customer), config=config, m=m, depth=1, max_candidates=8,
        )
        eject_ok = bool(att.feasible)
    except Exception:
        eject_ok = False

    capacity_blocker = demand > cap + 1e-9 or best_reason == "capacity_remaining"
    tw_blocker = best_reason in ("tw_arrival",) or float(node.due_time) <= float(node.ready_time) + 1e-9
    battery_blocker = best_reason in ("battery_path", "depot_return")

    best_detail.update({
        "best_failed_reason": best_reason,
        "ejection_chain_recoverability": eject_ok,
        "new_route_recoverability": new_route_ok,
        "required_station_insertion": None,
        "charging_time_impact": None,
    })
    return CustomerFailureRecord(
        customer=int(customer),
        recoverable_by_insertion=best is not None,
        capacity_blocker=capacity_blocker,
        tw_blocker=tw_blocker,
        battery_blocker=battery_blocker,
        candidate_routes=sorted(set(candidate_routes)),
        extra_fleet_recoverable=new_route_ok,
        detail=best_detail,
    )


def critical_rescue(
    instance,
    routes: list,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    max_unserved: int = 8,
) -> tuple[list, SearchScore, list[dict]]:
    """Reduced rescue subproblem: related routes → destroy → columns → set partition."""
    cfg = config or {}
    w = weights or CostWeights()
    current = _clone_routes(routes)
    res = replay_solution(instance, current, config=cfg, weights=w, m=m)
    assert res.score is not None
    served = set(res.served_customers)
    unserved = [i for i, n in enumerate(instance.nodes) if n.is_customer and i not in served]
    logs: list[dict] = []
    for u in unserved:
        rec = attribute_blockers(instance, u, current, config=cfg, m=m)
        logs.append(asdict(rec))

    if not unserved or len(unserved) > max_unserved:
        return current, res.score, logs

    import numpy as np

    def _centroid(nodes):
        xs, ys = [], []
        for n in nodes:
            if instance.nodes[n].is_customer:
                xs.append(float(instance.nodes[n].x))
                ys.append(float(instance.nodes[n].y))
        if not xs:
            return None
        return float(np.mean(xs)), float(np.mean(ys))

    u_xy = [(float(instance.nodes[u].x), float(instance.nodes[u].y)) for u in unserved]
    u_cx = float(np.mean([p[0] for p in u_xy]))
    u_cy = float(np.mean([p[1] for p in u_xy]))

    # Compatibility ranking: spatial + residual capacity + rough TW
    scored_routes = []
    for ri, (nodes, _) in enumerate(current):
        c = _centroid(nodes)
        if c is None:
            continue
        load = sum(float(instance.nodes[n].demand) for n in nodes if instance.nodes[n].is_customer)
        cap = float(instance.vehicle.load_capacity)
        residual = cap - load
        dist = (c[0] - u_cx) ** 2 + (c[1] - u_cy) ** 2
        scored_routes.append((dist - 0.01 * residual, ri))
    scored_routes.sort()
    related = [ri for _, ri in scored_routes[: max(3, min(8, len(scored_routes)))]]

    trial = _clone_routes(current)
    removed = list(unserved)
    for ri in related:
        nodes, labels = trial[ri]
        keep_n, keep_l = [], []
        for n, lab in zip(nodes, labels or [None] * len(nodes)):
            if instance.nodes[n].is_customer:
                dx = float(instance.nodes[n].x) - u_cx
                dy = float(instance.nodes[n].y) - u_cy
                # Adaptive radius: 25 or median pairwise among related
                if dx * dx + dy * dy < (35.0 ** 2):
                    removed.append(int(n))
                    continue
            keep_n.append(n)
            keep_l.append(lab)
        if len(keep_n) < 2:
            depot = instance.depot_index
            trial[ri] = ([depot, depot], [None, None])
        else:
            trial[ri] = (keep_n, keep_l)

    trial, rep = repair_solution(
        instance, trial, config=cfg, weights=w, m=m, max_rounds=20, mode="regret",
    )
    # Ejection chains for remaining
    res2 = replay_solution(instance, trial, config=cfg, weights=w, m=m)
    still = [i for i, n in enumerate(instance.nodes) if n.is_customer and i not in set(res2.served_customers)]
    if still:
        trial, _, ej_logs = ejection_rescue(
            instance, trial, still, config=cfg, weights=w, m=m, depth=2,
        )
        for ej in ej_logs:
            logs.append({"ejection": ej.__dict__})

    # Charging optimize + route pool recombination on related columns
    pool = RoutePool()
    pool.add_from_solution(instance, current, score=res.score, source_method="pre_rescue", source_fleet_size=m or len(current), config=cfg)
    optimized = []
    for nodes, labels in trial:
        custs = [n for n in nodes if instance.nodes[n].is_customer]
        if not custs:
            optimized.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
            continue
        plan = optimize_charging_plan(instance, custs, config=cfg, m=m or len(trial))
        if plan.feasible and plan.nodes:
            optimized.append((list(plan.nodes), list(plan.labels)))
        else:
            optimized.append((list(nodes), list(labels) if labels else [None] * len(nodes)))
    trial = optimized
    pool.add_from_solution(instance, trial, source_method="rescue_columns", source_fleet_size=m or len(trial), config=cfg)
    rebuilt, pool_score, meta = solve_set_partitioning(instance, pool, config=cfg, weights=w, m=m)
    if rebuilt and pool_score is not None:
        trial, score = rebuilt, pool_score
    else:
        final = replay_solution(instance, trial, config=cfg, weights=w, m=m)
        score = final.score or rep.score or res.score

    if score.dominates(res.score):
        return trial, score, logs
    return current, res.score, logs
