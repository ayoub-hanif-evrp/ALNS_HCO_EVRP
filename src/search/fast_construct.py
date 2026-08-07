"""Fast construction helpers for budget-limited fleet search."""
from __future__ import annotations

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import PolicyAction, apply_policy_action
from src.search.deadline import SearchDeadline
from src.search.repair import repair_solution
from src.search.route_replay import extract_routes_from_env, replay_solution
from src.search.score import SearchScore


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
