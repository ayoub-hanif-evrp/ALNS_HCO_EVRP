"""Diversified exact expert portfolio for TG-FPS construction."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from src.hanif.env.cost import CostWeights
from src.hanif.env.evrptw_env import EVRPTWEnv
from src.hanif.env.policy_action import PolicyAction, apply_policy_action
from src.hanif.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.hanif.search.repair import repair_solution
from src.hanif.search.route_replay import extract_routes_from_env, replay_solution
from src.hanif.search.score import SearchScore
from src.hanif.search.state_features import compute_construction_pressure

ExpertVariant = Literal["edf", "regret", "energy", "fleet", "hybrid"]


@dataclass
class PortfolioResult:
    variant: str
    seed: int
    routes: list[tuple[list[int], list[float | None]]]
    score: SearchScore
    m: int
    feasible: bool


def _rank_customers(env, analysis, variant: ExpertVariant, pressure) -> list[int]:
    inst = env.instance
    mask = analysis.final_node_mask
    customers = [j for j, n in enumerate(inst.nodes) if n.is_customer and mask[j] and not env.served_mask[j]]
    if variant == "edf" or (variant == "hybrid" and pressure.prefer_earliest_due):
        customers.sort(key=lambda j: (float(inst.nodes[j].due_time), j))
    elif variant == "energy" or (variant == "hybrid" and pressure.prefer_energy_first):
        ev = env.evs[analysis.active_ev_id]
        customers.sort(
            key=lambda j: (
                float(inst.energy_matrix[ev.current_node, j]),
                float(inst.nodes[j].due_time),
            )
        )
    elif variant == "fleet" or (variant == "hybrid" and pressure.prefer_fresh_ev_release):
        customers.sort(key=lambda j: (-float(inst.nodes[j].demand), float(inst.nodes[j].due_time)))
    else:  # regret-like: due * distance
        ev = env.evs[analysis.active_ev_id]
        customers.sort(
            key=lambda j: (
                float(inst.nodes[j].due_time) * (1.0 + 0.01 * float(inst.distance_matrix[ev.current_node, j])),
                j,
            )
        )
    return customers


def _select_action(env, expert: FeasibilityFirstExpert, variant: ExpertVariant, seed: int):
    pressure = compute_construction_pressure(env)
    # Prefer EV with tightest slack under fleet/energy pressure.
    if pressure.prefer_fresh_ev_release and hasattr(expert, "select_acting_ev_for_type1"):
        # Reuse existing urgency EV picker without filename gate by temporarily
        # calling bootstrap/seed helpers via pressure flags below.
        pass
    ev = env.select_acting_ev()
    if ev is None:
        return None, None, None
    analysis = env.analyze_decision(ev)
    if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
        return ev, analysis, None
    if env.try_auto_close_from_analysis(ev, analysis) is not None:
        return ev, analysis, None
    # Deterministic tie-break: rotate preference by seed.
    ranked = _rank_customers(env, analysis, variant, pressure)
    if seed and ranked:
        rot = int(seed) % max(len(ranked), 1)
        ranked = ranked[rot:] + ranked[:rot]
    if ranked:
        return ev, analysis, PolicyAction(node=int(ranked[0]), charge_portion=None, depot_finish=False)
    # Fall back to base expert ranking.
    action = expert.select_action(env, ev, analysis)
    return ev, analysis, action


def run_variant_episode(
    instance,
    m: int,
    *,
    variant: ExpertVariant = "hybrid",
    seed: int = 0,
    config: dict | None = None,
    weights: CostWeights | None = None,
    max_steps_factor: int = 6,
    deadline=None,
) -> PortfolioResult:
    cfg = config or {}
    problem = cfg.get("problem") or {}
    portions = list(problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    w = weights or (CostWeights.from_eval_config(cfg) if cfg else CostWeights())
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(cfg) if cfg else ExpertConfig())
    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=portions,
        use_horizon_check=bool(problem.get("use_horizon_check", True)),
        use_depot_charging=bool(problem.get("use_depot_charging", True)),
        max_steps_factor=max_steps_factor,
        hard_time_windows=True,
    )
    # State-based seeding: when pressure high, bootstrap earliest-due onto idle EVs.
    pressure = compute_construction_pressure(env)
    if pressure.prefer_fresh_ev_release or pressure.prefer_earliest_due:
        if hasattr(expert, "bootstrap_type1_seeds"):
            # Call underlying seed logic by temporarily ignoring name check:
            # use a small local seed of critical customers onto idle depot EVs.
            _seed_from_pressure(env, pressure)

    steps = 0
    while not env.is_done() and steps < env.max_steps:
        if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
            break
        ev, analysis, action = _select_action(env, expert, variant, seed)
        if ev is None:
            break
        if action is None:
            steps += 1
            continue
        try:
            apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
        except Exception:
            # Fall back to expert action if ranked action illegal under mask race.
            try:
                action = expert.select_action(env, ev, analysis)
                if action is not None:
                    apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
            except Exception:
                break
        steps += 1

    # Close open EVs.
    for ev in env.evs:
        if ev.used() and not ev.finished:
            try:
                env.apply_depot_finish(ev)
            except Exception:
                pass

    routes = extract_routes_from_env(env)
    # Repair is expensive; skip or shrink when the shared deadline is exhausted.
    if deadline is not None and hasattr(deadline, "expired") and deadline.expired():
        from src.hanif.search.route_replay import replay_solution
        final = replay_solution(instance, routes, config=cfg, weights=w, m=m)
        score = final.score or SearchScore.from_env(env, w)
        return PortfolioResult(
            variant=variant,
            seed=int(seed),
            routes=routes,
            score=score,
            m=m,
            feasible=bool(final.feasible) and int(final.unserved_customers) == 0,
        )
    rounds = 6
    if deadline is not None and hasattr(deadline, "remaining"):
        rem = float(deadline.remaining())
        if rem < 5:
            rounds = 1
        elif rem < 20:
            rounds = 3
        elif rem < 60:
            rounds = 4
    routes, res = repair_solution(
        instance, routes, config=cfg, weights=w, m=m, max_rounds=rounds, mode="regret",
        deadline=deadline, station_candidates=2,
    )
    score = res.score or SearchScore.from_env(env, w)
    return PortfolioResult(
        variant=variant,
        seed=int(seed),
        routes=routes,
        score=score,
        m=m,
        feasible=bool(res.feasible),
    )


def _seed_from_pressure(env, pressure) -> int:
    """Seed earliest-due unserved onto idle depot EVs using observable pressure only."""
    inst = env.instance
    idle = [
        e for e in env.evs
        if (not e.finished) and e.current_node == inst.depot_index and len(e.route_nodes) <= 1
    ]
    if not idle:
        return 0
    unserved = [
        j for j, n in enumerate(inst.nodes)
        if n.is_customer and not env.served_mask[j]
    ]
    unserved.sort(key=lambda j: float(inst.nodes[j].due_time))
    # Prefer customers that look hard for active routes when depot_illegal high.
    max_seeds = min(len(idle), max(1, pressure.n_critical_slack), len(unserved))
    seeded = 0
    for ev, cust in zip(idle[:max_seeds], unserved[:max_seeds]):
        analysis = env.analyze_decision(ev)
        if analysis.final_node_mask is not None and analysis.final_node_mask[cust]:
            try:
                apply_policy_action(
                    env, ev, PolicyAction(node=cust), mask_result=analysis.mask_result
                )
                seeded += 1
                continue
            except Exception:
                pass
        # Station then customer using nearest station.
        stations = [i for i, n in enumerate(inst.nodes) if n.is_charging_station]
        stations.sort(key=lambda s: float(inst.energy_matrix[ev.current_node, s]))
        for s in stations[:8]:
            analysis = env.analyze_decision(ev)
            if analysis.final_node_mask is None or not analysis.final_node_mask[s]:
                continue
            try:
                apply_policy_action(
                    env, ev, PolicyAction(node=s, charge_portion=1.0), mask_result=analysis.mask_result
                )
                analysis2 = env.analyze_decision(ev)
                if analysis2.final_node_mask is not None and analysis2.final_node_mask[cust]:
                    apply_policy_action(
                        env, ev, PolicyAction(node=cust), mask_result=analysis2.mask_result
                    )
                    seeded += 1
                    break
            except Exception:
                continue
    return seeded


DEFAULT_VARIANTS = ["edf", "regret", "energy", "fleet", "hybrid"]
DEFAULT_SEEDS = [0, 1, 2, 3]


def _edge_set(routes) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for nodes, _ in routes:
        for a, b in zip(nodes[:-1], nodes[1:]):
            edges.add((int(a), int(b)))
    return edges


def _assignment_signature(instance, routes) -> frozenset[tuple[int, int]]:
    """Customer → route-index assignment for diversity."""
    pairs = []
    for ri, (nodes, _) in enumerate(routes):
        for n in nodes:
            if instance.nodes[n].is_customer:
                pairs.append((int(n), int(ri)))
    return frozenset(pairs)


def select_elite_pool(
    instance,
    results: list[PortfolioResult],
    *,
    elite_size: int = 8,
) -> list[PortfolioResult]:
    """Select elites by score + edge / assignment / charging diversity."""
    if not results:
        return []
    ranked = sorted(results, key=lambda r: r.score.as_tuple())
    elites: list[PortfolioResult] = [ranked[0]]
    for cand in ranked[1:]:
        if len(elites) >= elite_size:
            break
        e_edges = _edge_set(cand.routes)
        e_assign = _assignment_signature(instance, cand.routes)
        e_charge = sum(
            1 for nodes, labs in cand.routes
            for n, lab in zip(nodes, labs or [])
            if instance.nodes[n].is_charging_station and lab is not None
        )
        diverse = True
        for ex in elites:
            inter = len(e_edges & _edge_set(ex.routes))
            union = max(1, len(e_edges | _edge_set(ex.routes)))
            jaccard = inter / union
            same_assign = e_assign == _assignment_signature(instance, ex.routes)
            ex_charge = sum(
                1 for nodes, labs in ex.routes
                for n, lab in zip(nodes, labs or [])
                if instance.nodes[n].is_charging_station and lab is not None
            )
            if jaccard > 0.92 and same_assign and abs(e_charge - ex_charge) <= 1:
                diverse = False
                break
        if diverse:
            elites.append(cand)
    return elites


def run_expert_portfolio(
    instance,
    m: int,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    max_episodes: int | None = None,
    budget_seconds: float | None = None,
) -> list[PortfolioResult]:
    cfg = config or {}
    hs = (cfg.get("hybrid_search") or {}).get("construction") or {}
    variants = variants or list(hs.get("expert_variants_list") or DEFAULT_VARIANTS)
    seeds = seeds or list((cfg.get("fleet_search") or {}).get("seeds") or DEFAULT_SEEDS)
    results: list[PortfolioResult] = []
    t0 = time.perf_counter()
    # Round-robin so variants stay diverse when truncated.
    pairs: list[tuple[str, int]] = []
    for s in seeds:
        for v in variants:
            pairs.append((v, s))
    if max_episodes is not None:
        pairs = pairs[: max(1, int(max_episodes))]
    for v, s in pairs:
        if budget_seconds is not None and (time.perf_counter() - t0) > float(budget_seconds):
            break
        results.append(
            run_variant_episode(
                instance, m, variant=v, seed=s, config=config, weights=weights  # type: ignore[arg-type]
            )
        )
    results.sort(key=lambda r: r.score.as_tuple())
    return results


def best_portfolio_solution(
    instance,
    m: int,
    **kwargs,
) -> PortfolioResult:
    results = run_expert_portfolio(instance, m, **kwargs)
    cfg = kwargs.get("config") or {}
    elite_size = int(
        (((cfg.get("hybrid_search") or {}).get("construction") or {}).get("elite_size", 8))
    )
    elites = select_elite_pool(instance, results, elite_size=elite_size)
    return elites[0] if elites else results[0]
