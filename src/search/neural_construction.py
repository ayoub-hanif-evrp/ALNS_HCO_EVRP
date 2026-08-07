"""Neural-guided exact construction with expert fallback and replay shield."""
from __future__ import annotations

import time
from typing import Any

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import apply_policy_action
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.search.neural_guide import NeuralSearchGuide
from src.search.repair import repair_solution
from src.search.route_replay import extract_routes_from_env, replay_solution
from src.search.score import SearchScore


def run_neural_construction(
    instance,
    m: int,
    guide: NeuralSearchGuide,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    max_steps_factor: int = 6,
    repair: bool = True,
) -> tuple[list, SearchScore, dict[str, Any]]:
    """Construct with Transformer ranking of exact legal candidates."""
    cfg = config or {}
    problem = cfg.get("problem") or {}
    portions = list(problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    w = weights or (CostWeights.from_eval_config(cfg) if cfg else CostWeights())
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(cfg) if cfg else ExpertConfig())
    t0 = time.perf_counter()

    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=portions,
        use_horizon_check=bool(problem.get("use_horizon_check", True)),
        use_depot_charging=bool(problem.get("use_depot_charging", True)),
        max_steps_factor=max_steps_factor,
        hard_time_windows=True,
    )
    # Optional pressure seeding via expert (state-based).
    if hasattr(expert, "bootstrap_type1_seeds"):
        try:
            expert.bootstrap_type1_seeds(env)
        except Exception:
            pass

    emb = guide.encode_instance(instance)
    steps = 0
    while not env.is_done() and steps < env.max_steps:
        if hasattr(expert, "select_acting_ev_for_type1"):
            ev = expert.select_acting_ev_for_type1(env)
        else:
            ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
            steps += 1
            continue
        legal_c = [
            int(c) for c in instance.customer_indices
            if bool(analysis.final_node_mask[int(c)]) and not env.served_mask[int(c)]
        ]
        if not legal_c and env.try_auto_close_from_analysis(ev, analysis) is not None:
            steps += 1
            continue

        expert_action = expert.select_action(env, ev, analysis)
        action, meta = guide.select_construction_action(
            env, ev, analysis, expert_action=expert_action,
        )
        if action is None:
            break
        try:
            apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
            if meta.get("source") == "neural":
                guide.telemetry.accepted_neural_guided_moves += 1
            guide.telemetry.candidates_replayed += 1
        except Exception:
            # Exact shield: fall back to expert if neural action illegal under race.
            guide.telemetry.expert_fallbacks += 1
            if expert_action is None:
                break
            try:
                apply_policy_action(
                    env, ev, expert_action, mask_result=getattr(analysis, "mask_result", None),
                )
            except Exception:
                break
        steps += 1
        _ = emb  # embeddings reused via guide cache

    for ev in env.evs:
        if ev.used() and not ev.finished:
            try:
                env.apply_depot_finish(ev)
            except Exception:
                pass

    routes = extract_routes_from_env(env)
    if repair:
        routes, res = repair_solution(
            instance, routes, config=cfg, weights=w, m=m, max_rounds=12, mode="regret",
        )
        score = res.score or SearchScore.from_env(env, w)
        feasible = bool(res.feasible)
    else:
        final = replay_solution(instance, routes, config=cfg, weights=w, m=m)
        score = final.score or SearchScore.from_env(env, w)
        feasible = bool(final.feasible)

    guide.telemetry.search_seconds += time.perf_counter() - t0
    return routes, score, {
        "feasible": feasible,
        "telemetry": guide.telemetry.as_dict(),
        "steps": steps,
    }
