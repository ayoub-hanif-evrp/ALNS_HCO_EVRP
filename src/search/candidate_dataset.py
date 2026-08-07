"""Candidate-level decision dataset for TG-FPS ranking BC."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import PolicyAction, apply_policy_action
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.search.score import SearchScore
from src.search.state_features import compute_construction_pressure


@dataclass
class CandidateDecision:
    instance_name: str
    fleet_size: int
    expert_variant: str
    acting_ev_id: int
    candidate_nodes: list[int]
    candidate_mask: list[bool]
    best_candidate_index: int
    expert_node: int
    score_vectors: list[list[float]]  # lex deltas / absolute scores per candidate
    global_features: list[float]
    acting_ev_features: list[float]
    fleet_pressure: float


def _short_continuation_score(
    env: EVRPTWEnv,
    ev,
    action: PolicyAction,
    expert: FeasibilityFirstExpert,
    weights: CostWeights,
    horizon: int = 8,
) -> SearchScore:
    """Clone-free light continuation: apply on a deepcopy env."""
    import copy

    e2 = copy.deepcopy(env)
    # Find matching EV by id.
    ev2 = next(x for x in e2.evs if x.ev_id == ev.ev_id)
    analysis = e2.analyze_decision(ev2)
    try:
        apply_policy_action(e2, ev2, action, mask_result=getattr(analysis, "mask_result", None))
    except Exception:
        return SearchScore.worst()
    steps = 0
    while not e2.is_done() and steps < horizon:
        cur = e2.select_acting_ev()
        if cur is None:
            break
        an = e2.analyze_decision(cur)
        if e2.try_auto_close_from_analysis(cur, an) is not None:
            steps += 1
            continue
        act = expert.select_action(e2, cur, an)
        if act is None:
            break
        try:
            apply_policy_action(e2, cur, act, mask_result=getattr(an, "mask_result", None))
        except Exception:
            break
        steps += 1
    return SearchScore.from_env(e2, weights)


def collect_candidate_decisions_from_env(
    env: EVRPTWEnv,
    *,
    expert: FeasibilityFirstExpert,
    weights: CostWeights,
    top_k: int = 16,
    horizon: int = 6,
    expert_variant: str = "hybrid",
) -> list[CandidateDecision]:
    """At current state, score top legal candidates via short exact continuation."""
    ev = env.select_acting_ev()
    if ev is None:
        return []
    analysis = env.analyze_decision(ev)
    if analysis.mask_result is None or analysis.final_node_mask is None:
        return []
    mask = analysis.final_node_mask
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        return []
    # Prefer customers, then stations, then depot.
    inst = env.instance

    def _prio(j: int) -> tuple:
        n = inst.nodes[j]
        if n.is_customer:
            return (0, float(n.due_time), j)
        if n.is_charging_station:
            return (1, float(inst.distance_matrix[ev.current_node, j]), j)
        return (2, 0.0, j)

    legal.sort(key=_prio)
    legal = legal[: max(1, top_k)]
    scores: list[SearchScore] = []
    for j in legal:
        portion = None
        depot_finish = False
        n = inst.nodes[j]
        if n.is_charging_station or (n.is_depot and not depot_finish):
            portion = 1.0
        if n.is_depot:
            # Prefer finish if legal.
            dm = analysis.mask_result.depot_mode_mask
            if dm is not None and len(dm) > 1 and bool(dm[1]):
                depot_finish = True
                portion = None
            elif dm is not None and len(dm) > 0 and bool(dm[0]):
                depot_finish = False
                portion = 1.0
        action = PolicyAction(node=j, charge_portion=portion, depot_finish=depot_finish)
        scores.append(_short_continuation_score(env, ev, action, expert, weights, horizon=horizon))

    best_i = int(np.argmin([s.as_tuple() for s in scores]))
    pressure = compute_construction_pressure(env)
    gf = list(np.asarray(analysis.global_features, dtype=np.float32).reshape(-1)) if analysis.global_features is not None else []
    from src.models.state_features import acting_ev_features

    aef = list(np.asarray(acting_ev_features(env, ev), dtype=np.float32).reshape(-1))
    return [
        CandidateDecision(
            instance_name=str(inst.instance_name),
            fleet_size=int(env.m),
            expert_variant=expert_variant,
            acting_ev_id=int(ev.ev_id),
            candidate_nodes=[int(x) for x in legal],
            candidate_mask=[True] * len(legal),
            best_candidate_index=best_i,
            expert_node=int(legal[best_i]),
            score_vectors=[list(s.as_tuple()) for s in scores],
            global_features=gf,
            acting_ev_features=aef,
            fleet_pressure=float(pressure.fleet_pressure),
        )
    ]


def build_candidate_dataset(
    episodes: list[dict],
    instance_by_name: dict,
    *,
    config: dict,
    max_states_per_episode: int = 40,
    top_k: int = 12,
) -> list[dict]:
    """Replay demos; at subsampled states collect candidate rankings."""
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(config))
    weights = CostWeights.from_eval_config(config)
    problem = config.get("problem") or {}
    portions = list(problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    out: list[dict] = []
    for ep in episodes:
        name = ep["instance_name"]
        if name not in instance_by_name:
            continue
        inst = instance_by_name[name]
        env = EVRPTWEnv(
            inst,
            m=int(ep.get("m") or ep.get("m_star") or 10),
            charging_portions=portions,
            use_horizon_check=bool(problem.get("use_horizon_check", True)),
            use_depot_charging=bool(problem.get("use_depot_charging", True)),
            hard_time_windows=True,
        )
        actions = list(ep.get("actions") or [])
        collected = 0
        ai = 0
        while not env.is_done() and ai < len(actions) and collected < max_states_per_episode:
            if ai % max(1, len(actions) // max_states_per_episode) == 0:
                decs = collect_candidate_decisions_from_env(
                    env, expert=expert, weights=weights, top_k=top_k
                )
                for d in decs:
                    out.append(asdict(d))
                    collected += 1
            ev = env.select_acting_ev()
            if ev is None:
                break
            analysis = env.analyze_decision(ev)
            if env.try_auto_close_from_analysis(ev, analysis) is not None:
                continue
            a = actions[ai]
            ai += 1
            try:
                apply_policy_action(
                    env,
                    ev,
                    PolicyAction(
                        node=int(a["node"]),
                        charge_portion=a.get("charge_portion"),
                        depot_finish=bool(a.get("depot_finish", False)),
                    ),
                    mask_result=getattr(analysis, "mask_result", None),
                )
            except Exception:
                break
    return out


def save_candidate_dataset(decisions: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"decisions": decisions, "format": "tgfps_candidate_v1"}, path)
    return path
