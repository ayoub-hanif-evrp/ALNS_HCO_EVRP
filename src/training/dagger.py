"""DAgger-style aggregation for TG-FPS candidate ranking."""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data.instance_loader import load_capacity_from_config
from src.data.evrptw_dataset import split_dataset
from src.data.training_phases import resolve_training_phase_split
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import PolicyAction, apply_policy_action
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.search.candidate_dataset import collect_candidate_decisions_from_env, save_candidate_dataset
from src.training.candidate_ranker import _CandidateAuxHead, train_candidate_ranker
from src.training.train_rl import load_policy
from src.utils.logging_utils import get_logger
from src.utils.seed import resolve_device

LOGGER = get_logger()

DEFAULT_BETAS = [1.0, 0.7, 0.4, 0.20]


def _load_ranker(ckpt: str | Path, device):
    policy = load_policy(ckpt, device=device)
    payload = torch.load(ckpt, map_location=device, weights_only=False)
    if not hasattr(policy, "candidate_aux_head"):
        policy.candidate_aux_head = _CandidateAuxHead().to(device)
    if payload.get("candidate_aux_head"):
        policy.candidate_aux_head.load_state_dict(payload["candidate_aux_head"])
    return policy


def _learner_action(policy, env, ev, analysis, device) -> PolicyAction | None:
    """Pick among exact legal candidates using aux head or score_candidates."""
    mask = analysis.final_node_mask
    legal = [i for i, ok in enumerate(mask) if ok]
    if not legal:
        return None
    legal = legal[:32]
    try:
        if hasattr(policy, "candidate_aux_head"):
            from src.models.state_features import acting_ev_features
            import numpy as np

            aef = torch.tensor(
                list(np.asarray(acting_ev_features(env, ev), dtype=np.float32).reshape(-1))[:32],
                dtype=torch.float32,
                device=device,
            )
            if aef.numel() < 32:
                aef = F.pad(aef, (0, 32 - aef.numel()))
            scores = torch.zeros(len(legal), 7, device=device)
            node_t = torch.tensor(legal, dtype=torch.long, device=device)
            logits = policy.candidate_aux_head(aef, scores, node_t)
            j = int(legal[int(torch.argmax(logits).item())])
        else:
            emb = policy.encode(env.instance, device=device)
            logits = policy.score_candidates(
                env, ev, emb, legal, mask_result=getattr(analysis, "mask_result", None),
            )
            j = int(legal[int(torch.argmax(logits).item())])
    except Exception:
        return None
    portion = None
    depot_finish = False
    n = env.instance.nodes[j]
    if n.is_charging_station:
        portion = 1.0
    if n.is_depot:
        dm = analysis.depot_mode_mask
        if dm is not None and len(dm) > 1 and bool(dm[1]):
            depot_finish = True
            portion = None
        elif dm is not None and len(dm) > 0 and bool(dm[0]):
            portion = 1.0
    return PolicyAction(node=j, charge_portion=portion, depot_finish=depot_finish)


def run_dagger_rounds(
    config: dict,
    *,
    initial_decisions: list[dict],
    rounds: int = 2,
    betas: list[float] | None = None,
    init_checkpoint: str | Path | None = None,
    max_train_instances: int = 4,
    max_steps: int = 40,
) -> dict:
    """Run at least ``rounds`` DAgger aggregations with exact-safe rollouts."""
    betas = list(betas or DEFAULT_BETAS[: max(2, rounds)])
    device = resolve_device(config.get("device", "cpu"))
    train_names, eval_names, _ = resolve_training_phase_split(config)
    val_names = list((config.get("dataset") or {}).get("validation_instances") or [])
    train_names = [n for n in train_names if n not in set(val_names)]
    ds_cfg = dict(config["dataset"])
    n_train = min(int(max_train_instances), max(1, len(train_names)))
    train_ds, _ = split_dataset(
        ds_cfg["path"],
        train_ratio=0.8,
        seed=int(config.get("seed", 42)),
        train_filter=ds_cfg.get("train_size_filter") if ds_cfg.get("train_size_filter") is not None else [],
        eval_filter=ds_cfg.get("eval_size_filter") if ds_cfg.get("eval_size_filter") is not None else [],
        train_instances=train_names[:n_train],
        eval_instances=eval_names,
        load_capacity=load_capacity_from_config(config),
    )
    instance_by_name = {n: train_ds[train_ds.names.index(n)] for n in train_ds.names}
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(config))
    weights = CostWeights.from_eval_config(config)
    problem = config.get("problem") or {}
    portions = list(problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0])

    decisions = list(initial_decisions)
    if not decisions:
        decisions = _seed_decisions(instance_by_name, expert, weights, portions, problem)
    history = []
    ckpt = init_checkpoint
    out_dir = Path((config.get("behavior_cloning") or {}).get("output_dir", "outputs/bc"))
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(config.get("seed", 42)))

    n_rounds = max(2, int(rounds))
    for r in range(n_rounds):
        beta = float(betas[min(r, len(betas) - 1)])
        LOGGER.info("DAgger round %d/%d beta=%.2f decisions=%d", r + 1, n_rounds, beta, len(decisions))
        summary = train_candidate_ranker(
            config,
            decisions,
            init_checkpoint=ckpt,
            epochs=2,
            device=device,
        )
        ckpt = summary["best_checkpoint"]
        policy = _load_ranker(ckpt, device)
        policy.eval()

        new_decisions: list[dict] = []
        for name, inst in list(instance_by_name.items())[:n_train]:
            m = max(6, int(getattr(inst, "m_min", 6) or 6) + 2)
            env = EVRPTWEnv(
                inst, m=m, charging_portions=portions,
                use_horizon_check=bool(problem.get("use_horizon_check", True)),
                use_depot_charging=bool(problem.get("use_depot_charging", True)),
                hard_time_windows=True,
            )
            steps = 0
            while not env.is_done() and steps < max_steps:
                ev = env.select_acting_ev()
                if ev is None:
                    break
                analysis = env.analyze_decision(ev)
                if env.try_auto_close_from_analysis(ev, analysis) is not None:
                    steps += 1
                    continue
                if steps % 4 == 0:
                    for d in collect_candidate_decisions_from_env(
                        env, expert=expert, weights=weights, top_k=6, horizon=2
                    ):
                        new_decisions.append(asdict(d))
                use_oracle = rng.random() < beta
                if use_oracle:
                    action = expert.select_action(env, ev, analysis)
                else:
                    action = _learner_action(policy, env, ev, analysis, device)
                    if action is None:
                        action = expert.select_action(env, ev, analysis)
                if action is None:
                    break
                try:
                    apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
                except Exception:
                    try:
                        action = expert.select_action(env, ev, analysis)
                        if action is None:
                            break
                        apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
                    except Exception:
                        break
                steps += 1

        decisions.extend(new_decisions)
        decisions = _rebalance_rare_actions(decisions, rng)
        dagger_path = out_dir / f"dagger_round_{r+1}_candidates.pt"
        save_candidate_dataset(decisions, dagger_path)
        best_path = out_dir / "dagger_best.pt"
        if Path(ckpt).is_file():
            import shutil
            shutil.copy2(ckpt, best_path)
        history.append({
            "round": r + 1,
            "beta": beta,
            "n_decisions": len(decisions),
            "n_new": len(new_decisions),
            "checkpoint": str(best_path),
            "candidate_cache": str(dagger_path),
        })
        LOGGER.info("DAgger round %d done | decisions=%d | %s", r + 1, len(decisions), best_path)

    return {
        "history": history,
        "best_checkpoint": str(out_dir / "dagger_best.pt"),
        "n_decisions": len(decisions),
        "n_rounds": n_rounds,
    }


def _seed_decisions(instance_by_name, expert, weights, portions, problem) -> list[dict]:
    out: list[dict] = []
    for name, inst in list(instance_by_name.items())[:2]:
        m = max(4, int(getattr(inst, "m_min", 4) or 4) + 1)
        env = EVRPTWEnv(
            inst, m=m, charging_portions=portions,
            use_horizon_check=bool(problem.get("use_horizon_check", True)),
            use_depot_charging=bool(problem.get("use_depot_charging", True)),
            hard_time_windows=True,
        )
        for _ in range(3):
            if env.is_done():
                break
            for d in collect_candidate_decisions_from_env(
                env, expert=expert, weights=weights, top_k=5, horizon=2
            ):
                out.append(asdict(d))
            ev = env.select_acting_ev()
            if ev is None:
                break
            analysis = env.analyze_decision(ev)
            if env.try_auto_close_from_analysis(ev, analysis) is not None:
                continue
            action = expert.select_action(env, ev, analysis)
            if action is None:
                break
            try:
                apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
            except Exception:
                break
    return out


def _rebalance_rare_actions(decisions: list[dict], rng: random.Random) -> list[dict]:
    rare, common = [], []
    for d in decisions:
        nodes = d.get("candidate_nodes") or []
        best = int(d.get("best_candidate_index", 0))
        if not nodes:
            continue
        if float(d.get("fleet_pressure", 0.0)) > 0.5 or best == 0:
            rare.append(d)
        else:
            common.append(d)
    if len(common) > 3 * max(len(rare), 1):
        rng.shuffle(common)
        common = common[: 3 * max(len(rare), 1)]
    return rare + rare + common
