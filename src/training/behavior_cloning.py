"""Behavior cloning of the Transformer actor from exact-mask expert demos."""
from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.data.evrptw_dataset import split_dataset
from src.data.instance_loader import load_capacity_from_config
from src.data.training_phases import resolve_training_phase_split
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import PolicyAction, apply_policy_action
from src.expert.generate_demonstrations import run_expert_episode
from src.expert.feasibility_first_expert import FeasibilityFirstExpert
from src.models.policy import TransformerEVRPTWPolicy
from src.models.state_features import (
    acting_ev_features,
    battery_state_features,
    ev_token_scalars,
    fleet_balance_features,
)
from src.training.flat_schedule import sample_fleet_size
from src.training.train_rl import build_policy, load_policy
from src.utils.config import load_config
from src.utils.logging_utils import get_logger
from src.utils.seed import describe_device, resolve_device, set_seed
from src.utils.tensor_utils import safe_mask_logits

LOGGER = get_logger()


@dataclass
class CachedBCDecision:
    """Frozen exact-mask decision for BC regularization (no re-analysis)."""

    instance_name: str
    expert_node: int
    expert_charge_portion: float | None
    expert_depot_finish: bool
    mask_result: object
    acting_ev_feats: np.ndarray
    token_scalars: np.ndarray
    fleet_balance: np.ndarray
    finished: np.ndarray
    current_nodes: np.ndarray
    acting_idx: int
    global_features: np.ndarray
    mask_summary_features: np.ndarray
    dynamic_candidate_features: np.ndarray
    battery_feats: np.ndarray


def _portion_index(portions: list[float], value: float) -> int:
    for i, p in enumerate(portions):
        if abs(float(p) - float(value)) < 1e-9:
            return i
    raise ValueError(f"charge portion {value} not in {portions}")


def masked_ce_from_logits(logits: torch.Tensor, target_idx: int, mask: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on masked logits; never trains illegal actions."""
    masked = safe_mask_logits(logits, mask)
    log_probs = torch.log_softmax(masked, dim=-1)
    if not bool(mask[target_idx].item()):
        raise RuntimeError(f"BC target index {target_idx} is illegal under mask")
    return -log_probs[target_idx]


def bc_loss_on_decision(
    policy: TransformerEVRPTWPolicy,
    env: EVRPTWEnv,
    ev,
    analysis,
    node_embeddings: torch.Tensor,
    expert_action: PolicyAction,
) -> tuple[torch.Tensor, dict]:
    """Supervise node / charge / depot-mode heads for one decision."""
    mask_result = analysis.mask_result
    state = policy.build_decision_state(
        env, ev, node_embeddings, mask_result, analysis=analysis,
    )
    device = node_embeddings.device
    scores = policy.pointer.node_scores(
        state.decoder_context,
        state.candidate_embeddings,
        state.dynamic_candidate_embeddings,
    )
    node_mask = torch.tensor(state.feasible_mask, dtype=torch.bool, device=device)
    node_loss = masked_ce_from_logits(scores, int(expert_action.node), node_mask)

    charge_loss = torch.zeros((), device=device)
    depot_loss = torch.zeros((), device=device)
    stats = {
        "node_correct": 0.0,
        "charge_correct": 0.0,
        "depot_correct": 0.0,
        "charge_used": 0.0,
        "depot_used": 0.0,
        "illegal_target": 0.0,
    }
    with torch.no_grad():
        pred_node = int(torch.argmax(safe_mask_logits(scores, node_mask)).item())
        stats["node_correct"] = float(pred_node == int(expert_action.node))

    inst = env.instance
    node = int(expert_action.node)
    if inst.nodes[node].is_depot:
        ok_r = bool(mask_result.depot_mode_mask[0])
        ok_f = bool(mask_result.depot_mode_mask[1])
        if ok_r and ok_f:
            mode_logits = policy.pointer.depot_mode_logits(
                state.decoder_context,
                state.candidate_embeddings[node],
                state.global_context,
                state.mask_context,
            )
            dm = torch.tensor(mask_result.depot_mode_mask, device=device, dtype=torch.bool)
            target_mode = 1 if expert_action.depot_finish else 0
            depot_loss = masked_ce_from_logits(mode_logits, target_mode, dm)
            stats["depot_used"] = 1.0
            with torch.no_grad():
                pred = int(torch.argmax(safe_mask_logits(mode_logits, dm)).item())
                stats["depot_correct"] = float(pred == target_mode)
        if not expert_action.depot_finish and expert_action.charge_portion is not None:
            charge_loss, cstats = _charge_loss(
                policy, env, ev, state, node, float(expert_action.charge_portion),
            )
            stats.update(cstats)
    elif inst.nodes[node].is_charging_station:
        if expert_action.charge_portion is None:
            raise RuntimeError("Expert station action missing charge portion")
        charge_loss, cstats = _charge_loss(
            policy, env, ev, state, node, float(expert_action.charge_portion),
        )
        stats.update(cstats)

    loss = node_loss + 0.5 * charge_loss + 0.5 * depot_loss
    stats["node_loss"] = float(node_loss.detach())
    stats["charge_loss"] = float(charge_loss.detach())
    stats["depot_loss"] = float(depot_loss.detach())
    return loss, stats


def _charge_loss(policy, env, ev, state, station: int, portion: float):
    from src.models.state_features import battery_state_features

    device = state.decoder_context.device
    dtype = state.decoder_context.dtype
    batt = torch.tensor(battery_state_features(env, ev), device=device, dtype=dtype)
    logits = policy.pointer.charging_logits(
        state.decoder_context,
        state.candidate_embeddings[station],
        state.acting_ev_context,
        batt,
    )
    cm = state.mask_result.charge_masks[station]
    mask_t = torch.tensor(cm, device=device, dtype=torch.bool)
    idx = _portion_index(list(policy.charging_portions), portion)
    loss = masked_ce_from_logits(logits, idx, mask_t)
    stats = {"charge_used": 1.0, "charge_correct": 0.0}
    with torch.no_grad():
        pred = int(torch.argmax(safe_mask_logits(logits, mask_t)).item())
        stats["charge_correct"] = float(pred == idx)
    return loss, stats


def bc_loss_on_demo_batch(
    policy: TransformerEVRPTWPolicy,
    episodes: list[dict],
    instance_by_name: dict,
    *,
    device: torch.device,
    batch_episodes: int = 1,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict]:
    """Replay random demo episodes and accumulate masked CE."""
    rng = rng or random.Random(0)
    if not episodes:
        z = torch.zeros((), device=device)
        return z, {}
    chosen = [episodes[rng.randrange(len(episodes))] for _ in range(max(1, batch_episodes))]
    losses = []
    agg = {
        "node_correct": 0.0, "charge_correct": 0.0, "depot_correct": 0.0,
        "charge_used": 0.0, "depot_used": 0.0, "n": 0.0,
    }
    for ep in chosen:
        instance = instance_by_name[ep["instance_name"]]
        env = EVRPTWEnv(
            instance,
            m=int(ep["m"]),
            charging_portions=list(policy.charging_portions),
            use_horizon_check=True,
            use_depot_charging=True,
            hard_time_windows=True,
        )
        node_embeddings = policy.encode(instance, device=device)
        action_i = 0
        while not env.is_done() and action_i < len(ep["actions"]):
            ev = env.select_acting_ev()
            if ev is None:
                break
            analysis = env.analyze_decision(ev)
            if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
                continue
            if env.try_auto_close_from_analysis(ev, analysis) is not None:
                continue
            a = ep["actions"][action_i]
            action_i += 1
            expert_action = PolicyAction(
                node=int(a["node"]),
                charge_portion=a.get("charge_portion"),
                depot_finish=bool(a.get("depot_finish", False)),
            )
            loss, stats = bc_loss_on_decision(
                policy, env, ev, analysis, node_embeddings, expert_action,
            )
            losses.append(loss)
            for k in ("node_correct", "charge_correct", "depot_correct", "charge_used", "depot_used"):
                agg[k] += float(stats.get(k, 0.0))
            agg["n"] += 1.0
            apply_policy_action(env, ev, expert_action, mask_result=analysis.mask_result)
    if not losses:
        z = torch.zeros((), device=device)
        return z, agg
    total = torch.stack(losses).mean()
    n = max(agg["n"], 1.0)
    metrics = {
        "bc_loss": float(total.detach()),
        "node_accuracy": agg["node_correct"] / n,
        "charge_accuracy": agg["charge_correct"] / max(agg["charge_used"], 1.0),
        "depot_accuracy": agg["depot_correct"] / max(agg["depot_used"], 1.0),
        "n_decisions": n,
    }
    return total, metrics


def _cache_decisions_for_one_episode(
    ep: dict,
    instance,
    *,
    charging_portions: list[float],
    environment_config: dict | None = None,
) -> list[CachedBCDecision]:
    """Replay one demo episode into frozen exact-mask BC decisions."""
    env_cfg = environment_config or {}
    name = ep["instance_name"]
    env = EVRPTWEnv(
        instance,
        m=int(ep["m"]),
        charging_portions=list(charging_portions),
        use_horizon_check=True,
        use_depot_charging=True,
        hard_time_windows=True,
    )
    env.environment_config = env_cfg
    cache: list[CachedBCDecision] = []
    action_i = 0
    while not env.is_done() and action_i < len(ep["actions"]):
        ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
            continue
        if env.try_auto_close_from_analysis(ev, analysis) is not None:
            continue
        a = ep["actions"][action_i]
        action_i += 1
        if analysis.mask_result is None:
            continue
        if analysis.global_features is None or analysis.mask_summary_features is None:
            continue
        if analysis.dynamic_candidate_features is None:
            continue
        cache.append(
            CachedBCDecision(
                instance_name=name,
                expert_node=int(a["node"]),
                expert_charge_portion=(
                    None if a.get("charge_portion") is None else float(a["charge_portion"])
                ),
                expert_depot_finish=bool(a.get("depot_finish", False)),
                mask_result=analysis.mask_result,
                acting_ev_feats=np.asarray(acting_ev_features(env, ev), dtype=np.float32),
                token_scalars=np.stack(
                    [ev_token_scalars(env, e) for e in env.evs]
                ).astype(np.float32),
                fleet_balance=np.asarray(fleet_balance_features(env), dtype=np.float32),
                finished=np.array([e.finished for e in env.evs], dtype=bool),
                current_nodes=np.array([e.current_node for e in env.evs], dtype=np.int64),
                acting_idx=[e.ev_id for e in env.evs].index(ev.ev_id),
                global_features=np.asarray(analysis.global_features, dtype=np.float32),
                mask_summary_features=np.asarray(
                    analysis.mask_summary_features, dtype=np.float32,
                ),
                dynamic_candidate_features=np.asarray(
                    analysis.dynamic_candidate_features, dtype=np.float32,
                ),
                battery_feats=np.asarray(battery_state_features(env, ev), dtype=np.float32),
            )
        )
        expert_action = PolicyAction(
            node=int(a["node"]),
            charge_portion=a.get("charge_portion"),
            depot_finish=bool(a.get("depot_finish", False)),
        )
        apply_policy_action(env, ev, expert_action, mask_result=analysis.mask_result)
    return cache


def _parallel_bc_cache_job(job: dict) -> tuple[str, list[CachedBCDecision]]:
    """Process-pool worker for one demo episode → BC cache entries."""
    from src.utils.gpu_runtime import limit_cpu_threads_for_worker

    limit_cpu_threads_for_worker(1)
    ep = job["episode"]
    name = str(ep["instance_name"])
    entries = _cache_decisions_for_one_episode(
        ep,
        job["instance"],
        charging_portions=list(job["charging_portions"]),
        environment_config=job.get("environment_config") or {},
    )
    return name, entries


def build_bc_decision_cache(
    episodes: list,
    instance_by_name: dict,
    *,
    charging_portions: list[float],
    environment_config: dict | None = None,
    num_workers: int = 1,
) -> list[CachedBCDecision]:
    """Replay expert demos once and freeze exact-mask features for BC.

    Exact legality is identical to live ``analyze_decision`` at demo time;
    later BC only re-forwards the policy (GPU), not CPU feasibility.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    env_cfg = environment_config or {}
    jobs = []
    for ep in episodes:
        name = ep["instance_name"]
        if name not in instance_by_name:
            continue
        jobs.append({
            "episode": ep,
            "instance": instance_by_name[name],
            "charging_portions": list(charging_portions),
            "environment_config": env_cfg,
        })

    n_workers = int(num_workers or 1)
    if n_workers <= 0:
        n_workers = min(8, max(1, (os.cpu_count() or 2) - 1))
    n_workers = max(1, min(n_workers, len(jobs) or 1))

    LOGGER.info(
        "BC decision cache | episodes=%d workers=%d "
        "(one-time exact-mask replay; progress logs below)",
        len(jobs), n_workers,
    )

    cache: list[CachedBCDecision] = []
    if n_workers <= 1 or len(jobs) <= 1:
        for i, job in enumerate(jobs):
            name = job["episode"]["instance_name"]
            LOGGER.info("BC cache episode %d/%d: %s …", i + 1, len(jobs), name)
            t0 = time.perf_counter()
            entries = _cache_decisions_for_one_episode(
                job["episode"],
                job["instance"],
                charging_portions=list(charging_portions),
                environment_config=env_cfg,
            )
            cache.extend(entries)
            LOGGER.info(
                "BC cache episode %d/%d: %s done | decisions=%d in %.1fs",
                i + 1, len(jobs), name, len(entries), time.perf_counter() - t0,
            )
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            for i, (name, entries) in enumerate(pool.map(_parallel_bc_cache_job, jobs)):
                cache.extend(entries)
                LOGGER.info(
                    "BC cache episode %d/%d: %s done | decisions=%d",
                    i + 1, len(jobs), name, len(entries),
                )

    LOGGER.info(
        "BC decision cache built | decisions=%d from %d demo episodes",
        len(cache), len(jobs),
    )
    return cache


def bc_loss_on_cached_decision(
    policy: TransformerEVRPTWPolicy,
    cached: CachedBCDecision,
    node_embeddings: torch.Tensor,
    instance,
) -> tuple[torch.Tensor, dict]:
    """CE on one cached exact-mask decision (policy forward only)."""
    mask_result = cached.mask_result
    state = policy.build_decision_state_from_arrays(
        node_embeddings,
        mask_result,
        acting_ev_feats=cached.acting_ev_feats,
        token_scalars_np=cached.token_scalars,
        fleet_balance_np=cached.fleet_balance,
        finished_np=cached.finished,
        current_nodes=cached.current_nodes,
        acting_idx=cached.acting_idx,
        global_features=cached.global_features,
        mask_summary_features=cached.mask_summary_features,
        dynamic_candidate_features=cached.dynamic_candidate_features,
        instance_name=cached.instance_name,
    )
    device = node_embeddings.device
    dtype = node_embeddings.dtype
    scores = policy.pointer.node_scores(
        state.decoder_context,
        state.candidate_embeddings,
        state.dynamic_candidate_embeddings,
    )
    node_mask = torch.tensor(state.feasible_mask, dtype=torch.bool, device=device)
    node_loss = masked_ce_from_logits(scores, int(cached.expert_node), node_mask)

    charge_loss = torch.zeros((), device=device)
    depot_loss = torch.zeros((), device=device)
    stats = {
        "node_correct": 0.0,
        "charge_correct": 0.0,
        "depot_correct": 0.0,
        "charge_used": 0.0,
        "depot_used": 0.0,
    }
    with torch.no_grad():
        pred_node = int(torch.argmax(safe_mask_logits(scores, node_mask)).item())
        stats["node_correct"] = float(pred_node == int(cached.expert_node))

    node = int(cached.expert_node)
    inst = instance
    if inst.nodes[node].is_depot:
        ok_r = bool(mask_result.depot_mode_mask[0])
        ok_f = bool(mask_result.depot_mode_mask[1])
        if ok_r and ok_f:
            mode_logits = policy.pointer.depot_mode_logits(
                state.decoder_context,
                state.candidate_embeddings[node],
                state.global_context,
                state.mask_context,
            )
            dm = torch.tensor(mask_result.depot_mode_mask, device=device, dtype=torch.bool)
            target_mode = 1 if cached.expert_depot_finish else 0
            depot_loss = masked_ce_from_logits(mode_logits, target_mode, dm)
            stats["depot_used"] = 1.0
            with torch.no_grad():
                pred = int(torch.argmax(safe_mask_logits(mode_logits, dm)).item())
                stats["depot_correct"] = float(pred == target_mode)
        if not cached.expert_depot_finish and cached.expert_charge_portion is not None:
            charge_loss, cstats = _charge_loss_from_battery(
                policy, state, node, float(cached.expert_charge_portion), cached.battery_feats,
            )
            stats.update(cstats)
    elif inst.nodes[node].is_charging_station:
        if cached.expert_charge_portion is None:
            raise RuntimeError("Cached station action missing charge portion")
        charge_loss, cstats = _charge_loss_from_battery(
            policy, state, node, float(cached.expert_charge_portion), cached.battery_feats,
        )
        stats.update(cstats)

    loss = node_loss + 0.5 * charge_loss + 0.5 * depot_loss
    return loss, stats


def _charge_loss_from_battery(policy, state, station: int, portion: float, battery_feats: np.ndarray):
    device = state.decoder_context.device
    dtype = state.decoder_context.dtype
    batt = torch.tensor(battery_feats, device=device, dtype=dtype)
    logits = policy.pointer.charging_logits(
        state.decoder_context,
        state.candidate_embeddings[station],
        state.acting_ev_context,
        batt,
    )
    cm = state.mask_result.charge_masks[station]
    mask_t = torch.tensor(cm, device=device, dtype=torch.bool)
    idx = _portion_index(list(policy.charging_portions), portion)
    loss = masked_ce_from_logits(logits, idx, mask_t)
    stats = {"charge_used": 1.0, "charge_correct": 0.0}
    with torch.no_grad():
        pred = int(torch.argmax(safe_mask_logits(logits, mask_t)).item())
        stats["charge_correct"] = float(pred == idx)
    return loss, stats


def bc_loss_on_cached_batch(
    policy: TransformerEVRPTWPolicy,
    cache: list[CachedBCDecision],
    instance_by_name: dict,
    instance_embeddings: dict[str, torch.Tensor],
    *,
    device: torch.device,
    batch_decisions: int = 32,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, dict]:
    """Sample cached decisions and accumulate masked CE (no CPU re-analysis)."""
    rng = rng or random.Random(0)
    if not cache:
        return torch.zeros((), device=device), {}
    n = max(1, int(batch_decisions))
    chosen = [cache[rng.randrange(len(cache))] for _ in range(n)]
    losses = []
    agg = {
        "node_correct": 0.0, "charge_correct": 0.0, "depot_correct": 0.0,
        "charge_used": 0.0, "depot_used": 0.0, "n": 0.0,
    }
    for sample in chosen:
        H = instance_embeddings[sample.instance_name]
        inst = instance_by_name[sample.instance_name]
        loss, stats = bc_loss_on_cached_decision(policy, sample, H, inst)
        losses.append(loss)
        for k in ("node_correct", "charge_correct", "depot_correct", "charge_used", "depot_used"):
            agg[k] += float(stats.get(k, 0.0))
        agg["n"] += 1.0
    total = torch.stack(losses).mean()
    n_dec = max(agg["n"], 1.0)
    metrics = {
        "bc_loss": float(total.detach()),
        "node_accuracy": agg["node_correct"] / n_dec,
        "charge_accuracy": agg["charge_correct"] / max(agg["charge_used"], 1.0),
        "depot_accuracy": agg["depot_correct"] / max(agg["depot_used"], 1.0),
        "n_decisions": n_dec,
    }
    return total, metrics


def evaluate_bc_policy(config: dict, policy: TransformerEVRPTWPolicy, device: torch.device) -> dict:
    """Greedy rollout of the cloned policy on train instances (compact report)."""
    train_names, eval_names, _ = resolve_training_phase_split(config)
    ds_cfg = dict(config["dataset"])
    train_ds, eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=int(config.get("seed", 42)),
        train_filter=ds_cfg.get("train_size_filter") or ["_21"],
        eval_filter=ds_cfg.get("eval_size_filter") or ["_21"],
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_capacity_from_config(config),
    )
    weights = CostWeights.from_config(config["cost"])
    extra_margin = int(config["fleet_search"].get("extra_margin", 4))
    max_steps_factor = int(config["decoding"].get("max_steps_factor", 6))
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    use_depot = bool(config["problem"].get("use_depot_charging", True))
    env_cfg = config.get("environment") or {}
    rng = random.Random(int(config.get("seed", 42)))
    policy.eval()
    rows = []
    # Evaluate on holdout preferentially.
    ds = eval_ds if len(eval_ds) else train_ds
    for i in range(len(ds)):
        instance = ds[i]
        m = sample_fleet_size(instance, extra_margin, rng)
        env = EVRPTWEnv(
            instance, m=m,
            charging_portions=list(policy.charging_portions),
            use_horizon_check=use_horizon,
            use_depot_charging=use_depot,
            max_steps_factor=max_steps_factor,
            hard_time_windows=True,
        )
        env.environment_config = env_cfg
        H = policy.encode(instance, device=device)
        while not env.is_done():
            ev = env.select_acting_ev()
            if ev is None:
                break
            analysis = env.analyze_decision(ev)
            if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
                continue
            if env.try_auto_close_from_analysis(ev, analysis) is not None:
                continue
            with torch.no_grad():
                out = policy.select_action(
                    env, ev, H, analysis.mask_result, mode="greedy", analysis=analysis,
                )
            apply_policy_action(env, ev, out, mask_result=analysis.mask_result)
        for ev in env.evs:
            if not ev.finished:
                env.auto_close_route(ev, forced=True)
        j, _, summary = env.score(weights)
        cust_a = float(getattr(env, "customer_actions", 0))
        st_a = float(getattr(env, "station_actions", 0))
        dep_a = float(getattr(env, "depot_finish_count", 0)) + float(
            getattr(env, "depot_recharge_count", 0)
        )
        n_act = max(cust_a + st_a + dep_a, 1.0)
        rows.append({
            "instance_name": instance.instance_name,
            "served_ratio": env.num_served_customers / max(instance.num_customers, 1),
            "feasible": env.feasible_solution(),
            "objective": float(j),
            "unserved_customers": int(summary.unserved_customers),
            "battery_violations": int(env.battery_violation_count),
            "time_window_violations": int(env.time_window_violation_count),
            "max_step_terminations": int(bool(getattr(env, "max_steps_reached", False))),
            "depot_finish_count": float(getattr(env, "depot_finish_count", 0)),
            "station_action_fraction": float(st_a / n_act),
            "depot_lateness": float(summary.depot_return_violation),
        })
    n = max(len(rows), 1)
    report = {
        "num_instances": len(rows),
        "mean_served_ratio": float(sum(r["served_ratio"] for r in rows) / n),
        "feasibility_rate": float(sum(1 for r in rows if r["feasible"]) / n),
        "mean_objective": float(sum(r["objective"] for r in rows) / n),
        "mean_unserved": float(sum(r["unserved_customers"] for r in rows) / n),
        "mean_battery_violations": float(sum(r["battery_violations"] for r in rows) / n),
        "mean_tw_violations": float(sum(r["time_window_violations"] for r in rows) / n),
        "mean_max_steps": float(sum(r["max_step_terminations"] for r in rows) / n),
        "mean_depot_finish_count": float(sum(r["depot_finish_count"] for r in rows) / n),
        "mean_station_action_fraction": float(sum(r["station_action_fraction"] for r in rows) / n),
        "mean_depot_lateness": float(sum(r["depot_lateness"] for r in rows) / n),
        "per_instance": rows,
    }
    LOGGER.info(
        "BC greedy eval | served=%.3f feas=%.3f J=%.1f unserved=%.1f",
        report["mean_served_ratio"], report["feasibility_rate"],
        report["mean_objective"], report["mean_unserved"],
    )
    return report


def train_behavior_cloning(config: dict) -> dict:
    set_seed(int(config.get("seed", 42)), deterministic=bool(config.get("deterministic", False)))
    device = resolve_device(config.get("device", "auto"))
    from src.utils.gpu_runtime import configure_cuda_runtime

    gpu_cfg = config.get("gpu") or {}
    configure_cuda_runtime(
        cudnn_benchmark=bool(gpu_cfg.get("cudnn_benchmark", True)),
        allow_tf32=bool(gpu_cfg.get("allow_tf32", True)),
    )
    LOGGER.info("BC training on %s", describe_device(device))

    demo_path = Path((config.get("demonstrations") or {}).get("input", "outputs/demonstrations/rc21_expert.pt"))
    if not demo_path.is_file():
        raise FileNotFoundError(
            f"Demonstration file not found: {demo_path}. "
            "Run the full pipeline first: python main_train.py --config configs/train.yaml"
        )
    payload = torch.load(demo_path, map_location="cpu", weights_only=False)

    train_names, eval_names, _ = resolve_training_phase_split(config)
    ds_cfg = dict(config["dataset"])
    from src.data.hanif_dataset import is_hanif_config

    train_filter = ds_cfg.get("train_size_filter")
    eval_filter = ds_cfg.get("eval_size_filter")
    if is_hanif_config(config):
        train_filter = train_filter if train_filter is not None else []
        eval_filter = eval_filter if eval_filter is not None else []
    else:
        train_filter = train_filter or ["_21"]
        eval_filter = eval_filter or ["_21"]

    train_ds, _eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=int(config.get("seed", 42)),
        train_filter=train_filter,
        eval_filter=eval_filter,
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_capacity_from_config(config),
    )
    instance_by_name = {n: train_ds[train_ds.names.index(n)] for n in train_ds.names}

    portions = list(config["problem"]["charging_portions"])
    policy = build_policy(config, portions).to(device)
    bcfg = config.get("behavior_cloning") or config.get("bc") or {}
    if bool(bcfg.get("early_stopping", False)):
        LOGGER.warning(
            "behavior_cloning.early_stopping=true ignored — BC always runs all configured epochs"
        )
    lr = float(bcfg.get("learning_rate", 1e-4))
    epochs = int(bcfg.get("epochs", 20))
    # Decisions per optimizer step (GPU only after the one-time exact-mask cache).
    batch_decisions = int(bcfg.get("batch_decisions", bcfg.get("batch_size", 64)))
    updates_per_epoch = max(int(bcfg.get("updates_per_epoch", 8)), 1)
    log_every = int(bcfg.get("log_every", 1))
    run_greedy_eval = bool(bcfg.get("run_greedy_eval", False))
    out_dir = Path(bcfg.get("output_dir", "outputs/bc"))
    out_dir.mkdir(parents=True, exist_ok=True)
    best_name = str(bcfg.get("best_checkpoint_name", "bc_best_policy.pt"))
    last_name = str(bcfg.get("last_checkpoint_name", "bc_last_policy.pt"))
    metrics_csv_name = str(bcfg.get("metrics_csv_name", "bc_metrics.csv"))
    eval_json_name = str(bcfg.get("eval_json_name", "bc_eval_metrics.json"))
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = random.Random(int(config.get("seed", 42)))
    env_cfg = config.get("environment") or {}

    # Prefer cached decision-level dataset (no episode replay during BC).
    if payload.get("decisions") is not None:
        decision_cache = list(payload["decisions"])
        LOGGER.info(
            "Loaded %d cached BC decisions from %s (no episode replay)",
            len(decision_cache), demo_path,
        )
    else:
        episodes = list(payload.get("episodes") or [])
        LOGGER.info("Loaded %d demo episodes from %s", len(episodes), demo_path)
        demo_workers = int((config.get("demonstrations") or {}).get("num_workers", 8) or 8)
        LOGGER.info("Building BC decision cache (one-time exact-mask replay)…")
        decision_cache = build_bc_decision_cache(
            episodes,
            instance_by_name,
            charging_portions=portions,
            environment_config=env_cfg,
            num_workers=demo_workers,
        )
    if not decision_cache:
        raise RuntimeError("BC decision cache is empty — demos produced no usable decisions")

    policy.eval()
    instance_embeddings: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, inst in instance_by_name.items():
            instance_embeddings[name] = policy.encode(inst, device=device)
    policy.train()

    best_val = float("inf")
    best_path = out_dir / best_name
    history = []
    LOGGER.info(
        "BC training for %d epochs × %d updates (batch_decisions=%d, cache=%d) — no early stopping",
        epochs, updates_per_epoch, batch_decisions, len(decision_cache),
    )

    for epoch in range(epochs):
        policy.train()
        epoch_metrics = {
            "bc_loss": 0.0, "node_accuracy": 0.0, "charge_accuracy": 0.0,
            "depot_accuracy": 0.0, "n_decisions": 0.0,
        }
        for _ in range(updates_per_epoch):
            loss, metrics = bc_loss_on_cached_batch(
                policy,
                decision_cache,
                instance_by_name,
                instance_embeddings,
                device=device,
                batch_decisions=batch_decisions,
                rng=rng,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), float(bcfg.get("grad_clip", 1.0)))
            optimizer.step()
            for k in epoch_metrics:
                epoch_metrics[k] += float(metrics.get(k, 0.0))
        for k in epoch_metrics:
            epoch_metrics[k] /= float(updates_per_epoch)
        row = {"epoch": epoch, **epoch_metrics}
        history.append(row)
        if epoch % log_every == 0:
            LOGGER.info(
                "BC epoch %d/%d | loss=%.4f node_acc=%.3f charge_acc=%.3f depot_acc=%.3f",
                epoch + 1, epochs, epoch_metrics.get("bc_loss", 0),
                epoch_metrics.get("node_accuracy", 0),
                epoch_metrics.get("charge_accuracy", 0),
                epoch_metrics.get("depot_accuracy", 0),
            )
        if epoch_metrics.get("bc_loss", 1e9) < best_val:
            best_val = float(epoch_metrics["bc_loss"])
            torch.save({
                "architecture_version": getattr(
                    __import__("src.models.feature_specs", fromlist=["ARCHITECTURE_VERSION"]),
                    "ARCHITECTURE_VERSION",
                    "exact_smdp_dynamic_v1",
                ),
                "model_state": policy.state_dict(),
                "model_config": config["model"],
                "charging_portions": portions,
                "bc_epoch": epoch,
                "bc_metrics": epoch_metrics,
            }, best_path)
            LOGGER.info("Saved best BC policy → %s (epoch %d)", best_path, epoch + 1)

    last_path = out_dir / last_name
    torch.save({
        "architecture_version": "exact_smdp_dynamic_v1",
        "model_state": policy.state_dict(),
        "model_config": config["model"],
        "charging_portions": portions,
        "bc_epoch": epochs - 1,
    }, last_path)
    if not best_path.is_file():
        shutil_copy = __import__("shutil").copy2
        shutil_copy(last_path, best_path)

    import csv as _csv
    metrics_csv = out_dir / metrics_csv_name
    if history:
        keys = sorted({k for r in history for k in r.keys()})
        with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for r in history:
                w.writerow(r)

    eval_report: dict = {"skipped": True, "reason": "run_greedy_eval=false"}
    if run_greedy_eval:
        policy = load_policy(best_path, device=device)
        eval_report = evaluate_bc_policy(config, policy, device)
    summary = {
        "best_bc_loss": best_val,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "epochs_completed": epochs,
        "early_stopping": False,
        "history": history,
        "greedy_eval": eval_report,
        "cached_decisions": len(decision_cache),
        "updates_per_epoch": updates_per_epoch,
        "batch_decisions": batch_decisions,
    }
    eval_path = out_dir / eval_json_name
    eval_path.write_text(json.dumps(eval_report, indent=2, default=str), encoding="utf-8")
    (out_dir / "bc_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOGGER.info("BC complete after %d epochs. Best checkpoint: %s", epochs, best_path)
    return summary


def bc_coef_at_update(rl_cfg_dict: dict, update_idx: int) -> float:
    """Linear decay of BC regularization coefficient (0 when skipped by every_n)."""
    raw = (rl_cfg_dict.get("bc_regularization") or {})
    if not bool(raw.get("enabled", False)):
        return 0.0
    every_n = max(int(raw.get("every_n_updates", 1)), 1)
    if int(update_idx) % every_n != 0:
        return 0.0
    c0 = float(raw.get("coefficient_initial", 0.1))
    c1 = float(raw.get("coefficient_final", 0.0))
    decay = max(int(raw.get("decay_updates", 200)), 1)
    t = min(max(int(update_idx), 0), decay) / float(decay)
    return float(c0 + (c1 - c0) * t)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Behavior-clone Transformer actor from expert demos.")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    train_behavior_cloning(config)


if __name__ == "__main__":
    main()
