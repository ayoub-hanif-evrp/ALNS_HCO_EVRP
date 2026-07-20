"""Generate / evaluate feasibility-first expert demonstrations on RC _21."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.data.instance_class import instance_class
from src.data.instance_loader import load_capacity_from_config
from src.data.training_phases import resolve_training_phase_split
from src.data.evrptw_dataset import split_dataset
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import apply_policy_action
from src.env.reward import StepRewardCalculator, snapshot_env
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.training.flat_schedule import sample_fleet_size
from src.utils.config import load_config
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def _episode_to_demo_dict(ep: EpisodeResult) -> dict:
    return {
        "instance_name": ep.instance_name,
        "m": ep.m,
        "actions": ep.actions,
        "metrics": ep.metrics,
        "transitions": [
            {
                **{k: v for k, v in asdict(t).items() if k not in {
                    "node_mask", "charge_mask_for_selected_node", "depot_mode_mask",
                }},
                "node_mask": t.node_mask,
                "charge_mask_for_selected_node": t.charge_mask_for_selected_node,
                "depot_mode_mask": t.depot_mode_mask,
            }
            for t in ep.transitions
        ],
    }


def _parallel_demo_job(job: dict) -> dict:
    """Process-pool worker: one exact-mask expert episode (Windows-safe top-level)."""
    cfg = job["config"]
    expert = FeasibilityFirstExpert(cfg)
    weights = CostWeights.from_config(cfg["cost"])
    problem = cfg.get("problem") or {}
    env_cfg = cfg.get("environment") or {}
    ep = run_expert_episode(
        job["instance"],
        int(job["m"]),
        expert,
        weights,
        charging_portions=list(problem.get("charging_portions") or [0.25, 0.5, 0.75, 1.0]),
        use_horizon_check=bool(problem.get("use_horizon_check", True)),
        use_depot_charging=bool(problem.get("use_depot_charging", True)),
        max_steps_factor=int((cfg.get("decoding") or {}).get("max_steps_factor", 6)),
        environment_config=env_cfg,
        reward_scale=float((cfg.get("rl") or {}).get("reward_scale", 10000.0)),
        collect_transitions=True,
    )
    return _episode_to_demo_dict(ep)


@dataclass
class ExpertTransition:
    instance_name: str
    class_name: str
    state_features: dict
    analysis_summary: dict
    action_node: int
    action_charge_portion: float | None
    action_depot_finish: bool | None
    node_mask: np.ndarray
    charge_mask_for_selected_node: np.ndarray | None
    depot_mode_mask: np.ndarray | None
    reward: float
    original_cost_delta: float
    served_count_before: int
    served_count_after: int
    remaining_unserved: int
    terminal: bool


@dataclass
class EpisodeResult:
    instance_name: str
    m: int
    transitions: list[ExpertTransition] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)


def _analysis_summary(analysis) -> dict:
    gs = analysis.global_serviceability
    return {
        "state_version": int(analysis.state_version),
        "active_ev_id": int(analysis.active_ev_id),
        "final_feasible_nodes": int(np.sum(analysis.final_node_mask)),
        "feasible_customers": int(analysis.mask_result.feasible_customer_count)
        if analysis.mask_result else 0,
        "feasible_stations": int(analysis.mask_result.feasible_station_count)
        if analysis.mask_result else 0,
        "depot_recharge": bool(analysis.depot_mode_mask[0])
        if analysis.depot_mode_mask is not None and len(analysis.depot_mode_mask) > 0 else False,
        "depot_finish": bool(analysis.depot_mode_mask[1])
        if analysis.depot_mode_mask is not None and len(analysis.depot_mode_mask) > 1 else False,
        "globally_serviceable": int(getattr(gs, "globally_serviceable_customers_count", 0) or 0),
        "globally_unserviceable": int(getattr(gs, "globally_unserviceable_customers_count", 0) or 0),
    }


def _state_features(env, ev) -> dict:
    return {
        "ev_id": int(ev.ev_id),
        "current_node": int(ev.current_node),
        "current_time": float(ev.current_time),
        "remaining_battery": float(ev.remaining_battery),
        "remaining_capacity": float(ev.remaining_capacity),
        "served_customers": int(env.num_served_customers),
        "unserved_customers": int(env.instance.num_customers - env.num_served_customers),
        "state_version": int(getattr(env, "state_version", 0)),
    }


def run_expert_episode(
    instance,
    m: int,
    expert: FeasibilityFirstExpert,
    weights: CostWeights,
    *,
    charging_portions: list[float],
    use_horizon_check: bool = True,
    use_depot_charging: bool = True,
    max_steps_factor: int = 6,
    environment_config: dict | None = None,
    reward_scale: float = 10000.0,
    collect_transitions: bool = True,
) -> EpisodeResult:
    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=charging_portions,
        use_horizon_check=use_horizon_check,
        use_depot_charging=use_depot_charging,
        max_steps_factor=max_steps_factor,
        hard_time_windows=True,
    )
    if environment_config is not None:
        env.environment_config = environment_config

    calc = StepRewardCalculator(weights, reward_scale=reward_scale)
    result = EpisodeResult(instance_name=instance.instance_name, m=m)
    name = instance.instance_name
    cls = instance_class(name)

    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
            continue
        if env.try_auto_close_from_analysis(ev, analysis) is not None:
            continue

        served_before = int(env.num_served_customers)
        prev_node = int(ev.current_node)
        before = snapshot_env(env)
        action = expert.select_action(env, ev, analysis)
        apply_policy_action(env, ev, action, mask_result=analysis.mask_result)
        after = snapshot_env(env)
        info = calc.step_reward(
            before, after,
            ev_id=int(ev.ev_id),
            selected_node=int(action.node),
            node_type="action",
            previous_node=prev_node,
        )
        served_after = int(env.num_served_customers)

        result.actions.append({
            "node": int(action.node),
            "charge_portion": None if action.charge_portion is None else float(action.charge_portion),
            "depot_finish": bool(action.depot_finish),
        })

        if collect_transitions:
            cm_sel = None
            if action.charge_portion is not None:
                raw = analysis.final_charge_masks.get(int(action.node))
                cm_sel = None if raw is None else np.asarray(raw, dtype=bool).copy()
            tr = ExpertTransition(
                instance_name=name,
                class_name=cls,
                state_features=_state_features(env, ev),
                analysis_summary=_analysis_summary(analysis),
                action_node=int(action.node),
                action_charge_portion=(
                    None if action.charge_portion is None else float(action.charge_portion)
                ),
                action_depot_finish=bool(action.depot_finish),
                node_mask=np.asarray(analysis.final_node_mask, dtype=bool).copy(),
                charge_mask_for_selected_node=cm_sel,
                depot_mode_mask=(
                    None if analysis.depot_mode_mask is None
                    else np.asarray(analysis.depot_mode_mask, dtype=bool).copy()
                ),
                reward=float(info.reward),
                original_cost_delta=float(info.delta_objective),
                served_count_before=served_before,
                served_count_after=served_after,
                remaining_unserved=int(instance.num_customers - served_after),
                terminal=False,
            )
            result.transitions.append(tr)

    for ev in env.evs:
        if not ev.finished:
            env.auto_close_route(ev, forced=True)

    j, breakdown, summary = env.score(weights)
    diag = env.get_episode_diagnostics()
    cust_a = float(getattr(env, "customer_actions", 0))
    st_a = float(getattr(env, "station_actions", 0))
    dep_a = float(getattr(env, "depot_finish_count", 0)) + float(getattr(env, "depot_recharge_count", 0))
    n_act = max(cust_a + st_a + dep_a, 1.0)
    result.metrics = {
        "objective": float(j),
        "feasible": bool(env.feasible_solution()),
        "served_ratio": float(env.num_served_customers / max(instance.num_customers, 1)),
        "unserved_customers": int(summary.unserved_customers),
        "used_evs": int(summary.used_evs),
        "station_action_fraction": float(st_a / n_act),
        "depot_finish_count": float(getattr(env, "depot_finish_count", 0)),
        "battery_violations": int(env.battery_violation_count),
        "time_window_violations": int(env.time_window_violation_count),
        "customer_lateness": float(env.total_lateness),
        "depot_lateness": float(summary.depot_return_violation),
        "capacity_violations": int(env.capacity_violation_count),
        "max_step_terminations": int(bool(getattr(env, "max_steps_reached", False))),
        "done_reason": env.get_done_reason(),
        "num_decisions": len(result.actions),
        "breakdown": {k: float(v) for k, v in (breakdown or {}).items()},
        **{k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v
           for k, v in diag.items() if k in {
               "termination_reason", "max_steps_reached", "success_all_served_all_returned",
           }},
    }
    # Objective reconstruction from step deltas (training reward path).
    if result.transitions:
        recon = sum(t.original_cost_delta for t in result.transitions)
        result.metrics["reconstructed_objective"] = float(recon)
        result.metrics["objective_reconstruction_mismatch"] = abs(float(recon) - float(j))
        result.metrics["relative_objective_mismatch"] = abs(float(recon) - float(j)) / max(1.0, abs(float(j)))
    else:
        result.metrics["objective_reconstruction_mismatch"] = 0.0
        result.metrics["relative_objective_mismatch"] = 0.0
    return result


def _load_train_eval(config: dict):
    seed = int(config.get("seed", 42))
    ds_cfg = dict(config["dataset"])
    train_names, eval_names, meta = resolve_training_phase_split(config)
    train_ds, eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=seed,
        train_filter=ds_cfg.get("train_size_filter") or ["_21"],
        eval_filter=ds_cfg.get("eval_size_filter") or ["_21"],
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_capacity_from_config(config),
    )
    return train_ds, eval_ds, meta


def aggregate_expert_report(
    rows: list[dict],
    *,
    split: str = "train",
    meta: dict | None = None,
) -> dict:
    """Build the standard expert metrics report from per-instance rows."""
    meta = meta or {}

    def _mean(key: str) -> float:
        vals = [float(r.get(key, 0.0) or 0.0) for r in rows]
        return float(sum(vals) / max(len(vals), 1))

    return {
        "split": split,
        "num_instances": len(rows),
        "mean_served_ratio": _mean("served_ratio"),
        "feasibility_rate": float(sum(1 for r in rows if r.get("feasible")) / max(len(rows), 1)),
        "mean_objective": _mean("objective"),
        "mean_unserved_customers": _mean("unserved_customers"),
        "mean_station_action_fraction": _mean("station_action_fraction"),
        "mean_depot_finish_count": _mean("depot_finish_count"),
        "mean_battery_violations": _mean("battery_violations"),
        "mean_time_window_violations": _mean("time_window_violations"),
        "mean_depot_lateness": _mean("depot_lateness"),
        "mean_capacity_violations": _mean("capacity_violations"),
        "mean_max_step_terminations": _mean("max_step_terminations"),
        "mean_relative_objective_mismatch": _mean("relative_objective_mismatch"),
        "hard_violation_free": all(
            float(r.get("battery_violations", 0)) == 0
            and float(r.get("time_window_violations", 0)) == 0
            and float(r.get("capacity_violations", 0)) == 0
            for r in rows
        ),
        "per_instance": rows,
        "meta": {
            k: meta.get(k)
            for k in ("training_phase", "train_instances", "eval_instances")
            if k in meta
        },
    }


def expert_report_from_demo_episodes(
    episodes: list[dict],
    *,
    split: str = "train",
    meta: dict | None = None,
) -> dict:
    """Derive expert evaluation metrics from demonstration episodes (no re-rollout)."""
    rows = []
    for ep in episodes:
        row = dict(ep.get("metrics") or {})
        row["instance_name"] = ep.get("instance_name")
        row["m"] = ep.get("m")
        rows.append(row)
    return aggregate_expert_report(rows, split=split, meta=meta)


def log_expert_report(report: dict) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("Expert evaluation (%s)", report["split"])
    LOGGER.info("  Served ratio: %.4f", report["mean_served_ratio"])
    LOGGER.info("  Feasibility rate: %.4f", report["feasibility_rate"])
    LOGGER.info("  Original objective: %.2f", report["mean_objective"])
    LOGGER.info("  Unserved customers: %.2f", report["mean_unserved_customers"])
    LOGGER.info("  Station-action fraction: %.4f", report["mean_station_action_fraction"])
    LOGGER.info("  Depot finish count: %.2f", report["mean_depot_finish_count"])
    LOGGER.info("  Battery violations: %.4f", report["mean_battery_violations"])
    LOGGER.info("  TW violations: %.4f", report["mean_time_window_violations"])
    LOGGER.info("  Depot lateness: %.4f", report["mean_depot_lateness"])
    LOGGER.info("  Capacity violations: %.4f", report["mean_capacity_violations"])
    LOGGER.info("  Max-step terminations: %.4f", report["mean_max_step_terminations"])
    LOGGER.info("  Obj reconstruction mismatch (rel): %.6f", report["mean_relative_objective_mismatch"])
    LOGGER.info("  Hard-violation-free: %s", report["hard_violation_free"])
    LOGGER.info("=" * 60)


def evaluate_expert(config: dict, *, on_holdout: bool = False) -> dict:
    train_ds, eval_ds, meta = _load_train_eval(config)
    ds = eval_ds if on_holdout else train_ds
    expert = FeasibilityFirstExpert(config)
    weights = CostWeights.from_config(config["cost"])
    portions = list(config["problem"]["charging_portions"])
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    use_depot = bool(config["problem"].get("use_depot_charging", True))
    max_steps_factor = int(config["decoding"].get("max_steps_factor", 6))
    extra_margin = int(config["fleet_search"].get("extra_margin", 4))
    env_cfg = config.get("environment") or {}
    rng = __import__("random").Random(int(config.get("seed", 42)))

    rows = []
    for i in range(len(ds)):
        instance = ds[i]
        m = sample_fleet_size(instance, extra_margin, rng)
        t0 = time.time()
        ep = run_expert_episode(
            instance, m, expert, weights,
            charging_portions=portions,
            use_horizon_check=use_horizon,
            use_depot_charging=use_depot,
            max_steps_factor=max_steps_factor,
            environment_config=env_cfg,
            reward_scale=float((config.get("rl") or {}).get("reward_scale", 10000.0)),
            collect_transitions=False,
        )
        row = dict(ep.metrics)
        row["instance_name"] = instance.instance_name
        row["m"] = m
        row["runtime_seconds"] = time.time() - t0
        rows.append(row)
        LOGGER.info(
            "Expert %s | served=%.3f feas=%s J=%.1f unserved=%d decisions=%d",
            instance.instance_name, row["served_ratio"], row["feasible"],
            row["objective"], row["unserved_customers"], row["num_decisions"],
        )

    report = aggregate_expert_report(
        rows,
        split="holdout" if on_holdout else "train",
        meta=meta,
    )
    log_expert_report(report)
    return report


def generate_demonstrations(config: dict) -> Path:
    train_ds, _eval_ds, meta = _load_train_eval(config)
    expert = FeasibilityFirstExpert(config)
    weights = CostWeights.from_config(config["cost"])
    portions = list(config["problem"]["charging_portions"])
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    use_depot = bool(config["problem"].get("use_depot_charging", True))
    max_steps_factor = int(config["decoding"].get("max_steps_factor", 6))
    extra_margin = int(config["fleet_search"].get("extra_margin", 4))
    env_cfg = config.get("environment") or {}
    out_path = Path((config.get("demonstrations") or {}).get("output", "outputs/demonstrations/rc21_expert.pt"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rng = __import__("random").Random(int(config.get("seed", 42)))

    # Sample fleet sizes first so parallel runs stay seed-reproducible.
    jobs = []
    for i in range(len(train_ds)):
        instance = train_ds[i]
        m = sample_fleet_size(instance, extra_margin, rng)
        jobs.append({"instance": instance, "m": m, "config": config})

    demo_cfg = config.get("demonstrations") or {}
    import os
    raw_workers = int(demo_cfg.get("num_workers", 0) or 0)
    n_workers = raw_workers if raw_workers > 0 else min(8, max(1, (os.cpu_count() or 2) - 1))
    n_workers = max(1, min(n_workers, len(jobs) or 1))

    episodes: list[dict] = []
    if n_workers <= 1 or len(jobs) <= 1:
        for job in jobs:
            ep = run_expert_episode(
                job["instance"], job["m"], expert, weights,
                charging_portions=portions,
                use_horizon_check=use_horizon,
                use_depot_charging=use_depot,
                max_steps_factor=max_steps_factor,
                environment_config=env_cfg,
                reward_scale=float((config.get("rl") or {}).get("reward_scale", 10000.0)),
                collect_transitions=True,
            )
            episodes.append(_episode_to_demo_dict(ep))
            LOGGER.info(
                "Demo %s | served=%.3f feas=%s decisions=%d",
                ep.instance_name, ep.metrics["served_ratio"], ep.metrics["feasible"],
                ep.metrics["num_decisions"],
            )
    else:
        from concurrent.futures import ProcessPoolExecutor

        LOGGER.info(
            "Demo generation parallel | workers=%d instances=%d (exact masks unchanged)",
            n_workers, len(jobs),
        )
        # Preserve instance order for reproducibility of the .pt file.
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            ordered = list(pool.map(_parallel_demo_job, jobs))
        episodes = ordered
        for ep in episodes:
            LOGGER.info(
                "Demo %s | served=%.3f feas=%s decisions=%d",
                ep["instance_name"], ep["metrics"]["served_ratio"], ep["metrics"]["feasible"],
                ep["metrics"]["num_decisions"],
            )

    payload = {
        "meta": {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "num_episodes": len(episodes),
            "train_instances": list(meta.get("train_instances") or train_ds.names),
            "config_expert": asdict(ExpertConfig.from_config(config)),
            "charging_portions": portions,
            "num_workers": n_workers,
        },
        "episodes": episodes,
    }
    torch.save(payload, out_path)
    LOGGER.info("Wrote demonstrations: %s (%d episodes)", out_path, len(episodes))
    summary_path = out_path.with_suffix(".summary.json")
    summary = {
        "num_episodes": len(episodes),
        "mean_served_ratio": float(np.mean([e["metrics"]["served_ratio"] for e in episodes])),
        "feasibility_rate": float(np.mean([1.0 if e["metrics"]["feasible"] else 0.0 for e in episodes])),
        "mean_objective": float(np.mean([e["metrics"]["objective"] for e in episodes])),
        "path": str(out_path),
        "num_workers": n_workers,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate / evaluate RC21 expert demonstrations.")
    p.add_argument("--config", type=str, default="configs/train.yaml")
    p.add_argument("--evaluate-only", action="store_true", help="Evaluate expert without writing demos.")
    p.add_argument("--holdout", action="store_true", help="Evaluate on holdout split.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.evaluate_only:
        report = evaluate_expert(config, on_holdout=bool(args.holdout))
        out = Path((config.get("demonstrations") or {}).get(
            "eval_report", "outputs/demonstrations/expert_eval_report.json",
        ))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        LOGGER.info("Wrote expert eval report: %s", out)
        return
    generate_demonstrations(config)


if __name__ == "__main__":
    main()
