"""Curriculum training orchestrator for the Multi-Agent Transformer E-VRPTW solver.

Pipeline:
    Phase 0  — environment & mask sanity checks (no learning)
    Phase 1  — 50 epochs on C
    Phase 2  — 70 epochs on C + R
    Phase 3  — 100 epochs on C + R + RC
    Phase 4  — 30 epochs fine-tuning on hard cases
Total: 250 training epochs after Phase 0.

Masked REINFORCE with a greedy-rollout baseline. Fleet size is sampled during
training and fully searched during validation. Checkpoints are kept for the best
overall / per-class / feasibility models plus the last epoch, using the score

    score = validation_cost + 10000 * (1 - feasibility_rate).
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data.evrptw_dataset import split_dataset
from src.data.instance_class import group_indices_by_class
from src.env.cost import CostWeights, training_reward
from src.training.baseline import GreedyRolloutBaseline, MovingAverageBaseline
from src.training.curriculum import (
    Phase,
    build_cyclic_schedule,
    cyclic_total_steps,
    extra_margin_for_class,
    phases_from_config,
    sample_balanced_batch,
    sample_hard_batch,
    sample_phase_m,
    select_hard_indices,
)
from src.training.phase0_checks import (
    format_report,
    phase0_passed,
    pick_check_instance,
    run_phase0_checks,
)
from src.training.rollout import run_episode
from src.training.train_rl import _save_checkpoint, build_policy
from src.training.validation import validate
from src.utils.logging_utils import CsvLogger, JsonlLogger, get_logger, write_csv
from src.utils.output_cleanup import clear_training_outputs
from src.utils.plotting import (
    plot_stacked_composition,
    plot_training_curves,
    plot_training_overview,
    plot_validation_curves,
)
from src.utils.seed import resolve_device, set_seed

LOGGER = get_logger()

CHECKPOINT_KEYS = ("overall", "C", "R", "RC", "feasibility")


class CheckpointManager:
    """Track and persist best-by-criterion checkpoints plus the last epoch."""

    def __init__(self, policy, config, ckpt_dir: Path):
        self.policy = policy
        self.config = config
        self.dir = ckpt_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.best_score = {k: float("inf") for k in ("overall", "C", "R", "RC")}
        self.best_feasibility = -1.0
        self.history: dict[str, dict] = {}

    def update(self, val: dict, global_epoch: int, phase_name: str) -> list[str]:
        saved = []
        for key in ("overall", "C", "R", "RC"):
            metrics = val.get(key)
            if not metrics:
                continue
            score = metrics["score"]
            if score < self.best_score[key]:
                self.best_score[key] = score
                path = self.dir / f"best_{key}.pt"
                _save_checkpoint(self.policy, self.config, path)
                if key == "overall":
                    _save_checkpoint(self.policy, self.config, self.dir / "best.pt")
                self.history[f"best_{key}"] = {
                    "global_epoch": global_epoch, "phase": phase_name, **metrics
                }
                saved.append(f"best_{key}")
        overall = val.get("overall") or {}
        feas = overall.get("feasibility_rate", -1.0)
        if feas > self.best_feasibility:
            self.best_feasibility = feas
            _save_checkpoint(self.policy, self.config, self.dir / "best_feasibility.pt")
            self.history["best_feasibility"] = {
                "global_epoch": global_epoch, "phase": phase_name, **overall
            }
            saved.append("best_feasibility")
        return saved

    def save_last(self):
        _save_checkpoint(self.policy, self.config, self.dir / "last.pt")


def train_curriculum(config: dict, run_phase0: bool = True, force: bool = False) -> dict:
    set_seed(int(config.get("seed", 42)))
    device = resolve_device(config.get("device", "auto"))
    LOGGER.info(f"Curriculum training on device: {device}")

    ds_cfg = config["dataset"]
    seed = int(config.get("seed", 42))
    train_ds, eval_ds = split_dataset(
        ds_cfg["path"], train_ratio=float(ds_cfg.get("train_ratio", 0.8)), seed=seed,
        train_filter=ds_cfg.get("train_size_filter"), eval_filter=ds_cfg.get("eval_size_filter"),
    )
    train_groups = group_indices_by_class(train_ds.names)
    eval_groups = group_indices_by_class(eval_ds.names)
    LOGGER.info(
        "Train by class: " + ", ".join(f"{c}={len(train_groups[c])}" for c in train_groups)
        + " | Eval by class: " + ", ".join(f"{c}={len(eval_groups[c])}" for c in eval_groups)
    )

    charging_portions = config["problem"]["charging_portions"]
    weights = CostWeights.from_config(config["cost"])
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    max_steps_factor = int(config["decoding"].get("max_steps_factor", 6))
    fcfg = config["fleet_search"]
    val_extra_margin = int(fcfg.get("extra_margin", 5))
    selection_mode = fcfg.get("selection_mode", "hierarchical")

    out_cfg = config["outputs"]
    clear_training_outputs(out_cfg)

    run_id = time.strftime("run_curriculum_%Y%m%d_%H%M%S")
    run_dir = Path(out_cfg["root"]) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    # ---------------- Phase 0 ----------------
    phase0_results = None
    if run_phase0:
        LOGGER.info("Running Phase 0 environment & mask checks...")
        check_instance = pick_check_instance(ds_cfg["path"])
        phase0_results = run_phase0_checks(check_instance)
        report = format_report(phase0_results)
        LOGGER.info("\n" + report)
        (run_dir / "phase0_report.txt").write_text(report, encoding="utf-8")
        Path(out_cfg["results"]).mkdir(parents=True, exist_ok=True)
        (Path(out_cfg["results"]) / "phase0_report.txt").write_text(report, encoding="utf-8")
        if not phase0_passed(phase0_results) and not force:
            raise RuntimeError(
                "Phase 0 checks failed; aborting before training. "
                "Pass force=True to override (not recommended)."
            )

    # ---------------- Training setup ----------------
    tcfg = config["training"]
    policy = build_policy(config, charging_portions).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(tcfg["learning_rate"]))
    entropy_coef = float(tcfg.get("entropy_coef", 0.01))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    batch_size = int(tcfg["batch_size"])
    validate_every = int(tcfg.get("validate_every", 10))
    val_decoding = config["evaluation"].get("decoding", "greedy")
    val_n_samples = int(config["evaluation"].get("multi_sample_n", 16))
    val_max_per_class = tcfg.get("val_max_per_class")

    baseline_kind = str(tcfg.get("baseline", "greedy_rollout"))
    reward_scale = float(tcfg.get("reward_scale", 1000.0))
    normalize_advantage = bool(tcfg.get("normalize_advantage", True))
    greedy_baseline = GreedyRolloutBaseline(
        weights, charging_portions, use_horizon, max_steps_factor, reward_scale=reward_scale,
    )
    ma_baseline = MovingAverageBaseline(beta=float(tcfg.get("baseline_beta", 0.9)))

    log_path = Path(out_cfg["logs"]) / "train_log.jsonl"
    train_logger = JsonlLogger(log_path)
    run_train_logger = JsonlLogger(run_dir / "metrics.jsonl")
    val_logger = JsonlLogger(Path(out_cfg["logs"]) / "val_log.jsonl")
    run_val_logger = JsonlLogger(run_dir / "val_metrics.jsonl")
    results_dir = Path(out_cfg["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    train_csv = CsvLogger(results_dir / "training_metrics.csv")
    val_csv = CsvLogger(results_dir / "validation_metrics.csv")
    run_train_csv = CsvLogger(run_dir / "metrics.csv")
    run_val_csv = CsvLogger(run_dir / "validation_metrics.csv")

    ckpt_mgr = CheckpointManager(policy, config, Path(out_cfg["checkpoints"]))
    rng = random.Random(seed)

    ccfg = config.get("curriculum") or {}
    cyclic_mode = str(ccfg.get("mode", "phases")).lower() == "cyclic"
    phases = phases_from_config(ccfg if not cyclic_mode else None)
    schedule_meta: dict = {}

    if cyclic_mode:
        class_order = [c.upper() for c in ccfg.get("class_order", ["C", "R", "RC"])]
        cycles = int(ccfg.get("cycles", 5))
        cyclic_steps = build_cyclic_schedule(
            train_groups,
            class_order=class_order,
            cycles=cycles,
            shuffle_within_class=bool(ccfg.get("shuffle_within_class", False)),
            rng=rng,
        )
        total_steps = len(cyclic_steps)
        schedule_meta = {
            "mode": "cyclic",
            "class_order": class_order,
            "cycles": cycles,
            "steps_per_cycle": cyclic_steps[0]["steps_per_cycle"] if cyclic_steps else 0,
            "total_epochs": total_steps,
        }
        LOGGER.info(
            f"Cyclic schedule: {' → '.join(class_order)} × {cycles} cycles "
            f"= {total_steps} iterations ({schedule_meta['steps_per_cycle']} instances/cycle)"
        )
    else:
        LOGGER.info("Schedule: " + ", ".join(f"P{p.index}:{p.name}({p.epochs})" for p in phases))

    train_records: list[dict] = []
    val_records: list[dict] = []
    global_epoch = 0
    hard_pool: list[int] = []

    _CLASS_PHASE_ID = {"C": 1, "R": 2, "RC": 3}

    def _run_validation(step_ctx, ge: int) -> dict:
        classes = step_ctx.classes if hasattr(step_ctx, "classes") and step_ctx.classes else ["C", "R", "RC"]
        val = validate(
            eval_ds, eval_groups, policy, weights,
            classes=classes, mode=val_decoding, device=device,
            n_samples=val_n_samples, charging_portions=charging_portions,
            use_horizon=use_horizon, max_steps_factor=max_steps_factor,
            extra_margin=val_extra_margin, selection_mode=selection_mode,
            max_instances=val_max_per_class,
        )
        flat = {
            "global_epoch": ge,
            "phase": getattr(step_ctx, "index", 0),
            "phase_name": getattr(step_ctx, "name", "cyclic"),
        }
        for cls in ("C", "R", "RC", "overall"):
            m = val.get(cls)
            if m:
                for k, v in m.items():
                    if isinstance(v, (int, float)):
                        flat[f"{cls}_{k}"] = v
        val_logger.log(flat)
        run_val_logger.log(flat)
        val_csv.log(flat)
        run_val_csv.log(flat)
        val_records.append(flat)
        saved = ckpt_mgr.update(val, ge, getattr(step_ctx, "name", "cyclic"))
        ov = val.get("overall") or {}
        LOGGER.info(
            f"  [val] epoch {ge} | overall score {ov.get('score', float('nan')):.1f} "
            f"| feas {ov.get('feasibility_rate', 0):.2f}"
            + (f" | saved {saved}" if saved else "")
        )
        return val

    def _log_and_store(record: dict) -> None:
        train_logger.log(record)
        run_train_logger.log(record)
        train_csv.log(record)
        run_train_csv.log(record)
        train_records.append(record)

    # ---------------- Cyclic training (C → R → RC, repeat) ----------------
    if cyclic_mode:
        cyclic_phase = Phase(0, "cyclic", ["C", "R", "RC"], 0)
        for step in cyclic_steps:
            cls = step["instance_class"]
            idx = step["instance_idx"]
            margin = extra_margin_for_class(config, cls)
            step_phase = Phase(
                _CLASS_PHASE_ID.get(cls, 0),
                f"cycle{step['cycle']}_{cls}",
                [cls],
                0,
                extra_margin=margin,
            )
            record = _train_step(
                policy, optimizer, train_ds, [idx], weights, device,
                charging_portions, use_horizon, max_steps_factor, step_phase,
                rng, baseline_kind, greedy_baseline, ma_baseline,
                entropy_coef, grad_clip, reward_scale, normalize_advantage,
            )
            record["epoch"] = global_epoch
            record["phase"] = step_phase.index
            record["phase_name"] = step_phase.name
            record["cycle"] = step["cycle"]
            record["instance_class"] = cls
            record["instance_idx"] = idx
            record["instance_name"] = train_ds.names[idx]
            record["step_in_cycle"] = step["step_in_cycle"]
            _log_and_store(record)

            if global_epoch % int(tcfg.get("log_every", 1)) == 0:
                LOGGER.info(
                    f"cycle {step['cycle']} {cls} e{global_epoch:4d} | "
                    f"{train_ds.names[idx]} | reward {record['reward']:8.2f} "
                    f"| feas {record['feasibility_rate']:.2f}"
                )

            if validate_every > 0 and (global_epoch + 1) % validate_every == 0:
                _run_validation(cyclic_phase, global_epoch)
            global_epoch += 1

        _run_validation(cyclic_phase, global_epoch - 1)
        ckpt_mgr.save_last()

    # ---------------- Phase-based training (legacy) ----------------
    else:
        for phase in phases:
            LOGGER.info(f"=== Phase {phase.index}: {phase.name} ({phase.epochs} epochs) ===")
            if phase.hard_finetune:
                LOGGER.info("Selecting hard cases from the training set...")
                all_train = [i for c in phase.classes for i in train_groups.get(c, [])]
                hard_pool = select_hard_indices(
                    train_ds, all_train, policy, weights, device=device,
                    charging_portions=charging_portions, use_horizon=use_horizon,
                    max_steps_factor=max_steps_factor, extra_margin=phase.extra_margin,
                    hard_fraction=float(tcfg.get("hard_fraction", 0.5)), rng=rng,
                )
                LOGGER.info(f"Hard pool size: {len(hard_pool)} / {len(all_train)}")

            for _ in range(phase.epochs):
                if phase.hard_finetune:
                    normal_pool = [i for c in phase.classes for i in train_groups.get(c, [])]
                    batch_idx = sample_hard_batch(rng, hard_pool, normal_pool, batch_size,
                                                  hard_share=float(tcfg.get("hard_share", 0.5)))
                else:
                    batch_idx = sample_balanced_batch(
                        rng, train_groups, phase.classes, phase.proportions, batch_size
                    )

                record = _train_step(
                    policy, optimizer, train_ds, batch_idx, weights, device,
                    charging_portions, use_horizon, max_steps_factor, phase,
                    rng, baseline_kind, greedy_baseline, ma_baseline,
                    entropy_coef, grad_clip, reward_scale, normalize_advantage,
                )
                record["epoch"] = global_epoch
                record["phase"] = phase.index
                record["phase_name"] = phase.name
                _log_and_store(record)

                if global_epoch % int(tcfg.get("log_every", 1)) == 0:
                    LOGGER.info(
                        f"P{phase.index} e{global_epoch:3d} | reward {record['reward']:9.1f} "
                        f"| feas {record['feasibility_rate']:.2f} | evs {record['avg_used_evs']:.1f} "
                        f"| dist {record['avg_distance']:.1f}"
                    )

                if validate_every > 0 and (global_epoch + 1) % validate_every == 0:
                    _run_validation(phase, global_epoch)
                global_epoch += 1

            _run_validation(phase, global_epoch - 1)
            ckpt_mgr.save_last()

    # ---------------- Persist results & figures ----------------
    summary = _finalize(
        config, run_id, run_dir, train_records, val_records, ckpt_mgr, phases,
        phase0_results, schedule_meta=schedule_meta,
    )

    # ---------------- Final evaluation (full fleet search) ----------------
    if bool(tcfg.get("run_final_eval", True)):
        try:
            from src.evaluation.evaluate import evaluate

            best_overall = str(Path(out_cfg["checkpoints"]) / "best.pt")
            LOGGER.info("Running final evaluation with full fleet-size search...")
            eval_summary = evaluate(config, checkpoint=best_overall)
            summary["evaluation"] = eval_summary
        except Exception as exc:  # evaluation must not lose the trained model
            LOGGER.warning(f"Final evaluation failed: {exc!r}")

    (run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (Path(out_cfg["results"]) / "curriculum_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    LOGGER.info(f"Curriculum training complete. Run archived at: {run_dir}")
    return summary


def _train_step(
    policy, optimizer, train_ds, batch_idx, weights, device, charging_portions,
    use_horizon, max_steps_factor, phase, rng, baseline_kind, greedy_baseline,
    ma_baseline, entropy_coef, grad_clip, reward_scale, normalize_advantage,
) -> dict:
    """One REINFORCE update on a sampled batch; return the metrics record."""
    policy.train()
    optimizer.zero_grad()

    raw_costs, rewards, feas_flags = [], [], []
    used_evs_list, dist_list, charge_list, wait_list = [], [], [], []
    viol_list, unserved_list, duration_list = [], [], []
    served_rate_list, customers_list = [], []
    breakdowns: list[dict] = []
    log_probs, entropies, advantages = [], [], []

    for idx in batch_idx:
        instance = train_ds[idx]
        m = sample_phase_m(instance, phase.extra_margin, rng)
        result = run_episode(
            instance, policy, m, weights, mode="sampling", device=device,
            charging_portions=charging_portions, use_horizon_check=use_horizon,
            max_steps_factor=max_steps_factor,
        )
        raw_j = float(result.objective)
        reward = training_reward(raw_j, reward_scale)

        if baseline_kind == "greedy_rollout":
            b = greedy_baseline.reward(instance, policy, m, device)
        else:
            b = ma_baseline.update(reward)
        advantages.append(reward - b)
        log_probs.append(result.log_prob_sum)
        entropies.append(result.entropy_sum)

        raw_costs.append(raw_j)
        rewards.append(reward)
        feas_flags.append(float(result.feasible))
        used_evs_list.append(result.summary.used_evs)
        dist_list.append(result.summary.total_distance)
        charge_list.append(result.summary.total_charging_time)
        wait_list.append(result.summary.total_waiting_time)
        unserved_list.append(result.summary.unserved_customers)
        duration_list.append(result.summary.total_duration)
        n_cust = max(int(instance.num_customers), 1)
        customers_list.append(n_cust)
        served_rate_list.append(1.0 - result.summary.unserved_customers / n_cust)
        viol_list.append(
            result.summary.total_battery_violation + result.summary.total_capacity_violation
            + result.summary.total_lateness + result.summary.unserved_customers
        )
        breakdowns.append(result.breakdown)

    adv_raw = torch.tensor(advantages, dtype=torch.float32)
    adv = adv_raw.clone()
    if normalize_advantage and adv.numel() > 1 and adv.std() > 1e-8:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    loss_terms = [-adv[i] * log_probs[i] for i in range(len(log_probs))]
    policy_loss = torch.stack(loss_terms).mean()
    entropy_term = torch.stack(entropies).mean()
    loss = policy_loss - entropy_coef * entropy_term
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
    optimizer.step()

    breakdown_means = {
        f"mean_{k}": float(np.mean([bd.get(k, 0.0) for bd in breakdowns])) for k in breakdowns[0]
    }
    mean_dist = float(np.mean(dist_list))
    mean_evs = float(np.mean(used_evs_list))
    mean_charge = float(np.mean(charge_list))
    mean_wait = float(np.mean(wait_list))
    mean_active = mean_dist + mean_charge + mean_wait
    mean_feas = float(np.mean(feas_flags))
    batch_mean_reward = float(np.mean(rewards))
    return {
        "cost": float(np.mean(raw_costs)),
        "cost_raw": float(np.mean(raw_costs)),
        "reward": batch_mean_reward,
        "reward_scaled": batch_mean_reward,
        "reward_scale": reward_scale,
        "reward_std": float(np.std(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "baseline": float(batch_mean_reward - float(adv_raw.mean())),
        "advantage": float(adv_raw.mean()),
        "loss": float(loss.item()),
        "policy_loss": float(policy_loss.item()),
        "grad_norm": float(grad_norm),
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
        "feasibility_rate": mean_feas,
        "feasibility_rate_pct": 100.0 * mean_feas,
        "served_rate": float(np.mean(served_rate_list)),
        "served_rate_pct": 100.0 * float(np.mean(served_rate_list)),
        "avg_used_evs": mean_evs,
        "avg_distance": mean_dist,
        "avg_duration": float(np.mean(duration_list)),
        "avg_charging_time": mean_charge,
        "avg_waiting_time": mean_wait,
        "avg_unserved": float(np.mean(unserved_list)),
        "avg_violation": float(np.mean(viol_list)),
        "distance_per_ev": mean_dist / mean_evs if mean_evs > 0 else 0.0,
        "distance_per_customer": mean_dist / float(np.mean(customers_list)),
        "charging_time_share": mean_charge / mean_active if mean_active > 0 else 0.0,
        "waiting_time_share": mean_wait / mean_active if mean_active > 0 else 0.0,
        "entropy": float(entropy_term.item()),
        **breakdown_means,
    }


def _finalize(
    config, run_id, run_dir, train_records, val_records, ckpt_mgr, phases, phase0_results,
    schedule_meta: dict | None = None,
) -> dict:
    out_cfg = config["outputs"]
    results_dir = Path(out_cfg["results"])
    figures_dir = Path(out_cfg["figures"]) / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if train_records:
        write_csv(results_dir / "training_metrics.csv", train_records)
        write_csv(run_dir / "metrics.csv", train_records)
    if val_records:
        write_csv(results_dir / "validation_metrics.csv", val_records)
        write_csv(run_dir / "validation_metrics.csv", val_records)

    n_figs = 0
    if train_records:
        n_figs += len(plot_training_curves(train_records, figures_dir))
        if plot_training_overview(train_records, figures_dir / "overview.png"):
            n_figs += 1
        if plot_stacked_composition(train_records, "mean_obj_", "Objective term composition",
                                    figures_dir / "objective_composition.png"):
            n_figs += 1
        if plot_stacked_composition(train_records, "mean_pen_", "Penalty composition",
                                    figures_dir / "penalty_composition.png"):
            n_figs += 1
    if val_records:
        n_figs += len(plot_validation_curves(val_records, figures_dir / "validation"))
    LOGGER.info(f"{n_figs} figures written to {figures_dir}")

    summary = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "schedule": schedule_meta or {
            "mode": "phases",
            "phases": [
                {"phase": p.index, "name": p.name, "classes": p.classes, "epochs": p.epochs}
                for p in phases
            ],
            "total_epochs": sum(p.epochs for p in phases),
        },
        "total_epochs": (schedule_meta or {}).get("total_epochs") or sum(p.epochs for p in phases),
        "best_checkpoints": ckpt_mgr.history,
        "phase0_passed": phase0_passed(phase0_results) if phase0_results else None,
    }
    if val_records:
        last_val = val_records[-1]
        summary["final_validation"] = last_val
    return summary
