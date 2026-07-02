"""Actor-critic training loop for the Transformer multi-agent E-VRPTW policy."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from src.data.evrptw_dataset import split_dataset
from src.data.instance_loader import load_capacity_from_config
from src.env.cost import CostWeights
from src.models.policy import TransformerEVRPTWPolicy
from src.training.ac_train import actor_critic_train_step, build_critics, critic_parameters
from src.training.actor_critic import RLConfig
from src.training.curriculum import Phase
from src.training.flat_schedule import build_sequential_schedule
from src.utils.logging_utils import CsvLogger, JsonlLogger, get_logger, write_csv
from src.utils.output_cleanup import clear_training_outputs
from src.utils.plotting import (
    plot_stacked_composition,
    plot_training_curves,
    plot_training_overview,
)
from src.utils.seed import describe_device, resolve_device, set_seed

LOGGER = get_logger()


def build_policy(config: dict, charging_portions: list[float]) -> TransformerEVRPTWPolicy:
    return TransformerEVRPTWPolicy(config["model"], charging_portions=charging_portions)


def _resolve_epochs_and_schedule(
    tcfg: dict,
    num_train: int,
    rng: random.Random,
) -> tuple[int, list[int] | None]:
    """Return (epochs, optional per-epoch instance index schedule)."""
    passes = tcfg.get("passes")
    batch_size = int(tcfg.get("batch_size", 1))
    shuffle_within_pass = bool(tcfg.get("shuffle_within_pass", False))

    if passes is not None:
        if batch_size != 1:
            raise ValueError("training.passes requires batch_size: 1 (one instance per step)")
        schedule = build_sequential_schedule(
            num_train,
            int(passes),
            shuffle_within_pass=shuffle_within_pass,
            rng=rng,
        )
        return len(schedule), schedule

    epochs = int(tcfg["epochs"])
    return epochs, None


def train(config: dict) -> dict:
    """Run flat actor-critic training (no curriculum schedule)."""
    deterministic = bool(config.get("deterministic", False))
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=deterministic)
    device = resolve_device(config.get("device", "auto"))
    LOGGER.info(f"Training on device: {describe_device(device)}")

    tcfg = config["training"]
    ds_cfg = config["dataset"]
    train_ds, eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=seed,
        train_filter=ds_cfg.get("train_size_filter"),
        eval_filter=ds_cfg.get("eval_size_filter"),
        train_instances=ds_cfg.get("train_instances"),
        eval_instances=ds_cfg.get("eval_instances"),
        load_capacity=load_capacity_from_config(config),
    )
    LOGGER.info(f"Training instances: {len(train_ds)} | held out: {len(eval_ds)}")
    if eval_ds.names:
        LOGGER.info(f"Held-out instances (for later testing): {', '.join(eval_ds.names)}")

    charging_portions = config["problem"]["charging_portions"]
    weights = CostWeights.from_config(config["cost"])
    policy = build_policy(config, charging_portions).to(device)
    rl_cfg = RLConfig.from_config(config)
    use_amp = bool(tcfg.get("use_amp", False)) and device.type == "cuda"
    grad_accum = int(tcfg.get("gradient_accumulation_steps", 1))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=use_amp) if use_amp else None
    value_critic, q_critic = build_critics(
        policy, device, build_q=rl_cfg.use_counterfactual_critic,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=float(tcfg["learning_rate"]))
    critic_lr = float((config.get("rl") or {}).get("critic_learning_rate", tcfg["learning_rate"]))
    critic_optimizer = torch.optim.Adam(critic_parameters(value_critic, q_critic), lr=critic_lr)
    entropy_coef = float(tcfg.get("entropy_coef", 0.01))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    batch_size = int(tcfg["batch_size"])
    extra_margin = int(config["fleet_search"].get("extra_margin", 5))
    decode_cfg = config["decoding"]
    max_steps_factor = int(decode_cfg.get("max_steps_factor", 6))
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    phase = Phase(0, "flat", [], extra_margin)

    out_cfg = config["outputs"]
    clear_training_outputs(out_cfg)
    Path(out_cfg["checkpoints"]).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_cfg["logs"]) / "train_log.jsonl"
    jsonl = JsonlLogger(log_path)
    results_dir = Path(out_cfg["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    train_csv = CsvLogger(results_dir / "training_metrics.csv")

    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(out_cfg["root"]) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    holdout_meta = {
        "train_instances": list(train_ds.names),
        "eval_instances": list(eval_ds.names),
    }
    (run_dir / "holdout_instances.json").write_text(
        json.dumps(holdout_meta, indent=2), encoding="utf-8",
    )
    run_logger = JsonlLogger(run_dir / "metrics.jsonl")
    run_train_csv = CsvLogger(run_dir / "metrics.csv")
    LOGGER.info(f"Run archive: {run_dir}")

    rng = random.Random(seed)
    indices = list(range(len(train_ds)))
    epoch_schedule, sequential_schedule = _resolve_epochs_and_schedule(tcfg, len(train_ds), rng)
    if sequential_schedule is not None:
        passes = int(tcfg["passes"])
        LOGGER.info(
            f"Sequential training: {len(train_ds)} instances × {passes} passes "
            f"= {epoch_schedule} epochs"
        )

    best_reward = -float("inf")
    checkpoint_every = int(tcfg.get("checkpoint_every", 10))
    min_decisions_warn = tcfg.get("min_decisions_warn")

    for epoch in range(epoch_schedule):
        if sequential_schedule is not None:
            batch_idx = [sequential_schedule[epoch]]
            pass_num = epoch // len(train_ds) + 1
        else:
            pass_num = None
            rng.shuffle(indices)
            batch_idx = indices[:batch_size] if batch_size <= len(indices) else [
                rng.choice(indices) for _ in range(batch_size)
            ]

        record = actor_critic_train_step(
            policy, value_critic, q_critic, optimizer, critic_optimizer,
            train_ds, batch_idx, weights, device, charging_portions,
            use_horizon, max_steps_factor, phase, rng, rl_cfg,
            entropy_coef, grad_clip,
            min_decisions_warn=min_decisions_warn,
            global_epoch=epoch,
            use_amp=use_amp,
            gradient_accumulation_steps=grad_accum,
            scaler=amp_scaler,
        )
        record["epoch"] = epoch
        if len(batch_idx) == 1:
            record["instance_name"] = train_ds.names[batch_idx[0]]
            record["instance_idx"] = batch_idx[0]
        if pass_num is not None:
            record["pass"] = pass_num

        batch_mean_reward = float(record.get("reward", 0.0))
        improved = batch_mean_reward > best_reward
        if improved:
            best_reward = batch_mean_reward

        jsonl.log(record)
        run_logger.log(record)
        train_csv.log(record)
        run_train_csv.log(record)
        if epoch % int(tcfg.get("log_every", 1)) == 0:
            inst = record.get("instance_name", "?")
            pass_tag = f" pass {pass_num}" if pass_num is not None else ""
            LOGGER.info(
                f"epoch {epoch:3d}{pass_tag} | {inst} | reward {batch_mean_reward:10.1f} | "
                f"feas {record['feasibility_rate']:.2f} | evs {record['avg_used_evs']:.1f} | "
                f"dist {record['avg_distance']:.1f}"
            )

        if improved:
            _save_checkpoint(
                policy, config, Path(out_cfg["checkpoints"]) / "best.pt",
                value_critic=value_critic, q_critic=q_critic,
            )
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(
                policy, config, Path(out_cfg["checkpoints"]) / f"epoch_{epoch + 1}.pt",
                value_critic=value_critic, q_critic=q_critic,
            )

    _save_checkpoint(
        policy, config, Path(out_cfg["checkpoints"]) / "last.pt",
        value_critic=value_critic, q_critic=q_critic,
    )

    rows = jsonl.read()
    results_dir = Path(out_cfg["results"])
    figures_dir = Path(out_cfg["figures"]) / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_final_summary(rows, run_id, epoch_schedule, len(train_ds), log_path, run_dir)
    summary["holdout_instances"] = holdout_meta

    if rows:
        write_csv(results_dir / "training_metrics.csv", rows)
        write_csv(run_dir / "metrics.csv", rows)

    from src.training.training_dashboard import generate_training_dashboard_from_config

    dash_report = generate_training_dashboard_from_config(
        config, rows=rows, holdout_meta=holdout_meta,
    )
    summary["training_dashboard"] = {
        "html": dash_report.html_path,
        "figures_index": dash_report.index_md_path,
        "figures_generated": sum(1 for f in dash_report.figures if f.generated),
        "figures_total": len(dash_report.figures),
    }

    (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    figures = plot_training_curves(rows, figures_dir)
    overview = plot_training_overview(rows, figures_dir / "overview.png")
    obj_fig = plot_stacked_composition(
        rows, "mean_obj_", "Objective term composition", figures_dir / "objective_composition.png"
    )
    pen_fig = plot_stacked_composition(
        rows, "mean_pen_", "Penalty composition", figures_dir / "penalty_composition.png"
    )
    n_legacy = len(figures) + sum(1 for f in (overview, obj_fig, pen_fig) if f is not None)
    n_dash = summary.get("training_dashboard", {}).get("figures_generated", 0)

    LOGGER.info(f"Training complete. Best mean reward: {best_reward:.1f}")
    LOGGER.info(f"Results written to: {results_dir}")
    LOGGER.info(f"{n_legacy} legacy + {n_dash} dashboard figures in: {figures_dir}")
    if summary.get("training_dashboard", {}).get("html"):
        LOGGER.info(f"Training dashboard: {summary['training_dashboard']['html']}")
    LOGGER.info(f"Run archived at: {run_dir}")
    return summary


def _build_final_summary(rows, run_id, epochs, num_train, log_path, run_dir) -> dict:
    summary = {
        "run_id": run_id,
        "epochs": epochs,
        "num_train_instances": num_train,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
        "algorithm": "actor_critic",
    }
    if not rows:
        return summary
    last = rows[-1]
    best_reward = max(r.get("reward", -float("inf")) for r in rows)
    best_cost = min(r.get("cost", float("inf")) for r in rows)
    best_feas = max(r.get("feasibility_rate", 0.0) for r in rows)
    summary.update(
        {
            "best_reward": best_reward,
            "best_cost": best_cost,
            "best_feasibility_rate": best_feas,
            "final_reward": last.get("reward"),
            "final_cost": last.get("cost"),
            "final_feasibility_rate": last.get("feasibility_rate"),
            "final_served_rate": last.get("served_rate"),
            "final_avg_used_evs": last.get("avg_used_evs"),
            "final_avg_distance": last.get("avg_distance"),
            "final_avg_violation": last.get("avg_violation"),
        }
    )
    return summary


def _save_checkpoint(
    policy,
    config: dict,
    path: Path,
    training_meta: dict | None = None,
    value_critic=None,
    q_critic=None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": policy.state_dict(),
        "model_config": config["model"],
        "charging_portions": policy.charging_portions,
    }
    if value_critic is not None:
        payload["value_critic_state"] = value_critic.state_dict()
    if q_critic is not None:
        payload["q_critic_state"] = q_critic.state_dict()
    if training_meta is not None:
        payload["training_meta"] = training_meta
    torch.save(payload, path)


def load_checkpoint_training_meta(checkpoint_path: str | Path) -> dict | None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("training_meta")


def load_policy(checkpoint_path: str | Path, device: torch.device | str = "cpu") -> TransformerEVRPTWPolicy:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    policy = TransformerEVRPTWPolicy(ckpt["model_config"], charging_portions=ckpt["charging_portions"])
    policy.load_state_dict(ckpt["model_state"])
    policy.to(device)
    policy.eval()
    return policy
