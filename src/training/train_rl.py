"""REINFORCE training loop for the Transformer multi-agent E-VRPTW policy."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.data.evrptw_dataset import split_dataset
from src.data.instance_loader import load_capacity_from_config
from src.env.cost import CostWeights, training_reward
from src.models.policy import TransformerEVRPTWPolicy
from src.training.baseline import GreedyRolloutBaseline, MovingAverageBaseline
from src.training.rollout import run_episode
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


def train(config: dict) -> dict:
    """Run REINFORCE training; return a summary dict of the run."""
    deterministic = bool(config.get("deterministic", False))
    set_seed(int(config.get("seed", 42)), deterministic=deterministic)
    device = resolve_device(config.get("device", "auto"))
    LOGGER.info(f"Training on device: {describe_device(device)}")

    ds_cfg = config["dataset"]
    train_ds, _eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=int(config.get("seed", 42)),
        train_filter=ds_cfg.get("train_size_filter"),
        eval_filter=ds_cfg.get("eval_size_filter"),
        load_capacity=load_capacity_from_config(config),
        ev_specs_path=ds_cfg.get("ev_specs_path"),
    )
    LOGGER.info(f"Training instances: {len(train_ds)}")

    charging_portions = config["problem"]["charging_portions"]
    weights = CostWeights.from_config(config["cost"])
    policy = build_policy(config, charging_portions).to(device)

    tcfg = config["training"]
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(tcfg["learning_rate"]))
    baseline_kind = str(tcfg.get("baseline", "moving_average"))
    reward_scale = float(tcfg.get("reward_scale", 1000.0))
    normalize_advantage = bool(tcfg.get("normalize_advantage", True))
    greedy_baseline = GreedyRolloutBaseline(
        weights, charging_portions, use_horizon, max_steps_factor, reward_scale=reward_scale,
    )
    ma_baseline = MovingAverageBaseline(beta=float(tcfg.get("baseline_beta", 0.9)))
    entropy_coef = float(tcfg.get("entropy_coef", 0.01))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    batch_size = int(tcfg["batch_size"])
    epochs = int(tcfg["epochs"])
    use_fleet_search = bool(tcfg.get("use_fleet_search", False))
    extra_margin = int(config["fleet_search"].get("extra_margin", 5))
    decode_cfg = config["decoding"]
    max_steps_factor = int(decode_cfg.get("max_steps_factor", 6))
    use_horizon = bool(config["problem"].get("use_horizon_check", True))

    out_cfg = config["outputs"]
    clear_training_outputs(out_cfg)
    Path(out_cfg["checkpoints"]).mkdir(parents=True, exist_ok=True)
    log_path = Path(out_cfg["logs"]) / "train_log.jsonl"
    jsonl = JsonlLogger(log_path)
    results_dir = Path(out_cfg["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    train_csv = CsvLogger(results_dir / "training_metrics.csv")

    # Archive this run (live outputs are cleared at the start of each new run).
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = Path(out_cfg["root"]) / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    run_logger = JsonlLogger(run_dir / "metrics.jsonl")
    run_train_csv = CsvLogger(run_dir / "metrics.csv")
    LOGGER.info(f"Run archive: {run_dir}")

    rng = random.Random(int(config.get("seed", 42)))
    indices = list(range(len(train_ds)))
    best_reward = -float("inf")
    checkpoint_every = int(tcfg.get("checkpoint_every", 10))

    for epoch in range(epochs):
        rng.shuffle(indices)
        batch_idx = indices[:batch_size] if batch_size <= len(indices) else [
            rng.choice(indices) for _ in range(batch_size)
        ]

        optimizer.zero_grad()
        raw_costs, rewards, feas_flags, used_evs_list = [], [], [], []
        dist_list, charge_list, price_list, wait_list, viol_list = [], [], [], [], []
        unserved_list, duration_list = [], []
        served_rate_list, customers_list = [], []
        breakdowns: list[dict] = []
        log_probs, entropies, advantages = [], [], []

        for idx in batch_idx:
            instance = train_ds[idx]
            m = instance.m_min if not use_fleet_search else _sample_m(instance, extra_margin, rng)
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
                b = 0.0  # filled after batch mean for moving-average baseline
            advantages.append(reward - b if baseline_kind == "greedy_rollout" else reward)
            log_probs.append(result.log_prob_sum)
            entropies.append(result.entropy_sum)
            raw_costs.append(raw_j)
            rewards.append(reward)
            feas_flags.append(float(result.feasible))
            used_evs_list.append(result.summary.used_evs)
            dist_list.append(result.summary.total_distance)
            charge_list.append(result.summary.total_charging_time)
            price_list.append(result.summary.total_charging_price)
            wait_list.append(result.summary.total_waiting_time)
            unserved_list.append(result.summary.unserved_customers)
            duration_list.append(result.summary.total_duration)
            n_cust = max(int(instance.num_customers), 1)
            customers_list.append(n_cust)
            served_rate_list.append(1.0 - result.summary.unserved_customers / n_cust)
            viol_list.append(
                result.summary.total_battery_violation
                + result.summary.total_capacity_violation
                + result.summary.total_lateness
                + result.summary.unserved_customers
            )
            breakdowns.append(result.breakdown)

        batch_mean_reward = float(np.mean(rewards))
        if baseline_kind != "greedy_rollout":
            b = ma_baseline.update(batch_mean_reward)
            advantages = [r - b for r in rewards]

        adv_raw = torch.tensor(advantages, dtype=torch.float32)
        adv = adv_raw.clone()
        if normalize_advantage and adv.numel() > 1 and adv.std() > 1e-8:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        policy_loss = torch.stack([-adv[i] * log_probs[i] for i in range(len(log_probs))]).mean()
        entropy_term = torch.stack(entropies).mean()
        loss = policy_loss - entropy_coef * entropy_term

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        improved = batch_mean_reward > best_reward
        if improved:
            best_reward = batch_mean_reward

        # Mean of every cost-breakdown component across the batch (paper-ready).
        breakdown_means = {
            f"mean_{k}": float(np.mean([bd.get(k, 0.0) for bd in breakdowns]))
            for k in breakdowns[0]
        }

        mean_dist = float(np.mean(dist_list))
        mean_evs = float(np.mean(used_evs_list))
        mean_served = float(np.mean(served_rate_list))
        mean_charge = float(np.mean(charge_list))
        mean_price = float(np.mean(price_list))
        mean_wait = float(np.mean(wait_list))
        mean_active = mean_dist + mean_charge + mean_wait
        mean_feas = float(np.mean(feas_flags))
        record = {
            "epoch": epoch,
            "reward": batch_mean_reward,
            "reward_scaled": batch_mean_reward,
            "reward_scale": reward_scale,
            "reward_std": float(np.std(rewards)),
            "reward_min": float(np.min(rewards)),
            "reward_max": float(np.max(rewards)),
            "best_reward": best_reward,
            "baseline": float(batch_mean_reward - float(adv_raw.mean())),
            "advantage": float(adv_raw.mean()),
            "cost": float(np.mean(raw_costs)),
            "cost_raw": float(np.mean(raw_costs)),
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "grad_norm": float(grad_norm),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "feasibility_rate": mean_feas,
            "feasibility_rate_pct": 100.0 * mean_feas,
            "served_rate": mean_served,
            "served_rate_pct": 100.0 * mean_served,
            "avg_used_evs": mean_evs,
            "avg_distance": mean_dist,
            "avg_duration": float(np.mean(duration_list)),
            "avg_charging_time": mean_charge,
            "avg_charging_price": mean_price,
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
        jsonl.log(record)
        run_logger.log(record)
        train_csv.log(record)
        run_train_csv.log(record)
        if epoch % int(tcfg.get("log_every", 1)) == 0:
            LOGGER.info(
                f"epoch {epoch:3d} | reward {batch_mean_reward:10.1f} | "
                f"feas {record['feasibility_rate']:.2f} | evs {record['avg_used_evs']:.1f} | "
                f"dist {record['avg_distance']:.1f}"
            )

        if improved:
            _save_checkpoint(policy, config, Path(out_cfg["checkpoints"]) / "best.pt")
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(policy, config, Path(out_cfg["checkpoints"]) / f"epoch_{epoch + 1}.pt")

    _save_checkpoint(policy, config, Path(out_cfg["checkpoints"]) / "last.pt")

    # ---- Persist final results and figures ----
    rows = jsonl.read()
    results_dir = Path(out_cfg["results"])
    figures_dir = Path(out_cfg["figures"]) / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_final_summary(rows, run_id, epochs, len(train_ds), log_path, run_dir)

    # Full per-epoch metrics + summary into outputs/results/ (and the run archive).
    if rows:
        write_csv(results_dir / "training_metrics.csv", rows)
        write_csv(run_dir / "metrics.csv", rows)
    (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Figures into outputs/figures/training/.
    figures = plot_training_curves(rows, figures_dir)
    overview = plot_training_overview(rows, figures_dir / "overview.png")
    obj_fig = plot_stacked_composition(
        rows, "mean_obj_", "Objective term composition", figures_dir / "objective_composition.png"
    )
    pen_fig = plot_stacked_composition(
        rows, "mean_pen_", "Penalty composition", figures_dir / "penalty_composition.png"
    )
    n_figs = len(figures) + sum(1 for f in (overview, obj_fig, pen_fig) if f is not None)

    LOGGER.info(f"Training complete. Best mean reward: {best_reward:.1f}")
    LOGGER.info(f"Results written to: {results_dir}")
    LOGGER.info(f"{n_figs} figures written to: {figures_dir}")
    LOGGER.info(f"Run archived at: {run_dir}")
    return summary


def _build_final_summary(rows, run_id, epochs, num_train, log_path, run_dir) -> dict:
    """Aggregate the training log into a compact, paper-ready summary."""
    summary = {
        "run_id": run_id,
        "epochs": epochs,
        "num_train_instances": num_train,
        "log_path": str(log_path),
        "run_dir": str(run_dir),
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


def _sample_m(instance, extra_margin: int, rng: random.Random) -> int:
    m_min = instance.m_min
    m_max = min(instance.num_customers, m_min + extra_margin)
    return rng.randint(m_min, max(m_min, m_max))


def _save_checkpoint(
    policy,
    config: dict,
    path: Path,
    training_meta: dict | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": policy.state_dict(),
        "model_config": config["model"],
        "charging_portions": policy.charging_portions,
    }
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
