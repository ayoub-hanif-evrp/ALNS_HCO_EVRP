"""Actor-critic training loop for the Transformer multi-agent E-VRPTW policy."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from src.data.evrptw_dataset import split_dataset
from src.data.instance_loader import load_capacity_from_config
from src.data.rc21_split import log_rc21_split_startup, resolve_rc21_split
from src.env.constraints import ConstraintConfig, log_constraint_startup
from src.data.immutability import log_dataset_immutability
from src.env.cost import CostWeights
from src.models.policy import TransformerEVRPTWPolicy
from src.models.feature_specs import ARCHITECTURE_VERSION
from src.training.ac_train import actor_critic_train_step, build_critics, critic_parameters
from src.training.actor_critic import RLConfig
from src.training.config_validation import apply_safe_training_defaults, validate_training_config
from src.training.cycle_summary import aggregate_cycle, dedupe_metric_rows
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


def _resolve_cycles(tcfg: dict) -> int:
    """Number of full passes over training instances (cycles / passes)."""
    if tcfg.get("cycles") is not None:
        return int(tcfg["cycles"])
    if tcfg.get("passes") is not None:
        return int(tcfg["passes"])
    return 0


def _resolve_epochs_and_schedule(
    tcfg: dict,
    num_train: int,
    rng: random.Random,
) -> tuple[int, list[int] | None]:
    """Return (epochs, optional per-epoch instance index schedule)."""
    cycles = _resolve_cycles(tcfg)
    batch_size = int(tcfg.get("batch_size", 1))
    shuffle_within_pass = bool(tcfg.get("shuffle_within_pass", False))

    if cycles > 0:
        if batch_size != 1:
            raise ValueError("training.cycles requires batch_size: 1 (one instance per step)")
        schedule = build_sequential_schedule(
            num_train,
            cycles,
            shuffle_within_pass=shuffle_within_pass,
            rng=rng,
        )
        return len(schedule), schedule

    epochs = int(tcfg["epochs"])
    return epochs, None


def _resolve_learning_rates(config: dict) -> tuple[float, float]:
    tcfg = config.get("training") or {}
    ocfg = config.get("optimizer") or {}
    rcfg = config.get("rl") or {}
    actor_lr = float(ocfg.get("actor_lr", tcfg.get("learning_rate", 1e-4)))
    critic_lr = float(ocfg.get("critic_lr", rcfg.get("critic_learning_rate", actor_lr * 0.5)))
    return actor_lr, critic_lr


def _apply_overfit_mode(config: dict, train_ds) -> tuple[object, int]:
    """Train on a single RC _21 instance for many updates (sanity test)."""
    tcfg = config["training"]
    name = str(
        tcfg.get("overfit_instance_name")
        or tcfg.get("debug_overfit_instance")
        or train_ds.names[0]
    )
    if name not in train_ds.names:
        raise ValueError(f"debug_overfit_instance '{name}' not in training set")
    idx = train_ds.names.index(name)
    updates = int(tcfg.get("debug_overfit_updates", 300))
    LOGGER.info("DEBUG overfit mode: instance=%s, updates=%d", name, updates)
    return idx, updates


def train(config: dict) -> dict:
    """Run flat actor-critic training (no curriculum schedule)."""
    config = apply_safe_training_defaults(dict(config))
    validate_training_config(config)
    constraint_cfg = ConstraintConfig.from_config(config)
    log_constraint_startup(constraint_cfg)

    deterministic = bool(config.get("deterministic", False))
    seed = int(config.get("seed", 42))
    set_seed(seed, deterministic=deterministic)
    device = resolve_device(config.get("device", "auto"))
    LOGGER.info(f"Training on device: {describe_device(device)}")
    LOGGER.info("Mode: Schneider RC _21 — exact-mask semi-MDP actor-critic")

    tcfg = config["training"]
    ds_cfg = config["dataset"]
    dbg = config.get("debug") or {}

    train_names, eval_names, split_meta = resolve_rc21_split(config)
    ds_cfg = dict(ds_cfg)
    ds_cfg["train_instances"] = train_names
    ds_cfg["eval_instances"] = eval_names

    train_ds, eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=seed,
        train_filter=ds_cfg.get("train_size_filter"),
        eval_filter=ds_cfg.get("eval_size_filter"),
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_capacity_from_config(config),
    )
    cycles = _resolve_cycles(tcfg)
    log_rc21_split_startup(split_meta, cycles if cycles > 0 else int(tcfg.get("debug_overfit_updates", 300)))
    log_dataset_immutability(config, train_names)

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

    actor_lr, critic_lr = _resolve_learning_rates(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic_parameters(value_critic, q_critic), lr=critic_lr)
    entropy_coef = float(tcfg.get("entropy_coef", 0.01))
    actor_grad_clip = float(rl_cfg.actor_grad_clip)
    critic_grad_clip = float(rl_cfg.critic_grad_clip)
    batch_size = int(tcfg["batch_size"])
    extra_margin = int(config["fleet_search"].get("extra_margin", 5))
    decode_cfg = config["decoding"]
    max_steps_factor = int(decode_cfg.get("max_steps_factor", 6))
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    use_depot_charging = bool(config["problem"].get("use_depot_charging", True))

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
        **split_meta,
    }
    (run_dir / "holdout_instances.json").write_text(
        json.dumps(holdout_meta, indent=2), encoding="utf-8",
    )
    run_logger = JsonlLogger(run_dir / "metrics.jsonl")
    run_train_csv = CsvLogger(run_dir / "metrics.csv")
    LOGGER.info(f"Run archive: {run_dir}")

    rng = random.Random(seed)
    indices = list(range(len(train_ds)))

    overfit_idx: int | None = None
    if bool(tcfg.get("debug_overfit_one_instance", False)):
        overfit_idx, epoch_schedule = _apply_overfit_mode(config, train_ds)
        sequential_schedule = [overfit_idx] * epoch_schedule
    else:
        epoch_schedule, sequential_schedule = _resolve_epochs_and_schedule(tcfg, len(train_ds), rng)

    cycles = _resolve_cycles(tcfg)
    if sequential_schedule is not None and not bool(tcfg.get("debug_overfit_one_instance", False)):
        LOGGER.info(
            f"Cyclic training: {len(train_ds)} RC _21 instances × {cycles} cycles "
            f"= {epoch_schedule} updates"
        )

    best_reward = -float("inf")
    best_cycle_objective = float("inf")
    best_cycle_feasibility = -float("inf")
    checkpoint_every = int(tcfg.get("checkpoint_every", 0))
    min_decisions_warn = tcfg.get("min_decisions_warn")
    mismatch_warn = float(dbg.get("objective_mismatch_warn_threshold", 0.05))
    fail_on_mismatch = bool(dbg.get("fail_on_large_objective_mismatch", False))

    cycle_rows: list[dict] = []
    cycle_logger = JsonlLogger(results_dir / "cycle_summaries.jsonl")

    critic_warmup = int(tcfg.get("critic_warmup_steps", 0))
    critic_updates = int(rl_cfg.critic_updates_per_actor_update)

    for epoch in range(epoch_schedule):
        if sequential_schedule is not None:
            batch_idx = [sequential_schedule[epoch]]
            cycle_num = epoch // len(train_ds) + 1 if not bool(tcfg.get("debug_overfit_one_instance")) else 1
        else:
            cycle_num = None
            rng.shuffle(indices)
            batch_idx = indices[:batch_size] if batch_size <= len(indices) else [
                rng.choice(indices) for _ in range(batch_size)
            ]

        record = actor_critic_train_step(
            policy, value_critic, q_critic, optimizer, critic_optimizer,
            train_ds, batch_idx, weights, device, charging_portions,
            use_horizon, use_depot_charging, max_steps_factor, extra_margin, rng, rl_cfg,
            entropy_coef, actor_grad_clip, critic_grad_clip,
            min_decisions_warn=min_decisions_warn,
            global_epoch=epoch,
            cycle_id=int(cycle_num or 1),
            use_amp=use_amp,
            gradient_accumulation_steps=grad_accum,
            scaler=amp_scaler,
            critic_warmup_steps=critic_warmup,
            critic_updates_per_step=critic_updates,
            environment_config=config.get("environment"),
        )
        record["epoch"] = epoch
        record["learning_rate"] = actor_lr
        record["critic_learning_rate"] = critic_lr
        if len(batch_idx) == 1:
            record["instance_name"] = train_ds.names[batch_idx[0]]
            record["instance_idx"] = batch_idx[0]
        if cycle_num is not None:
            record["cycle"] = cycle_num
            record["pass"] = cycle_num

        rel_mm = float(record.get("relative_objective_mismatch", 0.0))
        if rel_mm > mismatch_warn:
            LOGGER.warning(
                "epoch %d | relative objective mismatch %.4f > %.4f",
                epoch, rel_mm, mismatch_warn,
            )
            if fail_on_mismatch:
                raise RuntimeError(
                    f"Objective mismatch too large at epoch {epoch}: rel={rel_mm:.4f}"
                )

        batch_mean_reward = float(record.get("reward", 0.0))
        improved = batch_mean_reward > best_reward
        if improved:
            best_reward = batch_mean_reward

        jsonl.log(record)
        run_logger.log(record)
        train_csv.log(record)
        run_train_csv.log(record)
        cycle_rows.append(record)

        if epoch % int(tcfg.get("log_every", 1)) == 0:
            inst = record.get("instance_name", "?")
            cyc_tag = f" cycle {cycle_num}" if cycle_num is not None else ""
            LOGGER.info(
                f"epoch {epoch:4d}{cyc_tag} | {inst} | reward {batch_mean_reward:10.1f} | "
                f"served {record.get('served_rate', 0):.2f} | "
                f"mm {rel_mm:.4f} | ev {record.get('critic_explained_variance', 0):.3f}"
            )

        if improved:
            _save_checkpoint(
                policy, config, Path(out_cfg["checkpoints"]) / "best.pt",
                value_critic=value_critic, q_critic=q_critic,
            )
        live_ckpt_every = max(1, int(tcfg.get("live_checkpoint_every", 1)))
        if (epoch + 1) % live_ckpt_every == 0:
            _save_checkpoint(
                policy, config, Path(out_cfg["checkpoints"]) / "last.pt",
                value_critic=value_critic, q_critic=q_critic,
                training_meta={"epoch": epoch, "cycle": cycle_num},
            )
        maps_ckpt_every = max(1, int(tcfg.get("maps_checkpoint_every", 10)))
        if (epoch + 1) % maps_ckpt_every == 0:
            _save_checkpoint(
                policy, config,
                Path(out_cfg["checkpoints"]) / "history" / f"epoch_{epoch + 1}.pt",
                value_critic=value_critic, q_critic=q_critic,
                training_meta={"epoch": epoch, "cycle": cycle_num},
            )
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            _save_checkpoint(
                policy, config, Path(out_cfg["checkpoints"]) / f"epoch_{epoch + 1}.pt",
                value_critic=value_critic, q_critic=q_critic,
            )

        # End of cycle — aggregate and checkpoint best cycle metrics
        if sequential_schedule and not bool(tcfg.get("debug_overfit_one_instance")):
            if len(cycle_rows) >= len(train_ds) and len(cycle_rows) % len(train_ds) == 0:
                summary = aggregate_cycle(cycle_rows[-len(train_ds):], cycle_id=int(cycle_num or 0))
                cycle_logger.log(summary)
                c_obj = float(summary.get("mean_objective", float("inf")))
                c_feas = float(summary.get("feasibility_rate", 0.0))
                if c_obj < best_cycle_objective:
                    best_cycle_objective = c_obj
                    _save_checkpoint(
                        policy, config, Path(out_cfg["checkpoints"]) / "best_cycle_objective.pt",
                        value_critic=value_critic, q_critic=q_critic,
                        training_meta={"cycle": cycle_num, "mean_objective": c_obj},
                    )
                if c_feas > best_cycle_feasibility:
                    best_cycle_feasibility = c_feas
                    _save_checkpoint(
                        policy, config, Path(out_cfg["checkpoints"]) / "best_cycle_feasibility.pt",
                        value_critic=value_critic, q_critic=q_critic,
                        training_meta={"cycle": cycle_num, "feasibility_rate": c_feas},
                    )
                LOGGER.info(
                    "cycle %d summary | objective %.0f | served %.2f | feas %.2f",
                    cycle_num, c_obj, summary.get("mean_served_ratio", 0), c_feas,
                )

    _save_checkpoint(
        policy, config, Path(out_cfg["checkpoints"]) / "last.pt",
        value_critic=value_critic, q_critic=q_critic,
    )

    rows = dedupe_metric_rows(jsonl.read())

    if bool(tcfg.get("debug_overfit_one_instance", False)) and rows:
        _check_overfit_improvement(rows, tcfg)

    results_dir = Path(out_cfg["results"])
    figures_dir = Path(out_cfg["figures"]) / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary = _build_final_summary(rows, run_id, epoch_schedule, len(train_ds), log_path, run_dir)
    summary["holdout_instances"] = holdout_meta

    if rows:
        write_csv(results_dir / "training_metrics.csv", rows)
        write_csv(run_dir / "metrics.csv", rows)

    from src.training.training_dashboard import (
        DashboardReport,
        generate_training_dashboard_from_config,
        regenerate_dashboard_index,
    )

    run_solution_behavior_diagnostics = None
    try:
        from src.training.solution_diagnostics import run_solution_behavior_diagnostics
    except ImportError as exc:
        LOGGER.warning("Solution diagnostics module unavailable: %s", exc)

    dash_report = DashboardReport()
    if rows:
        try:
            dash_report = generate_training_dashboard_from_config(
                config, rows=rows, holdout_meta=holdout_meta,
                skip_train_count_check=True,
            )
        except ValueError as exc:
            LOGGER.error("Training dashboard not generated: %s", exc)

        dcfg = config.get("dashboard") or {}
        if (
            run_solution_behavior_diagnostics is not None
            and dash_report.figures
            and dcfg.get("save_representative_traces", True)
            and not bool(tcfg.get("debug_overfit_one_instance", False))
        ):
            try:
                import pandas as pd

                sol_figs, table_html = run_solution_behavior_diagnostics(
                    config,
                    pd.DataFrame(rows),
                    policy,
                    train_ds,
                    device,
                    figures_dir=figures_dir,
                    seed=seed,
                )
                dash_report.figures.extend(sol_figs)
                regenerate_dashboard_index(dash_report, figures_dir, extra_html=table_html)
                LOGGER.info(
                    "Solution behavior diagnostics: %d/%d figures",
                    sum(1 for f in sol_figs if f.generated),
                    len(sol_figs),
                )
            except Exception as exc:
                LOGGER.warning("Solution behavior diagnostics failed: %s", exc)
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


def _check_overfit_improvement(rows: list[dict], tcfg: dict) -> None:
    """Fail or warn if one-instance overfit did not improve served ratio / objective."""
    n = len(rows)
    if n < 20:
        return
    k = max(5, n // 5)
    early = rows[:k]
    late = rows[-k:]

    def _mean(recs, key):
        vals = [float(r.get(key, 0.0)) for r in recs]
        return sum(vals) / max(len(vals), 1)

    early_served = _mean(early, "served_rate")
    late_served = _mean(late, "served_rate")
    early_obj = _mean(early, "final_objective")
    late_obj = _mean(late, "final_objective")
    early_ev = _mean(early, "critic_explained_variance")
    late_ev = _mean(late, "critic_explained_variance")
    early_expired = _mean(early, "newly_expired_customers")
    late_expired = _mean(late, "newly_expired_customers")
    early_stations = _mean(early, "unnecessary_station_actions")
    late_stations = _mean(late, "unnecessary_station_actions")

    LOGGER.info(
        "Overfit check | served %.3f -> %.3f | objective %.0f -> %.0f | ev %.4f -> %.4f | "
        "expired %.1f -> %.1f | bad stations %.1f -> %.1f",
        early_served, late_served, early_obj, late_obj, early_ev, late_ev,
        early_expired, late_expired, early_stations, late_stations,
    )
    rel_obj_improve = (early_obj - late_obj) / max(abs(early_obj), 1.0)
    rel_served_improve = late_served - early_served
    ev_learning = late_ev > max(0.05, early_ev + 0.05)

    strong = (
        rel_served_improve >= 0.02
        or rel_obj_improve >= 0.02
        or (late_expired < early_expired - 0.5)
        or (late_stations < early_stations - 0.5)
    )
    moderate = (
        rel_served_improve >= 0.005
        or rel_obj_improve >= 0.005
        or (late_expired < early_expired - 0.2)
        or (late_stations < early_stations - 0.2)
    )
    quality_non_worse = (
        late_served >= early_served - 1e-6
        and late_obj <= early_obj * 1.001
        and late_expired <= early_expired + 0.5
    )
    # Pass if clear improvement, or critic is learning and quality did not regress.
    improved = strong or (ev_learning and moderate) or (ev_learning and quality_non_worse)

    LOGGER.info(
        "Overfit gate | rel_served=%+.4f rel_obj=%+.4f ev_learning=%s strong=%s moderate=%s -> %s",
        rel_served_improve, rel_obj_improve, ev_learning, strong, moderate,
        "PASS" if improved else "FAIL",
    )

    if not improved:
        msg = (
            "One-instance overfit did NOT improve (served ratio / objective). "
            "Fix reward signal, critic, or actor updates before full training."
        )
        if bool(tcfg.get("fail_on_overfit_regression", True)):
            raise RuntimeError(msg)
        LOGGER.error(msg)


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
        "architecture_version": ARCHITECTURE_VERSION,
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
    if hasattr(policy, "acting_ev_normalizer"):
        payload["feature_normalizer_state"] = {
            "acting_ev": policy.acting_ev_normalizer.state_dict_for_checkpoint(),
            "global": policy.global_normalizer.state_dict_for_checkpoint(),
            "dynamic_continuous": policy.dynamic_continuous_normalizer.state_dict_for_checkpoint(),
        }
    torch.save(payload, path)


def load_checkpoint_training_meta(checkpoint_path: str | Path) -> dict | None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt.get("training_meta")


def load_policy(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    *,
    allow_partial_migration: bool = False,
) -> TransformerEVRPTWPolicy:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = ckpt.get("architecture_version")
    if arch is not None and arch != ARCHITECTURE_VERSION:
        msg = (
            f"Checkpoint architecture_version={arch!r} != current {ARCHITECTURE_VERSION!r}. "
            "Training/eval cannot resume exactly without migration."
        )
        if not allow_partial_migration:
            raise RuntimeError(msg)
        import warnings
        warnings.warn(msg + " Loading with strict=False (partial actor init).")
    policy = TransformerEVRPTWPolicy(ckpt["model_config"], charging_portions=ckpt["charging_portions"])
    missing, unexpected = policy.load_state_dict(ckpt["model_state"], strict=False)
    if arch is None:
        import warnings
        warnings.warn(
            "Checkpoint has no architecture_version — likely incompatible with exact_smdp_dynamic_v1."
        )
    if missing or unexpected:
        import warnings
        warnings.warn(f"Checkpoint key mismatch: missing={missing}, unexpected={unexpected}")
    policy.to(device)
    norm_state = ckpt.get("feature_normalizer_state")
    if norm_state and hasattr(policy, "acting_ev_normalizer"):
        policy.acting_ev_normalizer.load_from_checkpoint(norm_state.get("acting_ev", {}))
        policy.global_normalizer.load_from_checkpoint(norm_state.get("global", {}))
        policy.dynamic_continuous_normalizer.load_from_checkpoint(norm_state.get("dynamic_continuous", {}))
    policy.eval()
    return policy
