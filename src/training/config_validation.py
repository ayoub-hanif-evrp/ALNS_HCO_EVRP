"""Validate training config for safe actor-critic baseline runs."""
from __future__ import annotations

import sys
from pathlib import Path

from src.data.instance_class import instance_class
from src.env.constraints import ConstraintConfig
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def _is_rc21(name: str) -> bool:
    return instance_class(name) == "RC" and str(name).endswith("_21")


def validate_training_config(config: dict) -> list[str]:
    """Return warnings; raise ValueError on hard failures."""
    warnings: list[str] = []
    ds = config.get("dataset") or {}
    tcfg = config.get("training") or {}
    rcfg = config.get("rl") or {}
    mcfg = config.get("model") or {}
    dbg = config.get("debug") or {}
    constraint_cfg = ConstraintConfig.from_config(config)
    constraint_cfg.validate_or_fail()

    train_names = list(ds.get("train_instances") or [])
    eval_names = list(ds.get("eval_instances") or [])

    for name in train_names + eval_names:
        if not _is_rc21(name):
            raise ValueError(
                f"Instance '{name}' is not Schneider RC _21 — training is RC-only."
            )

    overlap = set(train_names) & set(eval_names)
    if overlap:
        raise ValueError(f"train/eval instance overlap: {sorted(overlap)}")

    if train_names and len(train_names) != 20:
        expected = int(ds.get("expected_train_count", 20))
        if len(train_names) != expected:
            warnings.append(
                f"Train instance count {len(train_names)} != expected {expected} "
                f"(Schneider RC _21 has 16 total; R class has 23)."
            )

    q_on = bool(rcfg.get("use_counterfactual_critic", False))
    ms_on = bool(rcfg.get(
        "use_multi_sample_rollout_baseline",
        rcfg.get("use_pomo", False),
    ))
    edge_on = bool(mcfg.get("use_edge_aware_encoder", False))
    if q_on and ms_on and edge_on:
        if not bool(dbg.get("allow_all_advanced_modules", False)):
            raise ValueError(
                "Refusing to train with Q critic + multi-sample rollout baseline + edge encoder all enabled. "
                "Enable one module at a time, or set debug.allow_all_advanced_modules: true."
            )
        warnings.append("All advanced modules enabled — expect instability / OOM risk.")

    if q_on and ms_on:
        warnings.append(
            "Q critic and multi-sample rollout baseline both enabled — use only after baseline is stable."
        )
    if bool(rcfg.get("use_potential_shaping", False)) and not dbg.get("allow_potential_shaping"):
        warnings.append("Potential shaping is enabled — disable for objective-aligned baseline.")

    if bool(tcfg.get("use_local_search_in_training", False)):
        warnings.append("Local search in training is enabled — disable for baseline runs.")

    cycles = int(tcfg.get("cycles") or tcfg.get("passes") or 0)
    if cycles < 50:
        warnings.append(f"cycles={cycles} is short for MARL EVRP; recommend >= 50.")

    for msg in warnings:
        LOGGER.warning(msg)
    return warnings


def apply_safe_training_defaults(config: dict) -> dict:
    """Fill missing safe-baseline keys without overriding explicit user values."""
    config.setdefault("constraints", {})
    config.setdefault("rl", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("debug", {})
    ccfg = config["constraints"]
    ccfg.setdefault("time_windows_mode", "hard_original")
    ccfg.setdefault("allow_late_service", False)
    ccfg.setdefault("relax_time_windows", False)
    ccfg.setdefault("modify_instance_time_windows", False)
    rcfg = config["rl"]
    rcfg.setdefault("use_step_rewards", True)
    rcfg.setdefault("use_gae", True)
    rcfg.setdefault("use_counterfactual_critic", False)
    rcfg.setdefault("use_multi_sample_rollout_baseline", rcfg.get("use_pomo", False))
    rcfg.setdefault("multi_sample_K", rcfg.get("pomo_K", 1))
    rcfg.setdefault("multi_sample_mix_rho", rcfg.get("pomo_mix_rho", 0.1))
    rcfg.setdefault("use_potential_shaping", False)
    rcfg.setdefault("normalize_advantages", True)
    rcfg.setdefault("gamma", 1.0)
    rcfg.setdefault("gae_lambda", 0.95)
    rcfg.setdefault("normalize_returns_for_critic", False)
    rcfg.setdefault("use_served_customer_bonus", True)
    rcfg.setdefault("served_bonus_mode", "penalty_equivalent")
    rcfg.setdefault("served_bonus_coef", 0.5)
    rcfg.setdefault("use_early_depot_penalty", True)
    rcfg.setdefault("early_depot_penalty_coef", 2.0)
    rcfg.setdefault("use_urgency_bonus", True)
    rcfg.setdefault("urgency_bonus_coef", 0.5)
    rcfg.setdefault("use_newly_expired_customer_penalty", True)
    rcfg.setdefault("newly_expired_customer_penalty", 1.0)
    rcfg.setdefault("use_time_waste_penalty", True)
    rcfg.setdefault("time_waste_penalty_coef", 0.05)
    rcfg.setdefault("time_waste_scale", 1.0)
    rcfg.setdefault("use_unnecessary_station_penalty", True)
    rcfg.setdefault("unnecessary_station_penalty", 1.0)
    rcfg.setdefault("use_urgent_customer_rank_penalty", True)
    rcfg.setdefault("urgent_rank_penalty_coef", 0.2)
    rcfg.setdefault("urgent_rank_k", 10)
    rcfg.setdefault("use_customer_priority_logit_bias", True)
    rcfg.setdefault("customer_logit_bonus", 1.0)
    rcfg.setdefault("unnecessary_station_logit_penalty", 1.5)
    rcfg.setdefault("use_step_penalty", False)
    rcfg.setdefault("step_penalty", 0.01)
    rcfg.setdefault("use_depot_logit_penalty", True)
    rcfg.setdefault("depot_logit_penalty", 2.0)
    rcfg.setdefault("depot_penalty_until_served_ratio", 0.95)
    rcfg.setdefault("use_no_progress_penalty", True)
    rcfg.setdefault("no_progress_threshold", 10)
    rcfg.setdefault("no_progress_penalty_coef", 0.05)
    rcfg.setdefault("terminal_penalty_schedule", {
        "enabled": True, "start_coef": 0.25, "end_coef": 1.0, "warmup_cycles": 20,
    })
    rcfg.setdefault("q_eval_mode", "topk")
    rcfg.setdefault("max_q_actions", 64)
    config["model"].setdefault("use_edge_aware_encoder", False)
    config["training"].setdefault("use_local_search_in_training", False)
    config["training"].setdefault("method", "actor_critic")
    config["training"].setdefault("generate_static_dashboard", True)
    dcfg = config.setdefault("dashboard", {})
    dcfg.setdefault("save_representative_traces", True)
    dcfg.setdefault("num_random_solution_maps", 1)
    dcfg.setdefault("include_best_final_cycle", True)
    dcfg.setdefault("include_worst_final_cycle", True)
    dcfg.setdefault("include_best_overall", True)
    dcfg.setdefault("live_random_diagnostics", False)
    dcfg.setdefault("live_random_interval_sec", 45)
    dcfg.setdefault("live_refresh_min_epoch_delta", 1)
    mcfg = config.setdefault("maps_dashboard", {})
    mcfg.setdefault("port", 8081)
    mcfg.setdefault("cache_snapshots", True)
    mcfg.setdefault("use_cached_if_present", True)
    config["training"].setdefault("live_checkpoint_every", 1)
    config["training"].setdefault("maps_checkpoint_every", 10)
    out_cfg = config.setdefault("outputs", {})
    out_cfg.setdefault("traces", str(Path(out_cfg.get("root", "outputs")) / "traces"))
    return config
