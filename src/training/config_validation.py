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

    env_cfg = config.get("environment") or {}
    if str(env_cfg.get("mask_semantics", "")) != "exact_original_evrptw":
        raise ValueError(
            "environment.mask_semantics must be exact_original_evrptw — fast/relaxed masks are forbidden."
        )
    pruning = (env_cfg.get("action_pruning") or {})
    if not bool(pruning.get("enabled", True)):
        raise ValueError(
            "environment.action_pruning.enabled must stay true — hard feasibility and pruning layers are required."
        )

    from src.data.training_phases import normalize_training_phase

    phase = normalize_training_phase(config)
    train_names = list(ds.get("train_instances") or [])
    eval_names = list(ds.get("eval_instances") or [])
    # Phase 3 holdouts may be listed under holdout_by_class only.
    if phase == "all_classes":
        hold = ds.get("holdout_by_class") or {}
        for cls_names in hold.values():
            eval_names.extend(list(cls_names or []))

    for name in train_names + eval_names:
        if phase in {"single_rc", "full_rc"} and not _is_rc21(name):
            raise ValueError(
                f"Instance '{name}' is not Schneider RC _21 — phase {phase} is RC-only."
            )
        if phase == "all_classes" and not str(name).endswith("_21"):
            raise ValueError(f"Instance '{name}' must use full-size _21 for all_classes.")

    overlap = set(train_names) & set(eval_names)
    if overlap:
        raise ValueError(f"train/eval instance overlap: {sorted(overlap)}")

    if train_names and phase == "full_rc" and len(train_names) != 20:
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
    updates = int(tcfg.get("updates") or 0)
    if phase == "single_rc":
        if updates <= 0 and not bool(tcfg.get("debug_overfit_one_instance", False)):
            warnings.append("Phase 1 should set training.updates (e.g. 500).")
    elif cycles < 50:
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
        "enabled": False, "start_coef": 1.0, "end_coef": 1.0, "warmup_cycles": 0,
    })
    rcfg.setdefault("q_eval_mode", "topk")
    rcfg.setdefault("max_q_actions", 64)
    env_cfg = config.setdefault("environment", {})
    env_cfg.setdefault("mask_semantics", "exact_original_evrptw")
    env_cfg.setdefault("implementation", "optimized")
    pruning = env_cfg.setdefault("action_pruning", {})
    pruning.setdefault("enabled", True)
    pruning.setdefault("block_exact_noop_actions", True)
    pruning.setdefault("block_repeated_zero_progress_loops", True)
    pruning.setdefault("prune_dominated_recharge_labels", True)
    pruning.setdefault("prune_station_when_customer_available", False)
    rsearch = env_cfg.setdefault("recharge_search", {})
    rsearch.setdefault("max_station_hops", 3)
    rsearch.setdefault("use_dominance_pruning", True)
    cache = env_cfg.setdefault("cache", {})
    cache.setdefault("enabled", True)
    cache.setdefault("cache_by_state_version", True)
    diag = config.setdefault("diagnostics", {})
    diag.setdefault("parameter_delta_every_updates", 20)
    diag.setdefault("full_mask_reason_dump_every_updates", 20)
    diag.setdefault("timing_enabled", True)
    ckpt = config.setdefault("checkpointing", {})
    ckpt.setdefault("save_last_every_updates", 20)
    ckpt.setdefault("save_cycle_checkpoint", True)
    config["model"].setdefault("use_edge_aware_encoder", False)
    config["training"].setdefault("use_local_search_in_training", False)
    config["training"].setdefault("method", "actor_critic")
    config["training"].setdefault("generate_static_dashboard", False)
    config["training"].setdefault("live_checkpoint_every", 0)  # superseded by checkpointing.save_last_every_updates
    config["training"].setdefault("maps_checkpoint_every", 0)
    dcfg = config.setdefault("dashboard", {})
    dcfg.setdefault("save_representative_traces", True)
    dcfg.setdefault("num_random_solution_maps", 1)
    dcfg.setdefault("include_best_final_cycle", True)
    dcfg.setdefault("include_worst_final_cycle", True)
    dcfg.setdefault("include_best_overall", True)
    dcfg.setdefault("live_random_diagnostics", False)
    dcfg.setdefault("live_random_interval_sec", 45)
    dcfg.setdefault("live_refresh_min_epoch_delta", 1)
    dcfg.setdefault("refresh_every_cycles", 1)
    dcfg.setdefault("generate_full_artifacts_during_training", False)
    dcfg.setdefault("generate_full_solution_maps_every_cycles", 5)
    dcfg.setdefault("save_representative_traces_every_cycles", 5)
    mcfg = config.setdefault("maps_dashboard", {})
    mcfg.setdefault("port", 8081)
    mcfg.setdefault("cache_snapshots", True)
    mcfg.setdefault("use_cached_if_present", True)
    out_cfg = config.setdefault("outputs", {})
    out_cfg.setdefault("traces", str(Path(out_cfg.get("root", "outputs")) / "traces"))
    return config
