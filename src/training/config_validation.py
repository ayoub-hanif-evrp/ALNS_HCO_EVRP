"""Validate training config for safe actor-critic baseline runs."""
from __future__ import annotations

import sys

from src.data.instance_class import instance_class
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
    pomo_on = bool(rcfg.get("use_pomo", False))
    edge_on = bool(mcfg.get("use_edge_aware_encoder", False))
    if q_on and pomo_on and edge_on:
        if not bool(dbg.get("allow_all_advanced_modules", False)):
            raise ValueError(
                "Refusing to train with Q critic + POMO + edge encoder all enabled. "
                "Enable one module at a time, or set debug.allow_all_advanced_modules: true."
            )
        warnings.append("All advanced modules enabled — expect instability / OOM risk.")

    if q_on and pomo_on:
        warnings.append("Q critic and POMO both enabled — use only after baseline is stable.")
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
    config.setdefault("rl", {})
    config.setdefault("model", {})
    config.setdefault("training", {})
    config.setdefault("debug", {})
    rcfg = config["rl"]
    rcfg.setdefault("use_step_rewards", True)
    rcfg.setdefault("use_gae", True)
    rcfg.setdefault("use_counterfactual_critic", False)
    rcfg.setdefault("use_pomo", False)
    rcfg.setdefault("use_potential_shaping", False)
    rcfg.setdefault("normalize_advantages", True)
    rcfg.setdefault("gamma", 1.0)
    rcfg.setdefault("gae_lambda", 0.95)
    rcfg.setdefault("normalize_returns_for_critic", True)
    rcfg.setdefault("use_served_customer_bonus", True)
    rcfg.setdefault("served_bonus_mode", "penalty_equivalent")
    rcfg.setdefault("served_bonus_coef", 0.5)
    rcfg.setdefault("use_early_depot_penalty", True)
    rcfg.setdefault("early_depot_penalty_coef", 1.0)
    rcfg.setdefault("use_step_penalty", True)
    rcfg.setdefault("step_penalty", 0.01)
    rcfg.setdefault("use_depot_logit_penalty", True)
    rcfg.setdefault("depot_logit_penalty", 2.0)
    rcfg.setdefault("depot_penalty_until_served_ratio", 0.95)
    rcfg.setdefault("terminal_penalty_schedule", {
        "enabled": True, "start_coef": 0.25, "end_coef": 1.0, "warmup_cycles": 20,
    })
    rcfg.setdefault("q_eval_mode", "topk")
    rcfg.setdefault("max_q_actions", 64)
    config["model"].setdefault("use_edge_aware_encoder", False)
    config["training"].setdefault("use_local_search_in_training", False)
    config["training"].setdefault("method", "actor_critic")
    return config
