"""Training-phase dataset resolution (exactly three experimental phases).

Phases:
  single_rc   — Phase 1: one RC _21 instance, update-based debugging
  full_rc     — Phase 2: all RC _21 training instances (holdout reserved)
  all_classes — Phase 3: balanced C + R + RC joint training
"""
from __future__ import annotations

from pathlib import Path

from src.data.instance_class import instance_class
from src.data.instance_loader import list_instance_dirs
from src.data.mixed_class_split import (
    BalancedClassSampler,
    resolve_mixed_c_r_rc_split,
)
from src.data.rc21_split import log_rc21_split_startup, resolve_rc21_split
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

TRAINING_PHASES = ("single_rc", "full_rc", "all_classes")

# Backward-compatible aliases from older curriculum_stage keys.
_PHASE_ALIASES = {
    "single_rc": "single_rc",
    "single_rc_overfit": "single_rc",
    "full_rc": "full_rc",
    "rc_stabilization": "full_rc",
    "all_classes": "all_classes",
    "mixed_c_r_rc": "all_classes",
}


def normalize_training_phase(config: dict) -> str:
    """Return canonical phase name from ``dataset.training_phase`` or legacy keys."""
    ds = config.get("dataset") or {}
    raw = ds.get("training_phase")
    if raw is None:
        raw = ds.get("curriculum_stage", "full_rc")
    key = str(raw).lower().strip()
    if key not in _PHASE_ALIASES:
        raise ValueError(
            f"Unknown training_phase '{raw}'. Expected one of {TRAINING_PHASES} "
            f"(aliases: {sorted(_PHASE_ALIASES)})"
        )
    return _PHASE_ALIASES[key]


def _suffix_instances(dataset_path: str | Path, cls: str, suffix: str) -> list[str]:
    root = Path(dataset_path)
    names = []
    for d in list_instance_dirs(root):
        name = d.name
        if not str(name).endswith(suffix):
            continue
        if instance_class(name) == cls:
            names.append(name)
    return sorted(names)


def resolve_single_rc_split(config: dict) -> tuple[list[str], list[str], dict]:
    """Phase 1: exactly one RC training instance; holdouts remain reserved."""
    ds = config.get("dataset") or {}
    path = ds["path"]
    suffix = str(ds.get("instance_suffix", "_21"))
    all_rc = _suffix_instances(path, "RC", suffix)
    eval_names = list(ds.get("eval_instances") or [])
    holdout = set(eval_names)

    requested = list(ds.get("train_instances") or [])
    single = ds.get("single_instance") or ds.get("phase1_instance")
    if single:
        requested = [str(single)]

    n_want = int(ds.get("number_of_instances", 1))
    if n_want != 1:
        raise ValueError(
            f"Phase 1 (single_rc) requires number_of_instances: 1, got {n_want}"
        )

    pool = [n for n in all_rc if n not in holdout]
    if requested:
        train_names = requested
    else:
        # Prefer rc102_21 when available (stable historical debug instance).
        preferred = "rc102_21"
        train_names = [preferred] if preferred in pool else pool[:1]

    if len(train_names) != 1:
        raise ValueError(
            f"Phase 1 expects exactly 1 training instance, got {train_names}"
        )
    name = train_names[0]
    if name not in all_rc:
        raise ValueError(f"Phase 1 instance '{name}' is not an RC {suffix} instance")
    if name in holdout:
        raise ValueError(f"Phase 1 instance '{name}' overlaps reserved holdout")

    meta = {
        "training_phase": "single_rc",
        "curriculum_stage": "single_rc",
        "total_rc21_found": len(all_rc),
        "all_rc21_instances": all_rc,
        "train_instances": train_names,
        "eval_instances": eval_names,
        "num_train": 1,
        "num_eval": len(eval_names),
        "expected_train_count": 1,
        "expected_eval_count": len(eval_names),
        "number_of_instances": 1,
    }
    return train_names, eval_names, meta


def resolve_full_rc_split(config: dict) -> tuple[list[str], list[str], dict]:
    """Phase 2: all RC _21 training instances with reserved holdout."""
    ds = dict(config.get("dataset") or {})
    # Ensure auto-discovery uses holdout exclusion.
    ds.setdefault("auto_discover_rc21", True)
    if ds.get("use_all_training_instances", True) and not ds.get("train_instances"):
        ds["auto_discover_rc21"] = True
    cfg = dict(config)
    cfg["dataset"] = ds
    train_names, eval_names, meta = resolve_rc21_split(cfg)
    meta = dict(meta)
    meta["training_phase"] = "full_rc"
    meta["curriculum_stage"] = "full_rc"
    return train_names, eval_names, meta


def resolve_all_classes_split(config: dict) -> tuple[list[str], list[str], dict]:
    """Phase 3: balanced C+R+RC with per-class holdouts."""
    mixed = resolve_mixed_c_r_rc_split(config)
    meta = dict(mixed["meta"])
    meta["training_phase"] = "all_classes"
    meta["curriculum_stage"] = "all_classes"
    meta["mixed"] = mixed
    meta["train_instances"] = mixed["train_names"]
    meta["eval_instances"] = mixed["holdout_names"]
    meta["num_train"] = len(mixed["train_names"])
    meta["num_eval"] = len(mixed["holdout_names"])
    return mixed["train_names"], mixed["holdout_names"], meta


def resolve_training_phase_split(config: dict) -> tuple[list[str], list[str], dict]:
    """Dispatch on ``dataset.training_phase`` (preferred) or legacy curriculum_stage."""
    phase = normalize_training_phase(config)
    if phase == "single_rc":
        return resolve_single_rc_split(config)
    if phase == "full_rc":
        return resolve_full_rc_split(config)
    return resolve_all_classes_split(config)


def log_training_phase_startup(meta: dict, *, updates: int | None = None, cycles: int | None = None) -> None:
    phase = meta.get("training_phase") or meta.get("curriculum_stage", "?")
    LOGGER.info("=" * 60)
    LOGGER.info("Training phase: %s", phase)
    LOGGER.info("  Train instances (%d): %s", meta.get("num_train", 0), ", ".join(meta.get("train_instances") or []))
    LOGGER.info(
        "  Holdout / eval (%d): %s",
        meta.get("num_eval", 0),
        ", ".join(meta.get("eval_instances") or []) or "(none)",
    )
    if phase == "all_classes" and "mixed" in meta:
        counts = meta["mixed"]["meta"]["train_counts"]
        LOGGER.info(
            "  Balanced class sampling P(C)=P(R)=P(RC)=1/3 | train C=%d R=%d RC=%d",
            counts["C"], counts["R"], counts["RC"],
        )
    if updates is not None and int(updates) > 0:
        LOGGER.info("  Planned updates: %d", int(updates))
    if cycles is not None and int(cycles) > 0:
        n = int(meta.get("num_train") or 0)
        LOGGER.info("  Planned cycles: %d (= %d updates if sequential)", int(cycles), int(cycles) * max(n, 1))
    LOGGER.info("=" * 60)


def build_balanced_schedule(
    train_by_class: dict[str, list[str]],
    name_to_idx: dict[str, int],
    *,
    cycles: int,
    updates_per_cycle: int | None = None,
    rng,
) -> list[int]:
    """Build a flat index schedule with equal class probability each step."""
    sampler = BalancedClassSampler(train_by_class, rng=rng)
    if updates_per_cycle is None:
        # One cycle ≈ average class training size (keeps wall-clock comparable to RC).
        sizes = [len(v) for v in train_by_class.values() if v]
        updates_per_cycle = max(1, int(round(sum(sizes) / max(len(sizes), 1))))
    schedule: list[int] = []
    for _ in range(int(cycles)):
        for _ in range(int(updates_per_cycle)):
            name = sampler.sample_name()
            schedule.append(name_to_idx[name])
    return schedule


__all__ = [
    "TRAINING_PHASES",
    "BalancedClassSampler",
    "build_balanced_schedule",
    "log_training_phase_startup",
    "log_rc21_split_startup",
    "normalize_training_phase",
    "resolve_training_phase_split",
]
