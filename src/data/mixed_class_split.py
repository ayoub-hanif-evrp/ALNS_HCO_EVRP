"""Mixed C / R / RC curriculum split and balanced class sampling.

``resolve_rc21_split`` remains the resolver for Stages 1–2 (RC stabilization).
This module supports Stage 3+ balanced joint training and per-class holdouts.
"""
from __future__ import annotations

import random
from pathlib import Path

from src.data.instance_class import instance_class
from src.data.instance_loader import list_instance_dirs
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

CURRICULUM_STAGES = (
    "single_rc",
    "single_rc_overfit",
    "full_rc",
    "rc_stabilization",
    "all_classes",
    "mixed_c_r_rc",
)


def _full_size_instances(dataset_path: str | Path, cls: str) -> list[str]:
    """Return ``_21`` instances of class C, R, or RC."""
    root = Path(dataset_path)
    names = []
    for d in list_instance_dirs(root):
        name = d.name
        if not str(name).endswith("_21"):
            continue
        if instance_class(name) == cls:
            names.append(name)
    return sorted(names)


def resolve_mixed_c_r_rc_split(config: dict) -> dict:
    """Resolve per-class train/holdout lists for balanced C+R+RC training.

    Returns a dict:
      {
        "train_by_class": {"C": [...], "R": [...], "RC": [...]},
        "holdout_by_class": {"C": [...], "R": [...], "RC": [...]},
        "train_names": flat union,
        "holdout_names": flat union,
        "meta": {...},
      }
    """
    ds = config.get("dataset") or {}
    path = ds["path"]
    holdout_cfg = ds.get("holdout_by_class") or {}
    train_by_class: dict[str, list[str]] = {}
    holdout_by_class: dict[str, list[str]] = {}

    for cls in ("C", "R", "RC"):
        all_cls = _full_size_instances(path, cls)
        holdout = list(holdout_cfg.get(cls) or [])
        for name in holdout:
            if name not in all_cls:
                raise ValueError(f"Holdout '{name}' not found among {cls} _21 instances")
        train = sorted(n for n in all_cls if n not in set(holdout))
        if not train:
            raise ValueError(f"No training instances left for class {cls}")
        train_by_class[cls] = train
        holdout_by_class[cls] = sorted(holdout)

    train_names = sorted({n for lst in train_by_class.values() for n in lst})
    holdout_names = sorted({n for lst in holdout_by_class.values() for n in lst})
    overlap = set(train_names) & set(holdout_names)
    if overlap:
        raise ValueError(f"Train/holdout overlap in mixed split: {sorted(overlap)}")

    meta = {
        "curriculum_stage": "mixed_c_r_rc",
        "train_counts": {k: len(v) for k, v in train_by_class.items()},
        "holdout_counts": {k: len(v) for k, v in holdout_by_class.items()},
    }
    LOGGER.info(
        "Mixed C/R/RC split | train C=%d R=%d RC=%d | holdout C=%d R=%d RC=%d",
        meta["train_counts"]["C"], meta["train_counts"]["R"], meta["train_counts"]["RC"],
        meta["holdout_counts"]["C"], meta["holdout_counts"]["R"], meta["holdout_counts"]["RC"],
    )
    return {
        "train_by_class": train_by_class,
        "holdout_by_class": holdout_by_class,
        "train_names": train_names,
        "holdout_names": holdout_names,
        "meta": meta,
    }


class BalancedClassSampler:
    """Sample instances with P(C)=P(R)=P(RC)=1/3, then uniform within class."""

    def __init__(self, train_by_class: dict[str, list[str]], rng: random.Random | None = None):
        self.train_by_class = {
            k: list(v) for k, v in train_by_class.items() if v
        }
        if set(self.train_by_class) != {"C", "R", "RC"}:
            missing = {"C", "R", "RC"} - set(self.train_by_class)
            raise ValueError(f"BalancedClassSampler needs all classes; missing {missing}")
        self.rng = rng or random.Random(0)
        self.classes = ["C", "R", "RC"]

    def sample_name(self) -> str:
        cls = self.rng.choice(self.classes)
        return self.rng.choice(self.train_by_class[cls])

    def sample_batch(self, n: int) -> list[str]:
        return [self.sample_name() for _ in range(n)]


def resolve_curriculum_split(config: dict) -> tuple[list[str], list[str], dict]:
    """Dispatch on ``dataset.training_phase`` / legacy ``curriculum_stage``.

    Prefer :func:`src.data.training_phases.resolve_training_phase_split`.
    """
    from src.data.training_phases import resolve_training_phase_split

    return resolve_training_phase_split(config)
