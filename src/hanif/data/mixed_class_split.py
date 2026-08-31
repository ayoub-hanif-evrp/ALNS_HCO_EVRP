"""Mixed C / R / RC curriculum split and balanced class sampling.

``resolve_rc21_split`` remains the resolver for Stages 1–2 (RC stabilization).
This module supports Stage 3+ balanced joint training and per-class holdouts.
"""
from __future__ import annotations

import random
from pathlib import Path

from src.hanif.data.instance_class import instance_class
from src.hanif.data.instance_loader import list_instance_dirs
from src.hanif.utils.logging_utils import get_logger

LOGGER = get_logger()

CURRICULUM_STAGES = (
    "single_rc",
    "single_rc_overfit",
    "full_rc",
    "rc_stabilization",
    "all_classes",
    "mixed_c_r_rc",
)


def _is_type2_instance(name: str) -> bool:
    """Solomon/Schneider type-2 (loose TW): c2xx / r2xx / rc2xx."""
    n = str(name).lower()
    if n.startswith("rc"):
        return len(n) >= 3 and n[2] == "2"
    if n.startswith("c") or n.startswith("r"):
        return len(n) >= 2 and n[1] == "2"
    return False


def _is_type1_instance(name: str) -> bool:
    n = str(name).lower()
    if n.startswith("rc"):
        return len(n) >= 3 and n[2] == "1"
    if n.startswith("c") or n.startswith("r"):
        return len(n) >= 2 and n[1] == "1"
    return False


def _apply_tw_family_filter(names: list[str], tw_family: str) -> list[str]:
    fam = str(tw_family or "all").strip().lower()
    if fam in {"", "all", "both", "type1+type2", "mix", "mixed"}:
        return list(names)
    if fam in {"type2", "t2", "loose", "2xx"}:
        return [n for n in names if _is_type2_instance(n)]
    if fam in {"type1", "t1", "tight", "1xx"}:
        return [n for n in names if _is_type1_instance(n)]
    # Curriculum alias: start from type-2 pool when explicitly requested.
    if fam in {"curriculum", "type2_then_all", "type2_then_mix"}:
        return list(names)
    raise ValueError(
        f"Unknown dataset.tw_family '{tw_family}'. Use all | type1 | type2 | mix."
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
    tw_family = str(ds.get("tw_family", "all") or "all")
    train_by_class: dict[str, list[str]] = {}
    holdout_by_class: dict[str, list[str]] = {}

    for cls in ("C", "R", "RC"):
        all_cls = _apply_tw_family_filter(_full_size_instances(path, cls), tw_family)
        holdout = [n for n in list(holdout_cfg.get(cls) or []) if n in set(all_cls)]
        missing = [n for n in list(holdout_cfg.get(cls) or []) if n not in set(all_cls)]
        if missing and tw_family.lower() in {"all", "", "both"}:
            raise ValueError(f"Holdout '{missing}' not found among {cls} _21 instances")
        train = sorted(n for n in all_cls if n not in set(holdout))
        if not train:
            raise ValueError(
                f"No training instances left for class {cls} "
                f"(tw_family={tw_family})"
            )
        train_by_class[cls] = train
        holdout_by_class[cls] = sorted(holdout)

    train_names = sorted({n for lst in train_by_class.values() for n in lst})
    holdout_names = sorted({n for lst in holdout_by_class.values() for n in lst})
    overlap = set(train_names) & set(holdout_names)
    if overlap:
        raise ValueError(f"Train/holdout overlap in mixed split: {sorted(overlap)}")

    meta = {
        "curriculum_stage": "mixed_c_r_rc",
        "tw_family": tw_family,
        "train_counts": {k: len(v) for k, v in train_by_class.items()},
        "holdout_counts": {k: len(v) for k, v in holdout_by_class.items()},
    }
    LOGGER.info(
        "Mixed C/R/RC split | tw_family=%s | train C=%d R=%d RC=%d | holdout C=%d R=%d RC=%d",
        tw_family,
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

    Prefer :func:`src.hanif.data.training_phases.resolve_training_phase_split`.
    """
    from src.hanif.data.training_phases import resolve_training_phase_split

    return resolve_training_phase_split(config)
