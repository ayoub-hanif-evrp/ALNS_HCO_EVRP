"""Discover and validate Schneider RC _21 train/eval splits."""
from __future__ import annotations

from pathlib import Path

from src.hanif.data.instance_class import instance_class
from src.hanif.data.instance_loader import list_instance_dirs
from src.hanif.utils.logging_utils import get_logger

LOGGER = get_logger()


def is_rc21(name: str) -> bool:
    return instance_class(name) == "RC" and str(name).endswith("_21")


def discover_rc21_instances(dataset_path: str | Path) -> list[str]:
    """Return sorted RC _21 instance folder names under ``dataset_path``."""
    root = Path(dataset_path)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    names = sorted(d.name for d in list_instance_dirs(root) if is_rc21(d.name))
    return names


def resolve_rc21_split(config: dict) -> tuple[list[str], list[str], dict]:
    """Resolve train/eval instance lists with strict RC _21 validation.

    Returns ``(train_names, eval_names, meta)``.
    Raises ``ValueError`` when the expected train count cannot be satisfied.
    """
    ds = config.get("dataset") or {}
    path = ds["path"]
    all_rc21 = discover_rc21_instances(path)

    eval_names = list(ds.get("eval_instances") or [])
    for name in eval_names:
        if not is_rc21(name):
            raise ValueError(f"Eval instance '{name}' is not Schneider RC _21")
        if name not in all_rc21:
            raise ValueError(f"Eval instance '{name}' not found in {path}")

    expected_train = int(ds.get("expected_train_count", 20))
    expected_eval = int(ds.get("expected_eval_count", 3))
    fail_on_mismatch = bool(ds.get("fail_if_train_count_mismatch", True))

    train_names = list(ds.get("train_instances") or [])
    if ds.get("auto_discover_rc21", True) and not train_names:
        holdout = set(eval_names)
        train_names = [n for n in all_rc21 if n not in holdout]

    for name in train_names:
        if not is_rc21(name):
            raise ValueError(f"Train instance '{name}' is not Schneider RC _21")
        if name not in all_rc21:
            raise ValueError(f"Train instance '{name}' not found in {path}")

    overlap = set(train_names) & set(eval_names)
    if overlap:
        raise ValueError(f"Train/eval overlap: {sorted(overlap)}")

    if len(eval_names) != expected_eval:
        msg = (
            f"Expected {expected_eval} reserved RC _21 test instances, "
            f"got {len(eval_names)}: {eval_names}"
        )
        if fail_on_mismatch:
            raise ValueError(msg)
        LOGGER.warning(msg)

    if len(train_names) < expected_train:
        msg = (
            f"SCHNEIDER RC _21 SPLIT ERROR: need {expected_train} training instances, "
            f"but only {len(train_names)} available after reserving {len(eval_names)} test.\n"
            f"  Total RC _21 found in dataset: {len(all_rc21)}\n"
            f"  All RC _21: {all_rc21}\n"
            f"  Train ({len(train_names)}): {train_names}\n"
            f"  Reserved test ({len(eval_names)}): {eval_names}\n"
            f"Schneider RC class has {len(all_rc21)} full _21 instances (not 23 like R class). "
            f"Set dataset.expected_train_count: {len(train_names)} to match availability, "
            f"or set dataset.fail_if_train_count_mismatch: false to override."
        )
        if fail_on_mismatch:
            raise ValueError(msg)
        LOGGER.error(msg)
    elif len(train_names) != expected_train:
        msg = (
            f"Train instance count {len(train_names)} != expected {expected_train}. "
            f"Train: {train_names}"
        )
        if fail_on_mismatch:
            raise ValueError(msg)
        LOGGER.warning(msg)

    meta = {
        "total_rc21_found": len(all_rc21),
        "all_rc21_instances": all_rc21,
        "train_instances": train_names,
        "eval_instances": eval_names,
        "num_train": len(train_names),
        "num_eval": len(eval_names),
        "expected_train_count": expected_train,
        "expected_eval_count": expected_eval,
    }
    return train_names, eval_names, meta


def log_rc21_split_startup(meta: dict, cycles: int) -> None:
    """Print a clear startup banner for the RC _21 training split."""
    n_train = meta["num_train"]
    total_updates = n_train * cycles
    LOGGER.info("=" * 60)
    LOGGER.info("Schneider RC _21 dataset split")
    LOGGER.info("  Total RC _21 instances found : %d", meta["total_rc21_found"])
    LOGGER.info("  Train instances (%d)           : %s", n_train, ", ".join(meta["train_instances"]))
    LOGGER.info(
        "  Reserved test (%d, not in train): %s",
        meta["num_eval"],
        ", ".join(meta["eval_instances"]) or "(none)",
    )
    LOGGER.info("  Training cycles                : %d", cycles)
    LOGGER.info("  Total policy updates           : %d (= %d × %d)", total_updates, n_train, cycles)
    LOGGER.info("=" * 60)
