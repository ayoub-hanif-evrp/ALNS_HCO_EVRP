"""Fixed mini curriculum: exactly 5 C + 5 RC + 5 R instances, repeated over cycles."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

from src.data.evrptw_dataset import EVRPTWDataset, _SubsetDataset
from src.data.instance_class import CLASSES, group_indices_by_class, instance_class

INSTANCES_PER_CLASS = 5
MANIFEST_FILENAME = "fixed_train_manifest.json"


def _normalize_class_key(key: str) -> str:
    return str(key).strip().upper()


def _default_first_five_per_class(full_names: list[str]) -> dict[str, list[str]]:
    groups = group_indices_by_class(full_names)
    selected: dict[str, list[str]] = {}
    for cls in CLASSES:
        names = sorted(full_names[i] for i in groups.get(cls, []))
        if len(names) < INSTANCES_PER_CLASS:
            raise ValueError(
                f"Need at least {INSTANCES_PER_CLASS} {cls} instances in dataset; found {len(names)}"
            )
        selected[cls] = names[:INSTANCES_PER_CLASS]
    return selected


def resolve_fixed_train_instances(
    full_names: list[str],
    config: dict,
    seed: int,
) -> dict[str, list[str]]:
    """Resolve exactly 5 instance names per class (C, RC, R)."""
    ds_cfg = config.get("dataset") or {}
    raw = ds_cfg.get("fixed_train_instances") or {}
    name_set = set(full_names)

    if raw and any(raw.get(c) for c in CLASSES):
        selected: dict[str, list[str]] = {}
        for cls in CLASSES:
            cfg_key = next((k for k in raw if _normalize_class_key(k) == cls), cls)
            names = list(raw.get(cfg_key) or raw.get(cls) or [])
            if len(names) != INSTANCES_PER_CLASS:
                raise ValueError(
                    f"fixed_train_instances.{cls} must list exactly {INSTANCES_PER_CLASS} names; "
                    f"got {len(names)}"
                )
            for name in names:
                if name not in name_set:
                    raise ValueError(f"Fixed train instance not in dataset: {name}")
                if instance_class(name) != cls:
                    raise ValueError(f"Instance {name} is class {instance_class(name)}, expected {cls}")
            selected[cls] = list(names)
        return selected

    _ = seed  # reserved for future seeded selection strategies
    return _default_first_five_per_class(full_names)


def fixed_train_set_hash(fixed_by_class: dict[str, list[str]]) -> str:
    """Stable SHA256 of the 15 fixed training instance names."""
    all_names = sorted(n for names in fixed_by_class.values() for n in names)
    payload = "|".join(all_names).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def all_fixed_names(fixed_by_class: dict[str, list[str]]) -> set[str]:
    return {n for names in fixed_by_class.values() for n in names}


def total_expected_updates(fixed_by_class: dict[str, list[str]], cycles_by_class: dict[str, int]) -> int:
    total = 0
    for cls, names in fixed_by_class.items():
        cycles = int(cycles_by_class.get(cls, cycles_by_class.get(cls.upper(), 0)))
        total += len(names) * cycles
    return total


def write_fixed_train_manifest(
    run_dir: Path,
    *,
    fixed_by_class: dict[str, list[str]],
    seed: int,
    dataset_path: str,
    class_order: list[str],
    cycles_by_class: dict[str, int],
) -> Path:
    manifest = {
        "fixed_train_instances": fixed_by_class,
        "selection_seed": seed,
        "dataset_path": str(dataset_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "class_order": class_order,
        "cycles_by_class": {str(k).upper(): int(v) for k, v in cycles_by_class.items()},
        "instances_per_class": INSTANCES_PER_CLASS,
        "total_expected_updates": total_expected_updates(fixed_by_class, cycles_by_class),
        "fixed_train_set_hash": fixed_train_set_hash(fixed_by_class),
    }
    path = run_dir / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def load_fixed_train_manifest(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fixed train manifest not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    fixed = data.get("fixed_train_instances") or {}
    for cls in CLASSES:
        names = fixed.get(cls) or []
        if len(names) != INSTANCES_PER_CLASS:
            raise ValueError(
                f"Manifest {cls} must have {INSTANCES_PER_CLASS} instances; got {len(names)}"
            )
    return data


def manifest_fixed_by_class(manifest: dict[str, Any]) -> dict[str, list[str]]:
    raw = manifest["fixed_train_instances"]
    return {cls: list(raw[cls]) for cls in CLASSES}


def build_fixed_mini_datasets(
    full_ds: EVRPTWDataset,
    fixed_names: set[str],
) -> tuple[_SubsetDataset, _SubsetDataset]:
    """Train = exactly the fixed names; eval = all other instances in the full dataset."""
    name_to_full_idx = {name: i for i, name in enumerate(full_ds.names)}
    missing = sorted(fixed_names - set(name_to_full_idx))
    if missing:
        raise ValueError(f"Fixed train instances not found in dataset: {missing}")

    train_indices = [name_to_full_idx[n] for n in sorted(fixed_names)]
    eval_indices = [i for i, name in enumerate(full_ds.names) if name not in fixed_names]
    train_ds = _SubsetDataset(full_ds, sorted(train_indices), None)
    eval_ds = _SubsetDataset(full_ds, eval_indices, None)
    return train_ds, eval_ds


def assert_disjoint_train_eval(fixed_names: set[str], eval_names: list[str]) -> None:
    overlap = fixed_names & set(eval_names)
    if overlap:
        raise RuntimeError(
            f"Train/eval overlap not allowed for fixed_mini: {sorted(overlap)}"
        )


def _cycles_for_class(cls: str, cycles_by_class: dict) -> int:
    raw = cycles_by_class.get(cls) or cycles_by_class.get(cls.upper())
    if raw is None:
        raise ValueError(f"cycles_by_class missing entry for {cls}")
    if isinstance(raw, dict):
        raise ValueError(
            f"fixed_mini cycles_by_class.{cls} must be an integer, not {{small, full}}"
        )
    return int(raw)


def _cycle_shuffle_seed(base_seed: int, cls: str, cycle: int) -> int:
    key = f"{base_seed}:{cls}:{cycle}".encode("utf-8")
    return int(hashlib.md5(key).hexdigest()[:8], 16)


def build_fixed_mini_schedule(
    train_names: list[str],
    name_to_train_idx: dict[str, int],
    fixed_by_class: dict[str, list[str]],
    *,
    class_order: list[str],
    cycles_by_class: dict[str, int],
    shuffle_within_cycle: bool,
    seed: int,
) -> list[dict]:
    """Build training steps: for each class, for each cycle, iterate the fixed 5 instances."""
    schedule: list[dict] = []
    order = [_normalize_class_key(c) for c in class_order]

    for cls in order:
        instance_names = list(fixed_by_class[cls])
        cycles = _cycles_for_class(cls, cycles_by_class)
        for cycle in range(1, cycles + 1):
            names_this_cycle = list(instance_names)
            if shuffle_within_cycle:
                cycle_rng = random.Random(_cycle_shuffle_seed(seed, cls, cycle))
                cycle_rng.shuffle(names_this_cycle)
            for step_in_cycle, name in enumerate(names_this_cycle, start=1):
                schedule.append({
                    "instance_idx": name_to_train_idx[name],
                    "instance_name": name,
                    "instance_class": cls,
                    "cycle": cycle,
                    "step_in_cycle": step_in_cycle,
                    "steps_per_cycle": len(instance_names),
                    "size_tier": "fixed",
                    "tier_label": f"{cls}_fixed",
                })
    return schedule


def filter_eval_groups(
    groups: dict[str, list[int]],
    eval_names: list[str],
    exclude_names: set[str],
) -> dict[str, list[int]]:
    """Remove fixed training instances from eval index lists."""
    return {
        cls: [i for i in idxs if eval_names[i] not in exclude_names]
        for cls, idxs in groups.items()
    }


def log_fixed_train_startup(
    logger,
    fixed_by_class: dict[str, list[str]],
    manifest_path: Path,
    total_updates: int,
) -> None:
    h = fixed_train_set_hash(fixed_by_class)
    logger.info("=== Fixed mini curriculum: 15 training instances ===")
    for cls in CLASSES:
        logger.info(f"  {cls}: {', '.join(fixed_by_class[cls])}")
    logger.info(f"FIXED_TRAIN_SET_HASH={h}")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Total expected actor-critic updates: {total_updates}")
