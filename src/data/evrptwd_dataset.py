"""EVRP-TW-D dataset adapter: used_dataset_instances train/test (no validation)."""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.logging_utils import get_logger

LOGGER = get_logger()

DEFAULT_SOURCE = Path("dataset/EVRPTWD/unanchored/Cus_100/solomon")
DEFAULT_USED = Path("dataset/EVRPTWD/used_dataset_instances")
DEFAULT_SPLIT_DIR = Path("dataset/EVRPTWD/splits")
DEFAULT_MANIFEST = DEFAULT_USED / "split_manifest.json"
NAME_RE = re.compile(
    r"^solomon_dataset_(?P<id>\d+)_(?P<family>R|C|RC)_(?P<tw>wide|narrow)_(?P<ts>\d+)$"
)


class EvrptwdDatasetError(RuntimeError):
    """Fatal EVRP-TW-D dataset / split failure."""


def is_evrptwd_config(config: dict) -> bool:
    ds = config.get("dataset") or {}
    name = str(ds.get("name", "")).lower()
    path = str(ds.get("path") or ds.get("root") or "").lower()
    phase = str(ds.get("training_phase") or "").lower()
    return name in {"evrptwd", "evrp-tw-d", "evrptw-d"} or "evrptwd" in path or phase in {
        "evrptwd", "evrp_tw_d",
    }


def parse_evrptwd_name(stem: str) -> dict | None:
    m = NAME_RE.match(stem)
    if not m:
        return None
    return {
        "instance_id": int(m.group("id")),
        "family": m.group("family"),
        "tw": m.group("tw"),
        "stem": stem,
    }


def _load_manifest(path: Path) -> dict:
    if not path.is_file():
        raise EvrptwdDatasetError(
            f"Missing used-dataset manifest: {path}. "
            "Run: python scripts/build_evrptwd_easy_wide100_split.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_evrptwd_split(config: dict) -> tuple[list[str], list[str], list[str], dict]:
    """Return (train, validation=[], test, meta). Prefers used_dataset_instances manifest."""
    ds = config.get("dataset") or {}
    used_root = Path(ds.get("path") or ds.get("root") or DEFAULT_USED)
    manifest_path = Path(ds.get("manifest") or (used_root / "split_manifest.json"))
    if not manifest_path.is_file() and DEFAULT_MANIFEST.is_file():
        manifest_path = DEFAULT_MANIFEST
        used_root = DEFAULT_USED

    payload = _load_manifest(manifest_path)
    train_names = list(payload.get("train_instances") or [])
    test_names = list(payload.get("test_instances") or [])
    if not train_names or not test_names:
        raise EvrptwdDatasetError(f"Manifest missing train/test lists: {manifest_path}")

    overlap = set(train_names) & set(test_names)
    if overlap:
        raise EvrptwdDatasetError(f"train/test overlap: {sorted(overlap)[:5]}")

    missing = []
    for name in train_names + test_names:
        if not (used_root / name / "evrp_instance.json").is_file():
            missing.append(name)
    if missing:
        raise EvrptwdDatasetError(
            f"{len(missing)} converted instances missing under {used_root} "
            f"(e.g. {missing[:3]})"
        )

    n_train = int(ds.get("train_size", len(train_names)))
    n_test = int(ds.get("test_size", len(test_names)))
    if len(train_names) != n_train or len(test_names) != n_test:
        LOGGER.warning(
            "Config train_size/test_size (%d/%d) != manifest (%d/%d) — using manifest",
            n_train, n_test, len(train_names), len(test_names),
        )

    difficulty = str(payload.get("difficulty") or ds.get("difficulty") or "easy_wide")
    meta = {
        "training_phase": "evrptwd",
        "dataset": "evrptwd",
        "family": "RC",
        "scale": 100,
        "difficulty": difficulty,
        "split": f"unanchored_cus100_rc_{difficulty}",
        "manifest": str(manifest_path),
        "train_instances": train_names,
        "validation_instances": [],
        "test_instances": test_names,
        "eval_instances": test_names,
        "num_train": len(train_names),
        "num_validation": 0,
        "num_test": len(test_names),
        "converted_root": str(used_root),
        "source_txt": str(payload.get("source_solomon") or DEFAULT_SOURCE),
        "no_validation": True,
        "train_tw": payload.get("train_tw"),
        "test_tw": payload.get("test_tw"),
        "train_ease_mean": payload.get("train_ease_mean"),
        "test_ease_mean": payload.get("test_ease_mean"),
        "selection": payload.get("selection"),
    }
    LOGGER.info(
        "EVRPTWD %s split | train=%d test=%d | ease_train=%.3f ease_test=%.3f | %s",
        difficulty,
        len(train_names),
        len(test_names),
        float(meta.get("train_ease_mean") or 0.0),
        float(meta.get("test_ease_mean") or 0.0),
        used_root,
    )
    return train_names, [], test_names, meta


def save_split_into_config(
    config: dict,
    train: list[str],
    validation: list[str],
    test: list[str],
    meta: dict,
) -> None:
    ds = config.setdefault("dataset", {})
    ds["train_instances"] = list(train)
    ds["validation_instances"] = list(validation)
    ds["test_instances"] = list(test)
    ds["eval_instances"] = list(test)
    ds["path"] = str(meta.get("converted_root") or ds.get("path") or DEFAULT_USED)
    ds["root"] = ds["path"]
    ds["train_size_filter"] = []
    ds["eval_size_filter"] = []
    ds["training_phase"] = "evrptwd"
    ds["train_size"] = len(train)
    ds["test_size"] = len(test)
    ds["difficulty"] = meta.get("difficulty") or ds.get("difficulty")
    config["evrptwd_split_meta"] = meta


def ensure_evrptwd_dataset(config: dict) -> tuple[dict, dict]:
    """Load used_dataset_instances split; return (report, split_meta)."""
    if not is_evrptwd_config(config):
        raise EvrptwdDatasetError("dataset.name must be evrptwd for this study")
    train, validation, test, meta = resolve_evrptwd_split(config)
    save_split_into_config(config, train, validation, test, meta)
    report = {
        "dataset": "evrptwd",
        "family": "RC",
        "scale": 100,
        "difficulty": meta.get("difficulty"),
        "source": meta.get("source_txt"),
        "converted_root": meta.get("converted_root"),
        "manifest": meta.get("manifest"),
        "num_train": len(train),
        "num_validation": 0,
        "num_test": len(test),
        "no_validation": True,
        "train_tw": meta.get("train_tw"),
        "test_tw": meta.get("test_tw"),
        "train_ease_mean": meta.get("train_ease_mean"),
        "test_ease_mean": meta.get("test_ease_mean"),
        "selection": meta.get("selection"),
    }
    LOGGER.info(
        "EVRPTWD ready | train=%d val=0 test=%d | difficulty=%s | path=%s",
        len(train), len(test), meta.get("difficulty"), meta.get("converted_root"),
    )
    return report, meta
