"""Hanif-only dataset adapter: discovery, compatibility report, deterministic splits."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.data.hanif_convert import (
    DEFAULT_OUTPUT,
    DEFAULT_SOURCE,
    SOURCE_VARIANT,
)
from src.data.instance_class import CLASSES, instance_class
from src.data.instance_loader import list_instance_dirs, load_instance, read_instance_summary
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

SUPPORTED_STATION_ATTRS = (
    "charging_rate",
    "price_mu",
    "price_sigma",
    "wait_mu",
    "density_score",
)
# Stochastic wait is not sampled; masks/env use E[wait]=wait_mu only.
DOCUMENTED_NON_STOCHASTIC = ("wait_sigma",)


class HanifDatasetError(RuntimeError):
    """Fatal Hanif dataset / semantics failure — stop before training."""


def is_hanif_config(config: dict) -> bool:
    ds = config.get("dataset") or {}
    name = str(ds.get("name", "")).lower()
    path = str(ds.get("path") or ds.get("root") or "").lower()
    return name == "hanif" or "hanif" in path or "converted_hanif" in path


def resolve_hanif_paths(config: dict) -> dict[str, Path]:
    ds = config.get("dataset") or {}
    root = Path(ds.get("path") or ds.get("root") or DEFAULT_OUTPUT)
    source = Path(ds.get("source_txt") or ds.get("source_dir") or DEFAULT_SOURCE)
    return {"root": root, "source": source}


def ensure_hanif_dataset(config: dict) -> Path:
    """Convert the configured Hanif TW variant (A original / B relaxed)."""
    from src.data.hanif_convert import ensure_hanif_variant_converted

    if not is_hanif_config(config):
        raise HanifDatasetError("dataset.name must be 'hanif' for this study")
    ds = config.get("dataset") or {}
    if ds.get("use_only_this_dataset", True) is False:
        raise HanifDatasetError("dataset.use_only_this_dataset must be true")
    paths = resolve_hanif_paths(config)
    if "schneider" in str(paths["root"]).lower():
        raise HanifDatasetError("Refusing Schneider path in Hanif-only study")
    variant = str(ds.get("tw_variant") or ds.get("variant") or "A")
    root_l = str(paths["root"]).lower()
    if "relaxed" in root_l or "variant_b" in root_l:
        variant = "B"
    elif "original" in root_l or "variant_a" in root_l:
        variant = "A"
    return ensure_hanif_variant_converted(
        variant,
        source_dir=paths["source"] if ds.get("source_txt") or ds.get("source_dir") else None,
        output_dir=paths["root"],
        force=bool(ds.get("force_reconvert", False)),
    )


def discover_hanif_instances(root: Path) -> list[str]:
    return [d.name for d in list_instance_dirs(root)]


def group_by_class(names: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {c: [] for c in CLASSES}
    for name in sorted(names):
        groups[instance_class(name)].append(name)
    for c in CLASSES:
        groups[c] = sorted(groups[c])
    return groups


def deterministic_class_split(
    names_sorted: list[str],
    *,
    train_n: int,
    val_n: int,
    test_n: int,
) -> dict[str, list[str]]:
    """Filename-sorted split; no shuffle. Raises if not enough instances."""
    need = train_n + val_n + test_n
    if len(names_sorted) < need:
        raise HanifDatasetError(
            f"Need {need} instances for split, found {len(names_sorted)}: {names_sorted}"
        )
    train = names_sorted[:train_n]
    val = names_sorted[train_n:train_n + val_n]
    test = names_sorted[train_n + val_n:train_n + val_n + test_n]
    return {"train": train, "validation": val, "test": test}


def resolve_hanif_split(config: dict) -> tuple[list[str], list[str], list[str], dict]:
    """Return (train, validation, test, meta) with per-class deterministic splits."""
    root = ensure_hanif_dataset(config)
    ds = config.get("dataset") or {}
    classes = [c.strip().upper() for c in (ds.get("classes") or list(CLASSES))]
    for c in classes:
        if c not in CLASSES:
            raise HanifDatasetError(f"Unknown class {c}")

    pilot = ds.get("pilot") or {}
    pilot_enabled = bool(pilot.get("enabled", True))
    train_n = int(pilot.get("train_per_class", 2 if pilot_enabled else 14))
    val_n = int(pilot.get("val_per_class", 1 if pilot_enabled else 3))
    test_n = int(pilot.get("test_per_class", 1 if pilot_enabled else 3))

    all_names = discover_hanif_instances(root)
    by_class = group_by_class(all_names)
    split_by_class: dict[str, dict[str, list[str]]] = {}
    train: list[str] = []
    validation: list[str] = []
    test: list[str] = []
    for c in classes:
        pool = by_class.get(c) or []
        split_by_class[c] = deterministic_class_split(
            pool, train_n=train_n, val_n=val_n, test_n=test_n,
        )
        train.extend(split_by_class[c]["train"])
        validation.extend(split_by_class[c]["validation"])
        test.extend(split_by_class[c]["test"])

    train = sorted(train)
    validation = sorted(validation)
    test = sorted(test)
    overlap = (set(train) & set(validation)) | (set(train) & set(test)) | (set(validation) & set(test))
    if overlap:
        raise HanifDatasetError(f"Split leakage detected: {sorted(overlap)}")

    meta = {
        "training_phase": "hanif_pilot" if pilot_enabled else "hanif_full",
        "dataset_name": "hanif",
        "dataset_root": str(root).replace("\\", "/"),
        "source_variant": SOURCE_VARIANT,
        "time_windows_modified": False,
        "pilot_enabled": pilot_enabled,
        "train_per_class": train_n,
        "val_per_class": val_n,
        "test_per_class": test_n,
        "classes": classes,
        "split_by_class": split_by_class,
        "train_instances": train,
        "validation_instances": validation,
        "eval_instances": test,
        "test_instances": test,
        "num_train": len(train),
        "num_validation": len(validation),
        "num_eval": len(test),
        "num_test": len(test),
    }
    return train, validation, test, meta


def _inspect_instance(root: Path, name: str, *, full_load: bool = False) -> dict[str, Any]:
    inst_dir = root / name
    summary = read_instance_summary(inst_dir) or {}
    data = json.loads((inst_dir / "evrp_instance.json").read_text(encoding="utf-8"))
    meta = data.get("metadata") or {}
    variant = str(meta.get("source_variant") or summary.get("variant") or "")
    if "relaxed" in variant.lower():
        raise HanifDatasetError(
            f"Instance {name} has relaxed TW source_variant={variant}. "
            "Refuse. Re-convert from variant_A_original_TW."
        )
    if bool(meta.get("time_windows_modified", False)) or bool(summary.get("time_windows_modified", False)):
        raise HanifDatasetError(f"Instance {name}: time windows were modified")

    stations_path = inst_dir / "stations.csv"
    detected_attrs: list[str] = []
    unsupported: list[str] = []
    if stations_path.is_file():
        header = stations_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        detected_attrs = [h.strip() for h in header]
        known = set(SUPPORTED_STATION_ATTRS) | set(DOCUMENTED_NON_STOCHASTIC) | {
            "node_id", "hanif_station_id", "expected_price_per_energy", "station_id",
        }
        for h in detected_attrs:
            if h and h not in known:
                unsupported.append(h)
    else:
        raise HanifDatasetError(f"{name}: stations.csv missing (Hanif requires station attrs)")

    if unsupported:
        raise HanifDatasetError(
            f"{name}: unsupported station attributes {unsupported}. "
            f"Supported: {list(SUPPORTED_STATION_ATTRS)}; "
            f"non-stochastic documented: {list(DOCUMENTED_NON_STOCHASTIC)}"
        )

    num_customers = int(summary.get("num_customers") or meta.get("num_customers") or 0)
    num_stations = int(
        summary.get("num_stations")
        or meta.get("num_charging_stations")
        or 0
    )
    load_capacity = float(
        summary.get("load_capacity")
        or (data.get("vehicle_parameters") or {}).get("load_capacity")
        or 0.0
    )
    battery_capacity = float(
        summary.get("battery_capacity")
        or (data.get("vehicle_parameters") or {}).get("battery_capacity")
        or 0.0
    )
    energy_rate = float(
        summary.get("energy_consumption_rate")
        or (data.get("vehicle_parameters") or {}).get("energy_consumption_rate")
        or 0.0
    )

    if full_load:
        # Full parse + matrix load (expensive); use for smoke checks only.
        inst = load_instance(inst_dir, load_capacity=None)
        num_customers = inst.num_customers
        num_stations = inst.num_charging_stations
        load_capacity = float(inst.vehicle.load_capacity)
        battery_capacity = float(inst.vehicle.battery_capacity)
        energy_rate = float(inst.vehicle.energy_consumption_rate)

    return {
        "instance_name": name,
        "class": instance_class(name),
        "num_customers": num_customers,
        "num_stations": num_stations,
        "load_capacity": load_capacity,
        "battery_capacity": battery_capacity,
        "energy_consumption_rate": energy_rate,
        "source_variant": variant or SOURCE_VARIANT,
        "time_windows_modified": False,
        "detected_station_attributes": detected_attrs,
        "charging_model": "per_station_rate_plus_expected_wait_mu",
        "station_wait_model": "expected_wait_mu",
        "wait_sigma_policy": "not_sampled_use_wait_mu_mean_only",
        "density_score_policy": "dynamic_candidate_feature",
    }


def build_hanif_dataset_report(
    config: dict,
    *,
    run_dir: Path | None = None,
    split_meta: dict | None = None,
) -> dict[str, Any]:
    root = ensure_hanif_dataset(config)
    names = discover_hanif_instances(root)
    by_class = group_by_class(names)
    if split_meta is None:
        _, _, _, split_meta = resolve_hanif_split(config)

    inspections = []
    for i, name in enumerate(names):
        inspections.append(_inspect_instance(root, name, full_load=(i == 0)))
    # Ensure matrices.npz exists for every instance without full decode.
    for name in names:
        npz = root / name / "matrices.npz"
        if not npz.is_file():
            raise HanifDatasetError(f"{name}: matrices.npz missing")

    customers = sorted({r["num_customers"] for r in inspections})
    stations = sorted({r["num_stations"] for r in inspections})
    loads = sorted({r["load_capacity"] for r in inspections})
    batteries = sorted({r["battery_capacity"] for r in inspections})

    report = {
        "dataset_root": str(root).replace("\\", "/"),
        "source_variant": SOURCE_VARIANT,
        "number_of_files": len(names),
        "number_of_C_instances": len(by_class["C"]),
        "number_of_R_instances": len(by_class["R"]),
        "number_of_RC_instances": len(by_class["RC"]),
        "customers_per_instance": customers,
        "stations_per_instance": stations,
        "detected_vehicle_capacity": loads,
        "detected_battery_capacity": batteries,
        "detected_charging_model": "per_station_rate_plus_expected_wait_mu",
        "detected_station_attributes": list(SUPPORTED_STATION_ATTRS) + list(DOCUMENTED_NON_STOCHASTIC),
        "unsupported_attributes": [],
        "wait_sigma_note": "Stochastic wait_sigma is not sampled; env/masks use E[wait]=wait_mu.",
        "density_score_note": "density_score is a dynamic candidate feature (not physics).",
        "time_windows_modified": False,
        "confirmation_no_time_windows_modified": True,
        "train_validation_test_split": {
            "train": split_meta.get("train_instances"),
            "validation": split_meta.get("validation_instances"),
            "test": split_meta.get("test_instances"),
            "split_by_class": split_meta.get("split_by_class"),
            "pilot_enabled": split_meta.get("pilot_enabled"),
        },
        "instances": inspections,
    }

    if run_dir is not None:
        ds_dir = Path(run_dir) / "dataset"
        ds_dir.mkdir(parents=True, exist_ok=True)
        report_path = ds_dir / "hanif_dataset_report.json"
        report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        for cls in CLASSES:
            part = (split_meta.get("split_by_class") or {}).get(cls) or {}
            (ds_dir / f"split_{cls}.json").write_text(
                json.dumps(part, indent=2), encoding="utf-8",
            )
        (ds_dir / "split_all.json").write_text(
            json.dumps(
                {
                    "train": split_meta.get("train_instances"),
                    "validation": split_meta.get("validation_instances"),
                    "test": split_meta.get("test_instances"),
                    "split_by_class": split_meta.get("split_by_class"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER.info("Hanif dataset report -> %s", report_path)
    return report


def save_split_into_config(config: dict, train: list[str], validation: list[str], test: list[str], meta: dict) -> dict:
    """Mutate config dataset keys so the rest of the pipeline sees Hanif splits."""
    ds = config.setdefault("dataset", {})
    ds["name"] = "hanif"
    ds["path"] = str(meta.get("dataset_root") or resolve_hanif_paths(config)["root"])
    ds["train_instances"] = list(train)
    ds["validation_instances"] = list(validation)
    ds["eval_instances"] = list(test)
    ds["test_instances"] = list(test)
    ds["holdout_by_class"] = {
        cls: list(((meta.get("split_by_class") or {}).get(cls) or {}).get("test") or [])
        for cls in (ds.get("classes") or list(CLASSES))
    }
    ds["class_filter"] = list(ds.get("classes") or list(CLASSES))
    ds["training_phase"] = "hanif"
    ds["train_size_filter"] = []
    ds["eval_size_filter"] = []
    ds["instance_suffix"] = ""
    ds["auto_discover_rc21"] = False
    ds["shuffle_instances"] = False
    # Prevent Schneider capacity override.
    problem = config.setdefault("problem", {})
    problem["load_capacity"] = None
    return config


__all__ = [
    "HanifDatasetError",
    "build_hanif_dataset_report",
    "deterministic_class_split",
    "ensure_hanif_dataset",
    "is_hanif_config",
    "resolve_hanif_split",
    "save_split_into_config",
]
