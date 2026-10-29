"""Schneider dataset adapter for ALNS-FR / ALNS-PR / HCO-Detour (no algorithm changes)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.hanif.data.instance_loader import load_instance
from src.hanif.data.schneider_txt_convert import (
    DEFAULT_CONVERTED,
    DEFAULT_RAW,
    LARGE_56,
    convert_one_txt,
    ensure_schneider_converted,
    parse_schneider_txt,
    schneider_family,
)

# One large instance per family for independent routing validation.
SCHNEIDER_REPRESENTATIVE = (
    "c101_21", "c201_21",
    "r101_21", "r201_21",
    "rc101_21", "rc201_21",
)


def schneider_raw_dir() -> Path:
    return Path(DEFAULT_RAW)


def schneider_converted_dir() -> Path:
    return Path(DEFAULT_CONVERTED)


def ensure_representative_converted(*, force: bool = False) -> Path:
    return ensure_schneider_converted(
        names=list(SCHNEIDER_REPRESENTATIVE),
        source_dir=schneider_raw_dir(),
        output_dir=schneider_converted_dir(),
        force=force,
    )


def load_schneider(name: str):
    root = schneider_converted_dir() / name
    if not (root / "evrp_instance.json").is_file():
        txt = schneider_raw_dir() / f"{name}.txt"
        if not txt.is_file():
            raise FileNotFoundError(txt)
        convert_one_txt(txt, schneider_converted_dir())
    return load_instance(root)


def validate_against_source(name: str) -> dict:
    """Compare converted instance against source .txt parameters."""
    txt = schneider_raw_dir() / f"{name}.txt"
    parsed = parse_schneider_txt(txt)
    inst = load_schneider(name)
    meta = json.loads((schneider_converted_dir() / name / "evrp_instance.json").read_text(encoding="utf-8"))
    n_cust = len(parsed["customers"])
    n_st = len(parsed["stations"])
    errors = []
    if int(inst.num_customers) != n_cust:
        errors.append(f"customers {inst.num_customers} != {n_cust}")
    if len(inst.charging_station_indices) != n_st:
        errors.append(f"stations {len(inst.charging_station_indices)} != {n_st}")
    if abs(float(inst.vehicle.battery_capacity) - float(parsed["Q"])) > 1e-9:
        errors.append("battery_capacity mismatch")
    if abs(float(inst.vehicle.load_capacity) - float(parsed["C"])) > 1e-9:
        errors.append("load_capacity mismatch")
    if abs(float(inst.vehicle.energy_consumption_rate) - float(parsed["r"])) > 1e-9:
        errors.append("energy_rate mismatch")
    if abs(float(inst.vehicle.inverse_charging_rate) - float(parsed["g"])) > 1e-9:
        errors.append("inverse_charging_rate mismatch")
    # Matrices finite
    if not np.isfinite(inst.distance_matrix).all():
        errors.append("non-finite distance")
    if not np.isfinite(inst.energy_matrix).all():
        errors.append("non-finite energy")
    # Neutral wait: disabled array OR all zeros (never invent positive wait)
    waits = getattr(inst, "station_wait_expected", None)
    if waits is not None:
        for s in inst.charging_station_indices:
            w = float(waits[s])
            if w == w and abs(w) > 1e-12:  # finite and non-zero
                errors.append(f"non-zero wait at station {s}: {w}")
                break
    # Neutral price expectation ≈ 1
    prices = getattr(inst, "station_price_expected", None)
    if prices is not None:
        for s in inst.charging_station_indices:
            p = float(prices[s])
            if p == p and abs(p - 1.0) > 1e-6:
                errors.append(f"non-neutral price at station {s}: {p}")
                break
    return {
        "instance": name,
        "family": schneider_family(name),
        "num_customers": n_cust,
        "num_stations": n_st,
        "Q": parsed["Q"],
        "C": parsed["C"],
        "r": parsed["r"],
        "g": parsed["g"],
        "v": parsed["v"],
        "ok": not errors,
        "errors": errors,
        "heterogeneous_stations": False,
        "metadata_family": meta["metadata"].get("family"),
    }


def characteristics_table(names: list[str] | None = None) -> list[dict]:
    names = list(names or SCHNEIDER_REPRESENTATIVE)
    ensure_schneider_converted(names=names, source_dir=schneider_raw_dir(), output_dir=schneider_converted_dir())
    return [validate_against_source(n) for n in names]
