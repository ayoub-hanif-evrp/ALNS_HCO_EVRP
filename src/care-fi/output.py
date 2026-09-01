"""Canonical result records written by both benchmark runners."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = (
    "benchmark",
    "instance",
    "solver_version",
    "configuration",
    "runtime_s",
    "battery_capacity",
    "energy_consumption_rate",
    "used_k",
    "distance",
    "feasible",
    "termination_reason",
)

HANIF_STAGE2_FIELDS = (
    "duration",
    "charging_visits",
    "charged_energy",
    "station_waiting",
    "charging_time",
    "charging_expense",
    "phi",
)


def validate_record(row: dict[str, Any], *, stage2: bool = False) -> None:
    missing = [k for k in REQUIRED_FIELDS if k not in row]
    if stage2:
        missing.extend(k for k in HANIF_STAGE2_FIELDS if k not in row)
    if missing:
        raise ValueError(f"result record missing fields: {missing}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def serialize_routes(routes: list) -> list[dict[str, Any]]:
    out = []
    for nodes, labels in routes:
        labs = list(labels) if labels else [None] * len(nodes)
        out.append({
            "nodes": [int(n) for n in nodes],
            "labels": [None if x is None else float(x) for x in labs],
        })
    return out
