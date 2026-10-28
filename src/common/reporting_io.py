"""CSV export helpers that preserve reviewer reporting fields."""
from __future__ import annotations

import csv
from pathlib import Path

from src.common.config import MASTER_FIELDS, REPORTING_ONLY_FIELDS


def master_fieldnames(rows: list[dict] | None = None) -> list[str]:
    """Stable field order: MASTER_FIELDS first, then any extras seen in rows."""
    fields = list(MASTER_FIELDS)
    seen = set(fields)
    if rows:
        for row in rows:
            for k in row.keys():
                if k in {"station_visits", "route_rows", "hco_activity", "routes"}:
                    continue
                if k not in seen:
                    fields.append(k)
                    seen.add(k)
    return fields


def flatten_solution_row(row: dict) -> dict:
    """Drop nested payloads that should not enter the master CSV."""
    skip = {"station_visits", "route_rows", "hco_activity", "routes", "energy_routes"}
    return {k: v for k, v in row.items() if k not in skip}


def write_master_csv(path: Path, rows: list[dict], *, append: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = [flatten_solution_row(r) for r in rows]
    if not flat and not append:
        path.write_text("", encoding="utf-8")
        return
    fields = master_fieldnames(flat if flat else None)
    # Ensure reporting-only columns exist even for empty append bootstrap
    for f in REPORTING_ONLY_FIELDS:
        if f not in fields:
            fields.append(f)
    mode = "a" if append and path.is_file() and path.stat().st_size > 0 else "w"
    write_header = mode == "w"
    with path.open(mode, newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if write_header:
            w.writeheader()
        for r in flat:
            w.writerow({k: r.get(k) for k in fields})


def read_master_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def assert_reporting_fields_present(path: Path, required: tuple[str, ...] | None = None) -> list[str]:
    """Return missing required columns (empty list if OK)."""
    required = required or REPORTING_ONLY_FIELDS
    rows = read_master_csv(path)
    if not rows:
        return ["empty_or_missing_csv"]
    have = set(rows[0].keys())
    return [c for c in required if c not in have]
