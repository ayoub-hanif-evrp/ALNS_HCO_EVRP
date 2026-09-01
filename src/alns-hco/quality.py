"""Quality checks on the Hanif master CSV."""
from __future__ import annotations

import csv
from pathlib import Path

from src.common.config import METHOD_ALNS_HCO, TIE_TOL


def check_master(path: Path) -> list[str]:
    errors = []
    if not path.is_file():
        return [f"missing master CSV: {path}"]
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return ["master CSV is empty"]
    for i, r in enumerate(rows, 1):
        if str(r.get("feasible")).lower() not in {"true", "1"}:
            errors.append(f"row {i} {r.get('instance')} {r.get('method')} infeasible")
        if int(float(r.get("unserved") or 0)) != 0:
            errors.append(f"row {i} unserved={r.get('unserved')}")
        if int(float(r.get("hard_violations") or 0)) != 0:
            errors.append(f"row {i} hard_violations={r.get('hard_violations')}")
        if r.get("method") == METHOD_ALNS_HCO:
            k = r.get("k")
            k_star = r.get("k_star")
            if k not in (None, "") and k_star not in (None, ""):
                if int(float(k)) != int(float(k_star)):
                    errors.append(
                        f"row {i} ALNS-HCO K={k} != K*={k_star} ({r.get('instance')} seed={r.get('seed')})"
                    )
            frozen = str(r.get("k_frozen_ok") or "").lower()
            if frozen in {"false", "0"}:
                errors.append(f"row {i} k_frozen_ok is false")
    return errors
