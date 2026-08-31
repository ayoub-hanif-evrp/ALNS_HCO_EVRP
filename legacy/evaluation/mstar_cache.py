"""Persist / load expert lexicographic m* (min feasible fleet) per instance."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def mstar_cache_path(config: dict) -> Path:
    demo = config.get("demonstrations") or {}
    fleet = config.get("fleet_search") or {}
    raw = demo.get("mstar_cache") or fleet.get("mstar_cache") or "outputs/demonstrations/mstar.json"
    return Path(str(raw))


def load_mstar_map(config: dict) -> dict[str, int]:
    path = mstar_cache_path(config)
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(raw, dict) and isinstance(raw.get("mstar"), dict):
        raw = raw["mstar"]
    out: dict[str, int] = {}
    if isinstance(raw, dict):
        for k, v in raw.items():
            try:
                out[str(k)] = int(v)
            except (TypeError, ValueError):
                continue
    return out


def save_mstar_map(
    config: dict,
    mstar: dict[str, int],
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    path = mstar_cache_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mstar": {str(k): int(v) for k, v in sorted(mstar.items())},
        "meta": meta or {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
