"""Pick the lex-best CARE-FI Schneider route file across local archives."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_GLOBS = (
    "outputs/schneider/{mode}/routes",
    "outputs/schneider_v2/{mode}/routes",
    "outputs/schneider_boost/{mode}/routes",
    "outputs/schneider_boost2/{mode}/routes",
    "outputs/schneider_boost_c2/{mode}/routes",
    "outputs/schneider_hybrid_pilot/{mode}/routes",
    "outputs/schneider_alns_pilot/{mode}/routes",
    "outputs/schneider_v3/{mode}/routes",
    "outputs/schneider_v3_type1/{mode}/routes",
    "outputs/schneider_v3_all/{mode}/routes",
    "outputs/schneider_v4/{mode}/routes",
    "outputs/schneider_v4_rest/{mode}/routes",
    "outputs/schneider_v4_all/{mode}/routes",
    "outputs/schneider_recombine/{mode}/routes",
)


def _key(payload: dict) -> tuple:
    feas = 0 if payload.get("feasible") else 1
    used = int(payload.get("used_k") or 10**6)
    dist = float((payload.get("score") or {}).get("total_distance") or payload.get("distance") or 1e18)
    return (feas, used, dist)


def _deserialize(payload: dict) -> list:
    routes = []
    for r in payload.get("routes") or []:
        routes.append((
            [int(n) for n in r["nodes"]],
            [None if x is None else float(x) for x in (r.get("labels") or [None] * len(r["nodes"]))],
        ))
    return routes


def load_best_incumbent(name: str, mode: str, *, root: Path | None = None) -> tuple[list | None, dict]:
    base = root or ROOT
    best_payload = None
    best_path = None
    for pattern in ARCHIVE_GLOBS:
        folder = base / pattern.format(mode=mode)
        path = folder / f"{name}.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not payload.get("routes"):
            continue
        if best_payload is None or _key(payload) < _key(best_payload):
            best_payload = payload
            best_path = path
    if best_payload is None:
        return None, {}
    return _deserialize(best_payload), {
        "source": str(best_path).replace("\\", "/"),
        "used_k": int(best_payload.get("used_k") or 0),
        "feasible": bool(best_payload.get("feasible")),
        "distance": float((best_payload.get("score") or {}).get("total_distance") or 0.0),
    }


def iter_archive_solutions(name: str, mode: str, *, root: Path | None = None):
    """Yield (path, routes, payload) for every archived file of one instance."""
    base = root or ROOT
    seen: set[str] = set()
    for pattern in ARCHIVE_GLOBS:
        folder = base / pattern.format(mode=mode)
        path = folder / f"{name}.json"
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not payload.get("routes"):
            continue
        yield path, _deserialize(payload), payload
