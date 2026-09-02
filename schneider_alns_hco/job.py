"""One (instance, seed) ALNS-HCO solve. Written here so ProcessPool can pickle it."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT))

from config import PHASE1_S, PHASE2_S, ROUTES
from load import family_of, load_schneider

from src.alns_hco.solver import solve_alns_hco  # noqa: E402


def _dump_routes(path, routes) -> None:
    payload = []
    for nodes, labels in routes:
        payload.append({
            "nodes": [int(n) for n in nodes],
            "labels": [None if lab is None else float(lab) for lab in (labels or [])],
        })
    path.write_text(json.dumps(payload), encoding="utf-8")


def solve_job(job: dict) -> dict:
    name = job["name"]
    seed = int(job["seed"])
    p1 = float(job.get("phase1_s", PHASE1_S))
    p2 = float(job.get("phase2_s", PHASE2_S))
    inst = load_schneider(name)
    result = solve_alns_hco(inst, mode="alns_hco", phase1_s=p1, phase2_s=p2, seed=seed)
    m = result.metrics
    p1s = result.stages.get("phase1") or {}
    p2s = result.stages.get("phase2") or {}
    row = {
        "instance": name,
        "family": family_of(name),
        "seed": seed,
        "feasible": bool(m.feasible),
        "served": int(len(m.customers)),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
        "k": int(m.used_k),
        "distance": float(m.distance),
        "duration": float(m.duration),
        "charging_visits": int(m.charging_visits),
        "charged_energy": float(m.charged_energy),
        "charging_expense": float(m.charging_expense),
        "runtime_s": float(result.runtime_s),
        "phase1_s": p1s.get("runtime_s"),
        "phase2_s": p2s.get("runtime_s"),
        "phase1_k": p1s.get("k"),
        "phase1_distance": p1s.get("distance"),
        "k_star": result.validation.get("k_star"),
        "k_frozen_ok": result.validation.get("k_frozen_ok"),
    }
    if job.get("save_routes"):
        ROUTES.mkdir(parents=True, exist_ok=True)
        _dump_routes(ROUTES / f"{name}_seed{seed}.json", result.routes)
    return row
