r"""Serialize and load shared Phase-I solutions X^{PR}_{15}."""
from __future__ import annotations

import json
from pathlib import Path

from src.alns_hco.checksum import customer_checksum, customer_routes_payload
from src.alns_hco.phase1 import solve_phase1
from src.common.config import PHASE1_S
from src.hanif.loader import load_hanif
from src.hanif.replay import replay_solution


def shared_phase1_path(root: Path, instance: str, tw: str, seed: int) -> Path:
    return root / tw / instance / f"seed{int(seed)}.json"


def _metrics_dict(m) -> dict:
    return {
        "feasible": bool(m.feasible),
        "served": int(len(m.customers)),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
        "k": int(m.used_k),
        "distance": float(m.distance),
        "duration": float(m.duration),
        "customer_waiting": float(getattr(m, "customer_waiting", 0.0) or 0.0),
        "charging_visits": int(m.charging_visits),
        "charged_energy": float(m.charged_energy),
        "station_waiting": float(m.station_waiting),
        "charging_time": float(m.charging_time),
        "charging_expense": float(m.charging_expense),
    }


def serialize_routes(routes: list) -> list[dict]:
    out = []
    for item in routes:
        nodes, labels = item[0], item[1]
        out.append({
            "nodes": [int(n) for n in nodes],
            "labels": [None if x is None else float(x) for x in labels],
        })
    return out


def deserialize_routes(blob: list[dict]) -> list:
    return [
        ([int(n) for n in r["nodes"]], [None if x is None else float(x) for x in r["labels"]])
        for r in blob
    ]


def build_shared_record(instance, *, instance_name: str, tw: str, seed: int, phase1: dict) -> dict:
    routes = phase1["routes"]
    m = phase1["metrics"]
    return {
        "instance": instance_name,
        "family": instance_name.split("_")[0],
        "tw": tw,
        "seed": int(seed),
        "phase1_s": float(phase1["runtime_s"]),
        "budget_s": float(PHASE1_S),
        "k_star": int(phase1["k_star"]),
        "customer_checksum": customer_checksum(instance, routes),
        "customer_routes": customer_routes_payload(instance, routes),
        "routes": serialize_routes(routes),
        "metrics": _metrics_dict(m),
        "iterations": int(phase1.get("iterations") or 0),
        "accepted": int(phase1.get("accepted") or 0),
    }


def save_shared_phase1(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_shared_phase1(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_shared_phase1(
    root: Path,
    *,
    instance_name: str,
    tw: str,
    seed: int,
    time_limit_s: float | None = None,
    force: bool = False,
) -> dict:
    """Run Phase I once (or load) and return the serialized record + live routes."""
    path = shared_phase1_path(root, instance_name, tw, seed)
    if path.is_file() and not force:
        rec = load_shared_phase1(path)
        inst = load_hanif(instance_name, tw)
        routes = deserialize_routes(rec["routes"])
        # Re-verify checksum against live instance
        chk = customer_checksum(inst, routes)
        if chk != rec["customer_checksum"]:
            raise AssertionError(f"stored checksum mismatch for {path}")
        m = replay_solution(inst, routes)
        return {"path": str(path), "record": rec, "routes": routes, "metrics": m, "instance": inst}

    inst = load_hanif(instance_name, tw)
    phase1 = solve_phase1(inst, time_limit_s=float(time_limit_s if time_limit_s is not None else PHASE1_S), seed=int(seed))
    rec = build_shared_record(inst, instance_name=instance_name, tw=tw, seed=seed, phase1=phase1)
    save_shared_phase1(path, rec)
    return {
        "path": str(path),
        "record": rec,
        "routes": phase1["routes"],
        "metrics": phase1["metrics"],
        "instance": inst,
    }
