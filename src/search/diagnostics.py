"""Generic failure attribution for unserved customers (no instance-name branches)."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from src.search.route_replay import SolutionReplayResult, replay_solution
from src.search.repair import best_insertion, _clone_routes


def attribute_unserved(
    instance,
    routes: list[tuple[list[int], list[float | None]]],
    *,
    config: dict | None = None,
    m: int | None = None,
) -> list[dict[str, Any]]:
    """For each unserved customer, record insertion recoverability and blockers."""
    res = replay_solution(instance, routes, config=config, m=m)
    served = set(res.served_customers)
    customers = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    rows: list[dict[str, Any]] = []
    for j in customers:
        if j in served:
            continue
        node = instance.nodes[j]
        cand = best_insertion(instance, routes, j, config=config, m=m, station_candidates=5)
        rows.append({
            "instance": getattr(instance, "instance_name", ""),
            "customer": int(j),
            "ready_time": float(node.ready_time),
            "due_time": float(node.due_time),
            "demand": float(node.demand),
            "service_time": float(node.service_time),
            "recoverable_by_insertion": cand is not None,
            "insertion_route": None if cand is None else cand.route_index,
            "insertion_position": None if cand is None else cand.position,
            "needs_station_before": None if cand is None else cand.station_before,
            "needs_station_after": None if cand is None else cand.station_after,
            "current_unserved": int(res.unserved_customers),
            "current_feasible": bool(res.feasible),
            "hard_violations": int(res.hard_violation_count),
        })
    return rows


def write_failure_attribution(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("customer\n", encoding="utf-8")
        return path
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path
