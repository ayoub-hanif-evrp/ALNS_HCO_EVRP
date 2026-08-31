"""Per-cycle aggregation of epoch-level training metrics."""
from __future__ import annotations

from typing import Any

import numpy as np


def _mean(rows: list[dict], key: str, default: float = 0.0) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] is not None]
    return float(np.mean(vals)) if vals else default


def dedupe_metric_rows(rows: list[dict], key: str = "epoch") -> list[dict]:
    """Keep the last row per unique key (defensive against duplicate logging)."""
    if not rows:
        return rows
    if key not in rows[0]:
        return rows
    by_key: dict[Any, dict] = {}
    for row in rows:
        by_key[row[key]] = row
    return [by_key[k] for k in sorted(by_key.keys())]


def aggregate_cycle(rows: list[dict], *, cycle_id: int) -> dict[str, Any]:
    if not rows:
        return {"cycle_id": cycle_id}
    out = {
        "cycle_id": cycle_id,
        "cycle": cycle_id,
        "cycle_epochs": len(rows),
        "mean_objective": _mean(rows, "macro_objective") if any("macro_objective" in r for r in rows) else _mean(rows, "cost"),
        "mean_return": _mean(rows, "reward"),
        "feasibility_rate": _mean(rows, "macro_feasibility") if any("macro_feasibility" in r for r in rows) else _mean(rows, "feasibility_rate"),
        "mean_served_ratio": _mean(rows, "macro_served_rate") if any("macro_served_rate" in r for r in rows) else _mean(rows, "served_rate"),
        "mean_unserved_customers": _mean(rows, "unserved_customers"),
        "mean_battery_violations": _mean(rows, "avg_battery_violation"),
        "mean_time_window_violations": _mean(rows, "avg_lateness"),
        "mean_capacity_violations": _mean(rows, "avg_capacity_violation"),
        "mean_depot_violations": _mean(rows, "avg_depot_violation"),
        "mean_used_evs": _mean(rows, "avg_used_evs"),
        "mean_distance": _mean(rows, "avg_distance"),
        "mean_duration": _mean(rows, "avg_duration"),
        "mean_charging_time": _mean(rows, "avg_charging_time"),
        "mean_waiting_time": _mean(rows, "avg_waiting_time"),
        "mean_actor_loss": _mean(rows, "actor_loss"),
        "mean_critic_loss": _mean(rows, "critic_loss"),
        "mean_entropy": _mean(rows, "entropy"),
        "mean_explained_variance": _mean(rows, "critic_explained_variance"),
        "mean_objective_mismatch": _mean(rows, "objective_mismatch"),
        "mean_relative_objective_mismatch": _mean(rows, "relative_objective_mismatch"),
        "mean_feasible_action_ratio": _mean(rows, "feasible_action_ratio"),
        "mean_masked_by_time_window": _mean(rows, "masked_by_time_window_count"),
        "mean_newly_expired_customers": _mean(rows, "newly_expired_customers"),
        "mean_served_customer_slack": _mean(rows, "served_customer_slack"),
        "mean_unnecessary_station_actions": _mean(rows, "unnecessary_station_actions"),
        "mean_feasible_customer_ratio": _mean(rows, "feasible_customer_ratio"),
    }
    for cls in ("C", "R", "RC"):
        out[f"mean_objective_{cls}"] = _mean(rows, f"objective_{cls}")
        out[f"mean_served_rate_{cls}"] = _mean(rows, f"served_rate_{cls}")
        out[f"mean_feasibility_{cls}"] = _mean(rows, f"feasibility_{cls}")
        out[f"mean_serviceability_{cls}"] = _mean(rows, f"serviceability_{cls}")
    out["macro_objective"] = out["mean_objective"]
    out["macro_served_rate"] = out["mean_served_ratio"]
    out["macro_feasibility"] = out["feasibility_rate"]
    return out
