"""Extract station-visit, route, and solution logs from replayed solutions."""
from __future__ import annotations

from src.hanif.model import charging_rate, expected_price, expected_wait
from src.hanif.replay import customer_sequence, replay_route, replay_solution


def extract_station_visits(
    instance,
    routes: list,
    *,
    meta: dict,
) -> list[dict]:
    """One row per charging visit with heterogeneous attributes and detour."""
    rows = []
    for route_id, item in enumerate(routes):
        nodes, labels = item[0], item[1]
        plan = replay_route(instance, list(nodes), list(labels) if labels is not None else None)
        if not plan.feasible:
            continue
        # Walk with battery to recover arrival SoC / charged energy per stop.
        batt_cap = float(instance.vehicle.battery_capacity)
        energy_m = instance.energy_matrix
        dist_m = instance.distance_matrix
        batt = batt_cap
        ch_idx = 0
        for i in range(1, len(plan.nodes)):
            prev, cur = plan.nodes[i - 1], plan.nodes[i]
            batt -= float(energy_m[prev, cur])
            if not instance.nodes[cur].is_charging_station:
                continue
            nxt = plan.nodes[i + 1] if i + 1 < len(plan.nodes) else plan.nodes[-1]
            detour = float(dist_m[prev, cur]) + float(dist_m[cur, nxt]) - float(dist_m[prev, nxt])
            alpha = 0.0
            if plan.labels and i < len(plan.labels) and plan.labels[i] is not None:
                alpha = float(plan.labels[i])
            missing = max(0.0, batt_cap - batt)
            e_s = alpha * missing
            batt_arr = batt
            batt_dep = min(batt_cap, batt + e_s)
            rate = float(charging_rate(instance, cur))
            wait = float(expected_wait(instance, cur))
            price = float(expected_price(instance, cur))
            ch_time = (e_s / rate) if rate > 1e-12 else 0.0
            expense = e_s * price
            rows.append({
                **meta,
                "route_id": int(route_id),
                "charging_visit_index": int(ch_idx),
                "station_id": int(cur),
                "previous_node": int(prev),
                "next_node": int(nxt),
                "station_detour": detour,
                "arrival_battery": float(batt_arr),
                "charged_energy_es": float(e_s),
                "departure_battery": float(batt_dep),
                "expected_price": price,
                "expected_waiting": wait,
                "charging_rate": rate,
                "charging_time": float(ch_time),
                "charging_expense": float(expense),
                "alpha": float(alpha),
            })
            batt = batt_dep
            ch_idx += 1
    return rows


def extract_route_rows(instance, routes: list, *, meta: dict) -> list[dict]:
    out = []
    for route_id, item in enumerate(routes):
        nodes, labels = item[0], item[1]
        plan = replay_route(instance, list(nodes), list(labels) if labels is not None else None)
        custs = list(customer_sequence(instance, nodes))
        if not custs:
            continue
        out.append({
            **meta,
            "route_id": int(route_id),
            "n_customers": len(custs),
            "feasible": bool(plan.feasible),
            "distance": float(plan.distance),
            "duration": float(plan.duration),
            "charging_visits": int(plan.charging_visits),
            "charged_energy": float(plan.charged_energy),
            "station_waiting": float(plan.station_waiting),
            "charging_time": float(plan.charging_time),
            "charging_expense": float(plan.charging_expense),
            "customer_waiting": float(getattr(plan, "customer_waiting", 0.0) or 0.0),
            "hard_violations": int(plan.hard_violations),
        })
    return out


def solution_row(metrics, *, meta: dict, runtime_s: float, phase1_checksum: str | None = None) -> dict:
    return {
        **meta,
        "feasible": bool(metrics.feasible),
        "served": int(len(metrics.customers)),
        "unserved": int(metrics.unserved),
        "hard_violations": int(metrics.hard_violations),
        "k": int(metrics.used_k),
        "distance": float(metrics.distance),
        "duration": float(metrics.duration),
        "customer_waiting": float(getattr(metrics, "customer_waiting", 0.0) or 0.0),
        "charging_visits": int(metrics.charging_visits),
        "charged_energy": float(metrics.charged_energy),
        "station_waiting": float(metrics.station_waiting),
        "charging_time": float(metrics.charging_time),
        "charging_expense": float(metrics.charging_expense),
        "runtime_s": float(runtime_s),
        "phase1_checksum": phase1_checksum,
        "Cch_over_Ech": (
            float(metrics.charging_expense) / float(metrics.charged_energy)
            if metrics.charged_energy and metrics.charged_energy > 1e-12 else None
        ),
        "Twait_over_Nch": (
            float(metrics.station_waiting) / float(metrics.charging_visits)
            if metrics.charging_visits else None
        ),
        "Tch_over_Ech": (
            float(metrics.charging_time) / float(metrics.charged_energy)
            if metrics.charged_energy and metrics.charged_energy > 1e-12 else None
        ),
    }
