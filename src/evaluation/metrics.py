"""Per-instance evaluation metrics and route export."""
from __future__ import annotations

from src.env.evrptw_env import EVRPTWEnv


def build_result_row(
    instance,
    env: EVRPTWEnv,
    objective: float,
    selected_m: int,
    m_min: int,
    m_max: int,
    runtime_seconds: float,
    method: str,
) -> dict:
    """Assemble the flat metrics row written to results.csv."""
    summary = env.build_summary()
    station_visits = sum(
        1
        for ev in env.evs
        for node in ev.route_nodes[1:]
        if instance.nodes[node].is_charging_station
    )
    return {
        "method": method,
        "instance_name": instance.instance_name,
        "number_of_customers": instance.num_customers,
        "number_of_stations": instance.num_charging_stations,
        "m_min": m_min,
        "m_max": m_max,
        "selected_m": selected_m,
        "used_EVs": summary.used_evs,
        "feasible": env.feasible_solution(),
        "total_distance": round(summary.total_distance, 4),
        "total_duration": round(summary.total_duration, 4),
        "total_waiting_time": round(summary.total_waiting_time, 4),
        "total_charging_time": round(summary.total_charging_time, 4),
        "total_charging_price": round(summary.total_charging_price, 4),
        "number_of_station_visits": station_visits,
        "number_of_unserved_customers": summary.unserved_customers,
        "number_of_reserved_unfinished_customers": summary.unresolved_reserved_customers,
        "duplicate_customer_visits": summary.duplicate_customer_visits,
        "time_window_violations": env.time_window_violation_count,
        "battery_violations": env.battery_violation_count,
        "capacity_violations": env.capacity_violation_count,
        "depot_return_violations": summary.depot_return_violation,
        "objective_value": round(objective, 4),
        "runtime_seconds": round(runtime_seconds, 4),
    }


def build_routes(instance, env: EVRPTWEnv) -> dict:
    """Build the detailed route export for one solution."""
    routes = []
    for ev in env.evs:
        if not ev.used():
            continue
        duration = ev.current_time
        routes.append({
            "ev_id": ev.ev_id,
            "route_nodes": list(ev.route_nodes),
            "route_string_ids": [instance.nodes[n].string_id for n in ev.route_nodes],
            "arrival_times": [round(t, 4) for t in ev.arrival_times],
            "departure_times": [round(t, 4) for t in ev.departure_times],
            "battery_levels": [round(b, 4) for b in ev.battery_trace],
            "remaining_capacities": [round(c, 4) for c in ev.capacity_trace],
            "charging_actions": ev.charging_actions,
            "distance": round(ev.route_distance, 4),
            "duration": round(duration, 4),
            "waiting_time": round(ev.total_waiting_time, 4),
            "charging_time": round(ev.total_charging_time, 4),
            "charging_price": round(ev.total_charging_price, 4),
            "finished_at_depot": ev.finished and ev.current_node == instance.depot_index,
        })
    return {
        "instance_name": instance.instance_name,
        "feasible": env.feasible_solution(),
        "num_used_evs": env.used_evs(),
        "num_served_customers": env.num_served_customers,
        "routes": routes,
    }
