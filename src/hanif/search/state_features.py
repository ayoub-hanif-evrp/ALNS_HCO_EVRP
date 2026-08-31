"""Observable state features for expert decisions (no instance-name branches)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConstructionPressure:
    """Fleet / TW pressure inferred from state, not filenames."""

    n_unserved: int
    n_critical_slack: int  # due within early horizon
    mean_due: float
    due_std: float
    unused_depot_evs: int
    active_route_slack: float
    customers_depot_illegal: int
    customers_fresh_ev_accessible: int
    mean_battery_margin: float
    fleet_pressure: float  # higher => prefer releasing fresh EVs / seeding

    @property
    def prefer_earliest_due(self) -> bool:
        return self.n_critical_slack >= max(3, self.n_unserved // 5) or self.due_std < 0.35 * max(self.mean_due, 1.0)

    @property
    def prefer_fresh_ev_release(self) -> bool:
        return self.customers_depot_illegal > 0 or self.fleet_pressure > 0.55

    @property
    def prefer_energy_first(self) -> bool:
        return self.mean_battery_margin < 0.25


def compute_construction_pressure(env) -> ConstructionPressure:
    inst = env.instance
    unserved = [
        i for i, n in enumerate(inst.nodes)
        if n.is_customer and not env.served_mask[i]
    ]
    n_unserved = len(unserved)
    if n_unserved == 0:
        return ConstructionPressure(
            n_unserved=0, n_critical_slack=0, mean_due=0.0, due_std=0.0,
            unused_depot_evs=sum(1 for e in env.evs if (not e.finished) and len(e.route_nodes) <= 1),
            active_route_slack=0.0, customers_depot_illegal=0, customers_fresh_ev_accessible=0,
            mean_battery_margin=1.0, fleet_pressure=0.0,
        )
    dues = np.array([float(inst.nodes[j].due_time) for j in unserved], dtype=np.float64)
    mean_due = float(dues.mean())
    due_std = float(dues.std())
    # Critical: due in the earliest 30% of remaining due times.
    q30 = float(np.quantile(dues, 0.30))
    n_critical = int(np.sum(dues <= q30 + 1e-9))

    unused = 0
    battery_margins = []
    for ev in env.evs:
        if ev.finished:
            continue
        if len(ev.route_nodes) <= 1 and ev.current_node == inst.depot_index:
            unused += 1
        batt = float(ev.remaining_battery) / max(float(inst.vehicle.battery_capacity), 1e-9)
        battery_margins.append(batt)

    # Depot-illegal vs fresh EV accessible: use analyze on a depot EV if available.
    depot_illegal = 0
    fresh_ok = 0
    depot_ev = next(
        (e for e in env.evs if (not e.finished) and e.current_node == inst.depot_index and len(e.route_nodes) <= 1),
        None,
    )
    if depot_ev is not None:
        analysis = env.analyze_decision(depot_ev)
        mask = analysis.final_node_mask if analysis is not None else None
        if mask is not None:
            for j in unserved:
                if not bool(mask[j]):
                    depot_illegal += 1
                else:
                    fresh_ok += 1
    else:
        fresh_ok = n_unserved

    # Fleet pressure: unserved per unused+active vehicle.
    active = max(1, sum(1 for e in env.evs if not e.finished))
    fleet_pressure = float(n_unserved) / float(active + unused + 1e-9)
    fleet_pressure = float(np.clip(fleet_pressure / 8.0, 0.0, 1.0))

    # Active route slack: mean (due - time) for earliest unserved vs active EV clocks.
    clocks = [float(e.current_time) for e in env.evs if not e.finished] or [0.0]
    active_route_slack = float(mean_due - float(np.mean(clocks)))

    return ConstructionPressure(
        n_unserved=n_unserved,
        n_critical_slack=n_critical,
        mean_due=mean_due,
        due_std=due_std,
        unused_depot_evs=unused,
        active_route_slack=active_route_slack,
        customers_depot_illegal=depot_illegal,
        customers_fresh_ev_accessible=fresh_ok,
        mean_battery_margin=float(np.mean(battery_margins) if battery_margins else 1.0),
        fleet_pressure=fleet_pressure,
    )
