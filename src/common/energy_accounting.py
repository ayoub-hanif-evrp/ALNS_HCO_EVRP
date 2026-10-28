"""Post-hoc energy accounting for EVRPTW solutions (measurement only).

Does **not** participate in route construction, selection, feasibility, or
HCO decisions. Uses the same battery / consumption definitions as
``src.hanif.replay.replay_route``:

- Start each used route at full battery ``B_max``.
- Propulsion energy is the sum of ``energy_matrix`` arc consumptions.
- En-route purchased energy is energy charged at charging-station nodes
  (and mid-route depot charging when ``depot_has_charging``).
- Residual battery is the SoC on return to the depot.
- Depot replenishment is ``B_max - residual`` (energy needed to restore full).

Energy-balance identity (per route and in aggregate)::

    B_initial + E_enroute - E_propulsion = B_final
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

EPS = 1e-9
BALANCE_TOL = 1e-6


@dataclass(frozen=True)
class RouteEnergyAccount:
    """Energy account for one vehicle route."""

    route_id: int
    n_customers: int
    feasible_walk: bool
    initial_battery: float
    propulsion_energy: float
    enroute_purchased_energy: float
    residual_battery_at_depot: float
    depot_replenishment_energy: float
    charging_visits: int
    mean_energy_per_charging_visit: float | None
    energy_balance_ok: bool
    energy_balance_residual: float


@dataclass(frozen=True)
class SolutionEnergyAccount:
    """Fleet-level aggregation of route energy accounts."""

    initial_battery_energy: float
    propulsion_energy: float
    enroute_purchased_energy: float
    residual_battery_at_depot: float
    depot_replenishment_energy: float
    charging_visits: int
    mean_energy_per_charging_visit: float | None
    energy_balance_ok: bool
    energy_balance_residual: float
    n_routes: int
    routes: tuple[RouteEnergyAccount, ...] = ()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["routes"] = [asdict(r) for r in self.routes]
        return d

    def reporting_fields(self) -> dict:
        """Flat fields safe to merge into experiment result rows."""
        return {
            "propulsion_energy": float(self.propulsion_energy),
            "enroute_purchased_energy": float(self.enroute_purchased_energy),
            "residual_battery_at_depot": float(self.residual_battery_at_depot),
            "depot_replenishment_energy": float(self.depot_replenishment_energy),
            "mean_energy_per_charging_visit": (
                None
                if self.mean_energy_per_charging_visit is None
                else float(self.mean_energy_per_charging_visit)
            ),
            "initial_battery_energy": float(self.initial_battery_energy),
            "energy_balance_ok": bool(self.energy_balance_ok),
            "energy_balance_residual": float(self.energy_balance_residual),
        }


def _normalize(instance, nodes: list[int], labels: list[float | None] | None):
    depot = int(instance.depot_index)
    seq = [int(n) for n in nodes]
    if not seq:
        seq = [depot, depot]
    if seq[0] != depot:
        seq = [depot] + seq
    if seq[-1] != depot:
        seq = seq + [depot]
    labs = list(labels) if labels is not None else [None] * len(seq)
    if len(labs) < len(seq):
        labs = labs + [None] * (len(seq) - len(labs))
    return seq, labs[: len(seq)]


def _is_charge_stop(instance, node_idx: int, pos: int, n_nodes: int) -> bool:
    node = instance.nodes[int(node_idx)]
    if node.is_charging_station:
        return True
    if node.is_depot and pos < n_nodes - 1 and bool(getattr(instance, "depot_has_charging", False)):
        return True
    return False


def account_route_energy(
    instance,
    nodes: list[int],
    labels: list[float | None] | None = None,
    *,
    route_id: int = 0,
    default_portion: float = 1.0,
) -> RouteEnergyAccount:
    """Walk one route with the same charge rule as ``replay_route`` (measurement)."""
    seq, labs = _normalize(instance, nodes, labels)
    batt_cap = float(instance.vehicle.battery_capacity)
    energy_m = instance.energy_matrix

    batt = batt_cap
    propulsion = 0.0
    purchased = 0.0
    visits = 0
    ok_walk = True

    for i in range(1, len(seq)):
        prev, cur = seq[i - 1], seq[i]
        e_arc = float(energy_m[prev, cur])
        propulsion += e_arc
        batt -= e_arc
        if batt < -EPS:
            ok_walk = False
            # Continue accounting with clamped battery so residual is still defined.
            batt = min(batt, 0.0)

        if _is_charge_stop(instance, cur, i, len(seq)):
            portion = labs[i]
            alpha = float(default_portion if portion is None else portion)
            alpha = min(1.0, max(0.0, alpha))
            missing = max(0.0, batt_cap - batt)
            charged = alpha * missing
            if charged > EPS:
                purchased += charged
                visits += 1
            batt = min(batt_cap, batt + charged)

    residual = max(0.0, float(batt))
    replenish = max(0.0, batt_cap - residual)
    balance_lhs = batt_cap + purchased - propulsion
    balance_res = balance_lhs - residual
    balance_ok = abs(balance_res) <= BALANCE_TOL
    mean_visit = (purchased / visits) if visits > 0 else None
    n_cust = sum(1 for n in seq if instance.nodes[int(n)].is_customer)

    return RouteEnergyAccount(
        route_id=int(route_id),
        n_customers=int(n_cust),
        feasible_walk=bool(ok_walk),
        initial_battery=float(batt_cap),
        propulsion_energy=float(propulsion),
        enroute_purchased_energy=float(purchased),
        residual_battery_at_depot=float(residual),
        depot_replenishment_energy=float(replenish),
        charging_visits=int(visits),
        mean_energy_per_charging_visit=None if mean_visit is None else float(mean_visit),
        energy_balance_ok=bool(balance_ok),
        energy_balance_residual=float(balance_res),
    )


def account_solution_energy(instance, routes: list) -> SolutionEnergyAccount:
    """Aggregate energy accounts over a multi-route solution."""
    route_accounts: list[RouteEnergyAccount] = []
    for rid, item in enumerate(routes):
        if isinstance(item, (list, tuple)) and item and not isinstance(item[0], int):
            nodes, labels = item[0], item[1]
        else:
            nodes, labels = item, None
        # Skip empty/depot-only routes from fleet aggregates of "used" vehicles,
        # but still account their energy if they have travel (defensive).
        acc = account_route_energy(
            instance,
            list(nodes),
            list(labels) if labels is not None else None,
            route_id=rid,
        )
        if acc.n_customers <= 0 and abs(acc.propulsion_energy) <= EPS:
            continue
        route_accounts.append(acc)

    propulsion = sum(r.propulsion_energy for r in route_accounts)
    purchased = sum(r.enroute_purchased_energy for r in route_accounts)
    residual = sum(r.residual_battery_at_depot for r in route_accounts)
    initial = sum(r.initial_battery for r in route_accounts)
    replenish = sum(r.depot_replenishment_energy for r in route_accounts)
    visits = sum(r.charging_visits for r in route_accounts)
    balance_res = (initial + purchased - propulsion) - residual
    mean_visit = (purchased / visits) if visits > 0 else None

    return SolutionEnergyAccount(
        initial_battery_energy=float(initial),
        propulsion_energy=float(propulsion),
        enroute_purchased_energy=float(purchased),
        residual_battery_at_depot=float(residual),
        depot_replenishment_energy=float(replenish),
        charging_visits=int(visits),
        mean_energy_per_charging_visit=None if mean_visit is None else float(mean_visit),
        energy_balance_ok=bool(abs(balance_res) <= BALANCE_TOL) and all(r.energy_balance_ok for r in route_accounts),
        energy_balance_residual=float(balance_res),
        n_routes=len(route_accounts),
        routes=tuple(route_accounts),
    )


def attach_energy_reporting(row: dict, instance, routes: list | None) -> dict:
    """Merge energy reporting fields into an existing result row (in place)."""
    if not routes:
        row.setdefault("propulsion_energy", None)
        row.setdefault("enroute_purchased_energy", None)
        row.setdefault("residual_battery_at_depot", None)
        row.setdefault("depot_replenishment_energy", None)
        row.setdefault("mean_energy_per_charging_visit", None)
        row.setdefault("initial_battery_energy", None)
        row.setdefault("energy_balance_ok", None)
        row.setdefault("energy_balance_residual", None)
        return row
    acc = account_solution_energy(instance, routes)
    row.update(acc.reporting_fields())
    return row


def attach_search_diagnostics(row: dict, *, stages: dict | None = None, validation: dict | None = None) -> dict:
    """Attach ALNS iteration / HCO call / timing diagnostics without affecting solves."""
    stages = stages or {}
    validation = validation or {}
    p1 = stages.get("phase1") or {}
    p2 = stages.get("phase2") or {}
    post = stages.get("postprocess") or {}
    act = (
        p2.get("hco_activity")
        or post.get("hco_activity")
        or validation.get("hco_activity")
        or row.get("hco_activity")
        or {}
    )
    if not isinstance(act, dict):
        act = getattr(act, "as_dict", lambda: {})()

    row["alns_phase1_iterations"] = p1.get("iterations")
    row["alns_phase2_iterations"] = p2.get("iterations")
    row["alns_iterations"] = (
        (int(p1.get("iterations") or 0) + int(p2.get("iterations") or 0))
        if (p1.get("iterations") is not None or p2.get("iterations") is not None)
        else row.get("alns_iterations")
    )
    row["hco_calls"] = act.get("hco_calls", act.get("calls"))
    row["feasible_hco_calls"] = act.get("feasible_hco_calls", act.get("feasible_plans"))
    row["hco_eval_time_s"] = act.get("eval_time_s")
    row["phase1_runtime_s"] = p1.get("runtime_s", row.get("phase1_s"))
    row["phase2_runtime_s"] = p2.get("runtime_s", row.get("phase2_s"))
    # Wall-clock already stored as runtime_s on the row when available.
    row["execution_time_s"] = row.get("runtime_s")
    return row
