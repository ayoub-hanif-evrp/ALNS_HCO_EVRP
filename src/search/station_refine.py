"""Stage-2 heterogeneous charging refinement with frozen fleet size K*.

Stage 1 (Classical) minimizes lex(hard, unserved, K, distance, duration).
This module inherits that incumbent, never adds an EV, and searches only:

  - alternative charging stations (operational pool, not distance-only);
  - charge portions;
  - station insertion / deletion on existing routes.

Accept a change only when exact replay stays hard-feasible, 100% served,
used_evs <= K*, and Φ strictly improves.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.evaluation.paper_objective import (
    ObjectiveReference,
    charging_rate_s,
    charging_visits_from_events,
    expected_price_s,
    expected_wait_s,
    normalize_phi,
    objective_spec,
    operational_components,
)
from src.search.deadline import SearchDeadline
from src.search.exact_checks import count_nonempty_routes, exact_accept_solution
from src.search.route_replay import replay_route, replay_solution
from src.search.station_candidates import operational_station_pool

EPS = 1e-12
PORTIONS_DEFAULT = (0.5, 0.75, 1.0)


def _clone_routes(routes) -> list[tuple[list[int], list[float | None]]]:
    return [(list(nodes), list(labels) if labels else [None] * len(nodes)) for nodes, labels in routes]


def _k_star(instance, routes) -> int:
    return int(count_nonempty_routes(instance, routes))


def _anchor(instance, nodes: list[int], pos: int, step: int) -> int:
    i = pos + step
    while 0 <= i < len(nodes):
        nd = instance.nodes[int(nodes[i])]
        if nd.is_customer or nd.is_depot:
            return int(nodes[i])
        i += step
    return int(nodes[0] if step < 0 else nodes[-1])


def _station_positions(instance, nodes: list[int]) -> list[int]:
    return [i for i in range(1, len(nodes) - 1) if instance.nodes[int(nodes[i])].is_charging_station]


def _align_labels(nodes: list[int], labels: list[float | None] | None) -> list[float | None]:
    lab = list(labels) if labels else [None] * len(nodes)
    if len(lab) < len(nodes):
        lab = lab + [None] * (len(nodes) - len(lab))
    return lab[: len(nodes)]


def route_operational(instance, route_replay, config: dict | None) -> dict[str, float]:
    spec = objective_spec(config)
    visits = charging_visits_from_events(instance, [route_replay])
    return operational_components(
        instance,
        distance=float(route_replay.distance),
        charging_visits=visits,
        rho=float(spec["risk_blend"]),
        alpha=float(spec["risk_alpha"]),
    )


def estimated_phi(
    *,
    sol_ops: dict[str, float],
    old_ops: dict[str, float],
    new_ops: dict[str, float],
    reference: ObjectiveReference | None,
) -> float:
    distance = float(sol_ops["distance"]) - float(old_ops["distance"]) + float(new_ops["distance"])
    tstation = float(sol_ops["tstation"]) - float(old_ops["tstation"]) + float(new_ops["tstation"])
    crisk = float(sol_ops["crisk"]) - float(old_ops["crisk"]) + float(new_ops["crisk"])
    phi, _ = normalize_phi(distance=distance, tstation=tstation, crisk=crisk, reference=reference)
    return float(phi)


def visit_phi_burden(
    instance,
    station: int,
    energy: float,
    reference: ObjectiveReference | None,
) -> float:
    """Normalized contribution of one charge visit to the time and cost terms of Φ.

    Minutes and money are not added raw. Energy may be 0 when replay has not
    yet attributed a charge; the wait term still ranks congested stations.
    """
    rate = max(charging_rate_s(instance, int(station)), EPS)
    e = max(float(energy), 0.0)
    t = e / rate + expected_wait_s(instance, int(station))
    c = e * expected_price_s(instance, int(station))
    if reference is None:
        t_ref = max(abs(t), 1.0)
        c_ref = max(abs(c), 1.0)
    else:
        t_ref = max(float(reference.tstation), EPS)
        c_ref = max(float(reference.ccharge), EPS)
    return t / t_ref + c / c_ref


def _visit_energies(instance, route_replay, nodes: list[int]) -> dict[int, float]:
    positions = _station_positions(instance, nodes)
    events: list[tuple[int, float]] = []
    for ev in getattr(route_replay, "route_events", None) or []:
        energy = float(getattr(ev, "charged_energy", 0.0) or 0.0)
        node = int(getattr(ev, "node", -1))
        if energy > EPS and node >= 0:
            events.append((node, energy))
    out: dict[int, float] = {}
    ei = 0
    for pos in positions:
        node = int(nodes[pos])
        energy = 0.0
        for j in range(ei, len(events)):
            if events[j][0] == node:
                energy = events[j][1]
                ei = j + 1
                break
        out[pos] = energy
    return out


def acceptable_refinement(*, replay, k_star: int, current_phi: float) -> bool:
    """Hard no-regression rule for Stage 2."""
    if replay is None:
        return False
    if not bool(getattr(replay, "feasible", False)):
        return False
    if int(getattr(replay, "unserved_customers", 1) or 0) > 0:
        return False
    if int(getattr(replay, "hard_violation_count", 1) or 0) > 0:
        return False
    if int(getattr(replay, "used_evs", 10**9)) > int(k_star):
        return False
    score = getattr(replay, "score", None)
    phi = float(getattr(score, "phi", 1e18) if score is not None else 1e18)
    return phi < float(current_phi) - 1e-12


@dataclass
class StationRefineResult:
    routes: list[tuple[list[int], list[float | None]]]
    replay: Any
    k_star: int
    k_final: int
    accepted: int
    tried: int
    runtime_seconds: float
    phi_initial: float
    phi_final: float
    termination: str
    stats: dict[str, Any] = field(default_factory=dict)


def refine_charging(
    instance,
    routes,
    *,
    config: dict,
    weights: CostWeights | None = None,
    budget_seconds: float = 60.0,
    k_star: int | None = None,
) -> StationRefineResult:
    w = weights or CostWeights()
    current = _clone_routes(routes)
    k_frozen = int(k_star if k_star is not None else _k_star(instance, current))
    deadline = SearchDeadline.from_seconds(budget_seconds, stage="station_refine")
    spec = objective_spec(config)
    ref = ObjectiveReference.from_dict((config or {}).get("objective_reference"))
    portions = list((config.get("problem") or {}).get("charging_portions") or PORTIONS_DEFAULT)
    if 1.0 not in portions:
        portions = list(portions) + [1.0]

    base = replay_solution(instance, current, config=config, weights=w, m=max(k_frozen, 1))
    if base.score is None or not base.feasible:
        return StationRefineResult(
            routes=current,
            replay=base,
            k_star=k_frozen,
            k_final=int(base.used_evs),
            accepted=0,
            tried=0,
            runtime_seconds=deadline.elapsed(),
            phi_initial=float("nan"),
            phi_final=float("nan"),
            termination="infeasible_incumbent",
        )
    phi0 = float(base.score.phi)
    sol_ops = dict(base.phi_breakdown or {})
    if not sol_ops:
        visits = charging_visits_from_events(instance, base.route_results)
        sol_ops = operational_components(
            instance,
            distance=float(base.distance),
            charging_visits=visits,
            rho=float(spec["risk_blend"]),
            alpha=float(spec["risk_alpha"]),
        )
    accepted = 0
    tried = 0
    route_ops = [route_operational(instance, rr, config) for rr in base.route_results]
    termination = "local_optimum"

    def _commit(trial_routes, replay) -> None:
        nonlocal current, base, sol_ops, route_ops, accepted
        current = trial_routes
        base = replay
        accepted += 1
        sol_ops = dict(replay.phi_breakdown or sol_ops)
        route_ops = [route_operational(instance, rr, config) for rr in replay.route_results]

    def _try_route(ri: int, new_nodes: list[int], new_labels: list[float | None]) -> bool:
        nonlocal tried
        tried += 1
        if deadline.expired():
            return False
        rr = replay_route(instance, new_nodes, new_labels)
        if not rr.feasible:
            return False
        new_ops = route_operational(instance, rr, config)
        old_ops = route_ops[ri] if ri < len(route_ops) else {"distance": 0.0, "tstation": 0.0, "crisk": 0.0}
        est = estimated_phi(sol_ops=sol_ops, old_ops=old_ops, new_ops=new_ops, reference=ref)
        if est >= float(base.score.phi) - 1e-12:
            return False
        trial = _clone_routes(current)
        trial[ri] = (list(new_nodes), list(new_labels))
        replay = replay_solution(instance, trial, config=config, weights=w, m=max(k_frozen, 1))
        if not acceptable_refinement(replay=replay, k_star=k_frozen, current_phi=float(base.score.phi)):
            return False
        _commit(trial, replay)
        return True

    def _station_order(ri: int) -> list[int]:
        nodes, _labels = current[ri]
        rr = base.route_results[ri] if ri < len(base.route_results) else None
        energies = _visit_energies(instance, rr, nodes) if rr is not None else {}
        ranked = []
        for pos in _station_positions(instance, nodes):
            s = int(nodes[pos])
            ranked.append((-visit_phi_burden(instance, s, energies.get(pos, 0.0), ref), pos))
        ranked.sort()
        return [pos for _, pos in ranked]

    def _improve_route(ri: int) -> bool:
        nodes, labels = current[ri]
        if not any(instance.nodes[int(n)].is_customer for n in nodes):
            return False
        labels = _align_labels(nodes, labels)
        for pos in _station_order(ri):
            if deadline.expired():
                return False
            prev = _anchor(instance, nodes, pos, -1)
            nxt = _anchor(instance, nodes, pos, 1)
            pool = operational_station_pool(instance, prev, nxt)
            cur_s = int(nodes[pos])
            cur_p = labels[pos]
            for station in pool:
                if deadline.expired():
                    return False
                for portion in portions:
                    if int(station) == cur_s and cur_p is not None and abs(float(portion) - float(cur_p)) < 1e-9:
                        continue
                    n2, l2 = list(nodes), list(labels)
                    n2[pos] = int(station)
                    l2[pos] = float(portion)
                    if _try_route(ri, n2, l2):
                        return True
        for pos in _station_positions(instance, nodes):
            if deadline.expired():
                return False
            cur_p = labels[pos]
            for portion in portions:
                if cur_p is not None and abs(float(portion) - float(cur_p)) < 1e-9:
                    continue
                l2 = list(labels)
                l2[pos] = float(portion)
                if _try_route(ri, list(nodes), l2):
                    return True
        anchors = [
            i for i, n in enumerate(nodes)
            if instance.nodes[int(n)].is_customer or instance.nodes[int(n)].is_depot
        ]
        for a, b in zip(anchors, anchors[1:]):
            if deadline.expired():
                return False
            if b != a + 1:
                continue
            prev, nxt = int(nodes[a]), int(nodes[b])
            pool = operational_station_pool(instance, prev, nxt, k_total=8)
            for station in pool:
                if deadline.expired():
                    return False
                n2, l2 = list(nodes), list(labels)
                n2.insert(b, int(station))
                l2.insert(b, 1.0)
                if _try_route(ri, n2, l2):
                    return True
        for pos in list(_station_positions(instance, nodes)):
            if deadline.expired():
                return False
            n2, l2 = list(nodes), list(labels)
            del n2[pos]
            del l2[pos]
            if _try_route(ri, n2, l2):
                return True
        return False

    used_routes = [
        ri for ri, (nodes, _) in enumerate(current)
        if any(instance.nodes[int(n)].is_customer for n in nodes)
    ]
    while deadline.check():
        progressed = False
        for ri in used_routes:
            if deadline.expired():
                termination = "budget"
                break
            if _improve_route(ri):
                progressed = True
        if deadline.expired():
            termination = "budget"
            break
        if not progressed:
            break

    if deadline.expired() and termination != "infeasible_incumbent":
        termination = "budget"
    return StationRefineResult(
        routes=current,
        replay=base,
        k_star=k_frozen,
        k_final=int(base.used_evs),
        accepted=accepted,
        tried=tried,
        runtime_seconds=deadline.elapsed(),
        phi_initial=phi0,
        phi_final=float(base.score.phi) if base.score is not None else float("nan"),
        termination=termination,
        stats={
            "mode": spec["mode"],
            "k_frozen": True,
            "never_adds_ev": True,
            "route_decomposed": True,
            "visit_rank": "normalized_phi_burden",
            "k_delta": int(base.used_evs) - k_frozen,
        },
    )


def exact_row_from_replay(
    instance,
    routes,
    *,
    config: dict,
    weights: CostWeights | None = None,
    name: str,
    k_star: int,
    runtime_refine_s: float,
    runtime_classical_s: float,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """CSV row + exact-check dict matching run_hanif_fps columns."""
    from src.data.instance_class import instance_class
    from src.evaluation.hanif_protocols import evaluation_protocol
    from src.search.exact_checks import count_charging_visits
    from src.search.fleet_search import capacity_lower_bound

    report, replay, _missing = exact_accept_solution(
        instance, routes, config=config, weights=weights, m=max(k_star, 1), fail_loud=False,
    )
    exact = report.as_dict()
    served = int(instance.num_customers) - int(replay.unserved_customers)
    proto = evaluation_protocol(config)
    row = {
        "instance": name,
        "class": instance_class(name),
        "family": name.split("_")[0],
        "n_customers": int(instance.num_customers),
        "capacity_lb": int(capacity_lower_bound(instance)),
        "search_lo": int(k_star),
        "search_hi": int(k_star),
        "feasible": bool(replay.feasible) and report.ok and report.hard_violations == 0,
        "served": served,
        "used_k": int(replay.used_evs),
        "distance": float(replay.distance),
        "phi": float(report.phi),
        "paper_j": float(report.objective),
        "charging_visits": int(count_charging_visits(instance, routes)),
        "waiting_time": float(exact.get("waiting_time") or 0.0),
        "station_waiting_time": float(exact.get("station_waiting_time") or 0.0),
        "customer_early_waiting_time": float(exact.get("customer_early_waiting_time") or 0.0),
        "charging_time": float(exact.get("charging_time") or 0.0),
        "duration": float(exact.get("duration") or 0.0),
        "expected_charging_cost": float(exact.get("expected_charging_cost") or 0.0),
        "risk_adjusted_charging_cost": float((exact.get("station_metrics") or {}).get("risk_adjusted_charging_cost") or 0.0),
        "cvar95_charging_cost_exposure": float((exact.get("station_metrics") or {}).get("CVaR95_charging_cost_exposure") or 0.0),
        "station_waiting_time_expected": float(exact.get("station_waiting_time_expected") or exact.get("station_waiting_time") or 0.0),
        "risk_adjusted_station_waiting": float((exact.get("station_metrics") or {}).get("risk_adjusted_station_waiting") or 0.0),
        "cvar95_wait_exposure": float((exact.get("station_metrics") or {}).get("CVaR95_wait_exposure") or 0.0),
        "charged_energy": float(exact.get("total_energy_charged") or 0.0),
        "energy_consumed": float(exact.get("total_energy_consumed") or 0.0),
        "travel_time": float(exact.get("travel_time") or 0.0),
        "service_time": float(exact.get("service_time") or 0.0),
        "mean_route_minimum_soc": float((exact.get("battery_metrics") or {}).get("mean_route_minimum_soc") or 0.0),
        "median_route_minimum_soc": float((exact.get("battery_metrics") or {}).get("median_route_minimum_soc") or 0.0),
        "routes_requiring_charging": int((exact.get("battery_metrics") or {}).get("routes_requiring_charging") or 0),
        "hard_violations": int(exact.get("hard_violations") or 0),
        "served_ratio": served / max(int(instance.num_customers), 1),
        "minimum_soc": float(exact.get("minimum_soc_fraction") or 0.0),
        "unique_stations_used": int((exact.get("station_metrics") or {}).get("unique_stations_used") or 0),
        "mean_charging_rate_visited": float((exact.get("station_metrics") or {}).get("mean_charging_rate_visited") or 0.0),
        "mean_expected_price_visited": float((exact.get("station_metrics") or {}).get("mean_expected_price_visited") or 0.0),
        "mean_expected_wait_visited": float((exact.get("station_metrics") or {}).get("mean_expected_wait_visited") or 0.0),
        "d_ref": float((config.get("objective_reference") or {}).get("D_ref") or 0.0),
        "tstation_ref": float((config.get("objective_reference") or {}).get("Tstation_ref") or 0.0),
        "ccharge_ref": float((config.get("objective_reference") or {}).get("Ccharge_ref") or 0.0),
        "intensify_initial_k": k_star,
        "intensify_final_k": int(replay.used_evs),
        "intensify_eliminations": 0,
        "warm_start": True,
        "runtime_refine_s": float(runtime_refine_s),
        "runtime_classical_s": float(runtime_classical_s),
        "runtime_s": float(runtime_classical_s) + float(runtime_refine_s),
        "method": "StationRefinement",
        "evaluation_protocol": proto["name"],
        "intensification_enabled": False,
        "ab_warm_start_enabled": False,
        "k_frozen": True,
        "inherits_classical": True,
    }
    if extra:
        row.update(extra)
    return row, exact
