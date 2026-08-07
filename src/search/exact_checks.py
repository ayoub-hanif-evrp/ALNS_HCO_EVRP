"""Exact acceptance checks that fail loudly on inconsistency."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.env.cost import CostWeights
from src.search.route_replay import replay_solution


class ExactAcceptanceError(RuntimeError):
    """Raised when a claimed solution fails exact consistency checks."""


@dataclass
class ExactCheckReport:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    nonempty_routes: int = 0
    served_customers: list[int] = field(default_factory=list)
    charging_visits: int = 0
    replay_feasible: bool = False
    hard_violations: int = 0
    objective: float = 0.0
    used_vehicles: int = 0
    distance: float = 0.0
    charging_time: float = 0.0
    duration: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "nonempty_routes": self.nonempty_routes,
            "n_served": len(self.served_customers),
            "charging_visits": self.charging_visits,
            "replay_feasible": self.replay_feasible,
            "hard_violations": self.hard_violations,
            "objective": self.objective,
            "used_vehicles": self.used_vehicles,
            "distance": self.distance,
            "charging_time": self.charging_time,
            "duration": self.duration,
        }


def count_charging_visits(instance, routes) -> int:
    n = 0
    for nodes, _ in routes:
        for node in nodes:
            if instance.nodes[node].is_charging_station:
                n += 1
    return n


def count_nonempty_routes(instance, routes) -> int:
    n = 0
    for nodes, _ in routes:
        if any(instance.nodes[x].is_customer for x in nodes):
            n += 1
    return n


def exact_accept_solution(
    instance,
    routes,
    *,
    config: dict | None = None,
    weights: CostWeights | None = None,
    m: int | None = None,
    claimed_score=None,
    fail_loud: bool = True,
) -> tuple[ExactCheckReport, Any, list[int]]:
    """Replay and verify consistency; optionally raise on any disagreement."""
    w = weights or CostWeights()
    report = ExactCheckReport(ok=True)
    final = replay_solution(instance, routes, config=config, weights=w, m=m)
    report.replay_feasible = bool(final.feasible)
    report.hard_violations = int(final.hard_violation_count)
    report.served_customers = list(final.served_customers)
    report.charging_visits = count_charging_visits(instance, routes)
    report.nonempty_routes = count_nonempty_routes(instance, routes)
    if final.score is not None:
        report.used_vehicles = int(final.score.used_evs)
        report.distance = float(final.score.total_distance)
        report.charging_time = float(final.score.total_charging_time)
        report.duration = float(final.score.total_duration)
        report.objective = float(final.score.paper_objective)
    else:
        report.used_vehicles = int(final.used_evs)
        report.distance = float(final.distance)
        report.charging_time = float(final.charging_time)
        report.duration = float(final.duration)
        report.objective = float(getattr(final, "objective", 0.0) or 0.0)

    customers = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    served_set = set(report.served_customers)
    missing = [c for c in customers if c not in served_set]

    # Every customer at most once in routes
    seen = []
    for nodes, _ in routes:
        for node in nodes:
            if instance.nodes[node].is_customer:
                seen.append(int(node))
    if len(seen) != len(set(seen)):
        report.errors.append("duplicate_customer_in_routes")
    extras = [c for c in seen if c not in set(customers)]
    if extras:
        report.errors.append(f"non_customer_marked_served:{extras[:5]}")

    if report.hard_violations > 0:
        report.errors.append(f"hard_violations={report.hard_violations}")
    if final.first_failure:
        report.warnings.append(f"first_failure={final.first_failure}")

    # Used vehicles must equal nonempty routes
    if report.used_vehicles != report.nonempty_routes:
        if report.replay_feasible and missing == []:
            report.errors.append(
                f"used_vehicles_mismatch: score={report.used_vehicles} nonempty={report.nonempty_routes}"
            )
        else:
            report.warnings.append(
                f"used_vehicles_mismatch: score={report.used_vehicles} nonempty={report.nonempty_routes}"
            )

    if claimed_score is not None:
        claim_j = float(getattr(claimed_score, "paper_objective", getattr(claimed_score, "paper_j", 0.0)))
        if abs(claim_j - report.objective) > 1e-3 * max(1.0, abs(report.objective)):
            report.errors.append(
                f"objective_mismatch: claimed={claim_j:.6f} replay={report.objective:.6f}"
            )
        claim_d = float(getattr(claimed_score, "total_distance", 0.0))
        if abs(claim_d - report.distance) > 1e-2 * max(1.0, abs(report.distance)):
            report.errors.append(
                f"distance_mismatch: claimed={claim_d:.6f} replay={report.distance:.6f}"
            )

    # Legal charging portions
    for ri, (nodes, labels) in enumerate(routes):
        labs = list(labels) if labels else [None] * len(nodes)
        for i, node in enumerate(nodes):
            if instance.nodes[node].is_charging_station:
                lab = labs[i] if i < len(labs) else None
                if lab is None:
                    report.warnings.append(f"station_without_portion:route={ri}:pos={i}:node={node}")
                elif not (0.0 < float(lab) <= 1.0 + 1e-9):
                    report.errors.append(f"illegal_charge_portion:{lab}:route={ri}")

    report.ok = len(report.errors) == 0
    if fail_loud and not report.ok:
        raise ExactAcceptanceError("; ".join(report.errors))
    return report, final, missing
