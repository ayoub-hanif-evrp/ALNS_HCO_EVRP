"""Feasibility-first exact-mask expert for warm-start demonstrations."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.schneider.env.decision_analysis import DecisionStateAnalysis
from src.schneider.env.ev_state import EVState
from src.schneider.env.policy_action import PolicyAction, apply_policy_action
from src.schneider.utils.logging_utils import get_logger

LOGGER = get_logger()

# Slack buckets (smaller = more urgent).
_BUCKET_CRITICAL = 0
_BUCKET_URGENT = 1
_BUCKET_NORMAL = 2


@dataclass
class ExpertConfig:
    lookahead_depth: int = 1
    max_customer_candidates: int = 15
    max_recharge_candidates: int = 5
    # Cheap type-1 fix: only lookahead among critical-slack customers (not full set).
    critical_lookahead_only: bool = True
    max_critical_lookahead: int = 5
    # After constructive pass, try a short reinsertion repair for leftover customers.
    repair_unserved: bool = True
    repair_max_attempts: int = 40
    debug_dump_dir: str = "outputs/expert_diagnostics"

    @classmethod
    def from_config(cls, config: dict) -> "ExpertConfig":
        raw = (config.get("expert") or {})
        return cls(
            lookahead_depth=int(raw.get("lookahead_depth", 1)),
            max_customer_candidates=int(raw.get("max_customer_candidates", 15)),
            max_recharge_candidates=int(raw.get("max_recharge_candidates", 5)),
            critical_lookahead_only=bool(raw.get("critical_lookahead_only", True)),
            max_critical_lookahead=int(raw.get("max_critical_lookahead", 5)),
            repair_unserved=bool(raw.get("repair_unserved", True)),
            repair_max_attempts=int(raw.get("repair_max_attempts", 40)),
            debug_dump_dir=str(raw.get("debug_dump_dir", "outputs/expert_diagnostics")),
        )


@dataclass(order=True)
class _CustomerRank:
    # Type-1 (tight TW) needs earliest-due inside a slack bucket before follow-on count;
    # otherwise the expert finishes vans while critical customers expire.
    slack_bucket: int
    due_time: float
    neg_serviceable_after: float
    neg_continuation_margin: float
    unserviceable_risk: float
    service_start: float
    travel_distance: float
    node: int


@dataclass(order=True)
class _RechargeRank:
    neg_enabled_customers: float
    neg_enables_depot_return: float
    neg_continuation_margin: float
    travel_time: float
    node: int
    portion: float = 0.0
    depot_finish: bool = False


def _slack_bucket(slack: float, horizon: float) -> int:
    h = max(float(horizon), 1.0)
    if slack <= 0.05 * h:
        return _BUCKET_CRITICAL
    if slack <= 0.15 * h:
        return _BUCKET_URGENT
    return _BUCKET_NORMAL


def _legal_portions(env, analysis: DecisionStateAnalysis, node: int) -> list[float]:
    cm = analysis.final_charge_masks.get(int(node))
    if cm is None:
        return []
    portions = list(env.charging_portions)
    return [float(portions[i]) for i, ok in enumerate(cm) if ok]


def _assert_legal(analysis: DecisionStateAnalysis, env, action: PolicyAction) -> None:
    node = int(action.node)
    if not bool(analysis.final_node_mask[node]):
        raise RuntimeError(f"Expert selected masked node {node}")
    inst = env.instance
    if node == inst.depot_index:
        dmm = analysis.depot_mode_mask
        if action.depot_finish:
            if dmm is None or len(dmm) < 2 or not bool(dmm[1]):
                raise RuntimeError("Expert selected illegal depot finish")
        else:
            if dmm is None or len(dmm) < 1 or not bool(dmm[0]):
                raise RuntimeError("Expert selected illegal depot recharge")
            if action.charge_portion is None:
                raise RuntimeError("Depot recharge missing charge portion")
            legal = _legal_portions(env, analysis, node)
            if not any(abs(p - float(action.charge_portion)) < 1e-9 for p in legal):
                raise RuntimeError(
                    f"Illegal depot charge portion {action.charge_portion}; legal={legal}"
                )
    elif inst.nodes[node].is_charging_station:
        if action.charge_portion is None:
            raise RuntimeError(f"Station {node} missing charge portion")
        legal = _legal_portions(env, analysis, node)
        if not any(abs(p - float(action.charge_portion)) < 1e-9 for p in legal):
            raise RuntimeError(
                f"Illegal station charge portion {action.charge_portion}; legal={legal}"
            )


def _dump_illegal(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


class FeasibilityFirstExpert:
    """Deterministic urgency-aware / feasibility-preserving exact-mask expert."""

    def __init__(self, config: ExpertConfig | dict | None = None):
        if isinstance(config, dict):
            self.cfg = ExpertConfig.from_config(config)
        elif config is None:
            self.cfg = ExpertConfig()
        else:
            self.cfg = config

    def select(self, env, ev, mask_result) -> PolicyAction:
        """Baseline-compatible entry (mask_result ignored; re-uses env.last_decision_analysis)."""
        analysis = getattr(env, "last_decision_analysis", None)
        if analysis is None:
            analysis = env.analyze_decision(ev)
        return self.select_action(env, ev, analysis)

    def select_action(
        self,
        env,
        active_ev: EVState,
        analysis: DecisionStateAnalysis,
    ) -> PolicyAction:
        try:
            action = self._select_action_inner(env, active_ev, analysis)
            _assert_legal(analysis, env, action)
            # Validate through the same apply path as RL (does not mutate if we only validate).
            from src.schneider.env.action_masks import validate_action_against_mask

            validate_action_against_mask(
                analysis.mask_result, env, action.node,
                action.charge_portion, action.depot_finish,
            )
            return action
        except Exception as exc:
            dump = Path(self.cfg.debug_dump_dir) / f"expert_illegal_{int(time.time())}.json"
            _dump_illegal(dump, {
                "error": str(exc),
                "instance": getattr(env.instance, "instance_name", "?"),
                "ev_id": int(active_ev.ev_id),
                "state_version": int(getattr(env, "state_version", -1)),
                "served": int(env.num_served_customers),
                "final_node_mask_sum": int(np.sum(analysis.final_node_mask)),
                "depot_mode_mask": list(map(bool, analysis.depot_mode_mask))
                if analysis.depot_mode_mask is not None else None,
            })
            raise RuntimeError(f"Expert illegal action; diagnostic saved to {dump}") from exc

    def _select_action_inner(
        self,
        env,
        active_ev: EVState,
        analysis: DecisionStateAnalysis,
    ) -> PolicyAction:
        inst = env.instance
        mask = analysis.final_node_mask
        legal_customers = [int(c) for c in inst.customer_indices if bool(mask[c])]

        if legal_customers:
            ranked = self._rank_customers(env, active_ev, analysis, legal_customers)
            # Type-1 Solomon-style: release this EV when it cannot serve the
            # globally most urgent leftover and a fresh van is waiting.
            if self._solomon_type1_should_release_ev(env, active_ev, analysis, legal_customers):
                depot = int(env.instance.depot_index)
                return PolicyAction(node=depot, charge_portion=None, depot_finish=True)
            if self.cfg.lookahead_depth > 0 and len(ranked) > 1:
                candidates = ranked
                if self.cfg.critical_lookahead_only:
                    # Restrict expensive deepcopy lookahead to urgent customers only.
                    critical = []
                    due = env.instance.due_time_array()
                    ready = env.instance.ready_time_array()
                    travel = env.instance.travel_time_matrix
                    horizon = float(env.instance.nodes[env.instance.depot_index].due_time)
                    for j in ranked:
                        jj = int(j)
                        arrival = float(active_ev.current_time + travel[active_ev.current_node, jj])
                        slack = float(due[jj] - max(arrival, ready[jj]))
                        if _slack_bucket(slack, horizon) <= _BUCKET_URGENT:
                            critical.append(jj)
                    if critical:
                        candidates = critical[: max(1, int(self.cfg.max_critical_lookahead))]
                    else:
                        candidates = ranked[:1]  # no urgent → skip multi-candidate lookahead
                else:
                    candidates = ranked[: self.cfg.max_customer_candidates]
                if len(candidates) > 1:
                    best = self._lookahead_pick(
                        env, active_ev, analysis,
                        candidates=[self._customer_action(j) for j in candidates],
                    )
                    if best is not None:
                        return best
            return self._customer_action(ranked[0])

        # Priority 2/3 — useful recharge (station or depot recharge).
        recharge_actions = self._rank_recharges(env, active_ev, analysis)
        if recharge_actions:
            if self.cfg.lookahead_depth > 0 and len(recharge_actions) > 1:
                best = self._lookahead_pick(
                    env, active_ev, analysis,
                    candidates=recharge_actions[: self.cfg.max_recharge_candidates],
                )
                if best is not None:
                    return best
            return recharge_actions[0]

        # Priority 4 — depot finish when truly valid.
        depot = inst.depot_index
        dmm = analysis.depot_mode_mask
        if bool(mask[depot]) and dmm is not None and len(dmm) > 1 and bool(dmm[1]):
            if self._depot_finish_allowed(env, analysis):
                return PolicyAction(node=depot, charge_portion=None, depot_finish=True)

        # Last resort: any legal node (should be rare).
        legal_nodes = np.flatnonzero(mask).tolist()
        if not legal_nodes:
            raise RuntimeError("No legal actions for expert")
        node = int(min(legal_nodes))
        if node == depot:
            if dmm is not None and bool(dmm[1]):
                return PolicyAction(node=depot, charge_portion=None, depot_finish=True)
            portions = _legal_portions(env, analysis, depot)
            if portions:
                return PolicyAction(node=depot, charge_portion=portions[-1], depot_finish=False)
        if inst.nodes[node].is_charging_station:
            portions = _legal_portions(env, analysis, node)
            if portions:
                return PolicyAction(node=node, charge_portion=portions[-1], depot_finish=False)
        return PolicyAction(node=node, charge_portion=None, depot_finish=False)

    @staticmethod
    def _customer_action(j: int) -> PolicyAction:
        return PolicyAction(node=int(j), charge_portion=None, depot_finish=False)

    def _count_follow_on_customers(
        self,
        env,
        ev: EVState,
        j: int,
        *,
        service_end: float,
        batt_after: float,
        cap_after: float,
        legal_customers: list[int],
    ) -> tuple[int, float]:
        """Cheap estimate of remaining EV-local options after serving ``j``.

        Returns (follow_on_count, earliest_other_due_slack) without mutating env.
        """
        inst = env.instance
        travel = inst.travel_time_matrix
        energy = inst.energy_matrix
        ready = inst.ready_time_array()
        due = inst.due_time_array()
        svc = inst.service_time_array()
        demands = inst.demand_array()
        depot = inst.depot_index
        dep_due = float(inst.nodes[depot].due_time)
        count = 0
        risk = 0.0
        for k in legal_customers:
            kk = int(k)
            if kk == int(j):
                continue
            if env.served_mask[kk] or env.reserved_mask[kk]:
                continue
            if float(demands[kk]) > cap_after + 1e-9:
                risk += 1.0
                continue
            e_jk = float(energy[j, kk])
            if batt_after < e_jk - 1e-9:
                risk += 1.0
                continue
            arrival_k = float(service_end + travel[j, kk])
            start_k = float(max(arrival_k, ready[kk]))
            if start_k > float(due[kk]) + 1e-9:
                risk += 1.0
                continue
            end_k = float(start_k + svc[kk])
            batt_k = float(batt_after - e_jk)
            # Must still be able to return to depot after k (safe continuation proxy).
            e_kd = float(energy[kk, depot])
            if batt_k < e_kd - 1e-9:
                risk += 1.0
                continue
            arr_d = end_k + float(travel[kk, depot])
            if arr_d > dep_due + 1e-9:
                risk += 1.0
                continue
            count += 1
        return count, risk

    def _rank_customers(
        self,
        env,
        ev: EVState,
        analysis: DecisionStateAnalysis,
        legal_customers: list[int],
    ) -> list[int]:
        inst = env.instance
        horizon = float(inst.planning_horizon)
        travel = inst.travel_time_matrix
        dist = inst.distance_matrix
        energy = inst.energy_matrix
        ready = inst.ready_time_array()
        due = inst.due_time_array()
        svc = inst.service_time_array()
        demands = inst.demand_array()
        depot = inst.depot_index
        ea = analysis.active_ev_analysis
        gs = analysis.global_serviceability
        base_serviceable = int(getattr(gs, "globally_serviceable_customers_count", 0) or 0)

        ranks: list[_CustomerRank] = []
        for j in legal_customers:
            jj = int(j)
            # Exact mask already implies safe continuation — assert not served/reserved.
            if env.served_mask[jj] or env.reserved_mask[jj]:
                raise RuntimeError(f"Legal mask includes served/reserved customer {jj}")
            arrival = float(ev.current_time + travel[ev.current_node, jj])
            service_start = float(max(arrival, ready[jj]))
            service_end = float(service_start + svc[jj])
            slack = float(due[jj] - service_start)
            batt_after = float(ev.remaining_battery - energy[ev.current_node, jj])
            cap_after = float(ev.remaining_capacity - demands[jj])
            # Direct depot margin after service.
            e_dep = float(energy[jj, depot])
            t_dep = float(travel[jj, depot])
            arr_dep = service_end + t_dep
            dep_due = float(inst.nodes[depot].due_time)
            depot_margin = min(batt_after - e_dep, dep_due - arr_dep)
            follow_on, risk = self._count_follow_on_customers(
                env, ev, jj,
                service_end=service_end,
                batt_after=batt_after,
                cap_after=cap_after,
                legal_customers=legal_customers,
            )
            # Prefer preserving this EV's future options; fall back to global count.
            serviceable_after = float(follow_on) + 0.01 * max(0, base_serviceable - 1)
            cont_margin = float(depot_margin)
            if ea is not None and ea.direct_safe_customer_mask is not None:
                cust = np.asarray(inst.customer_indices, dtype=np.int64)
                idx = int(np.where(cust == jj)[0][0]) if jj in set(cust.tolist()) else -1
                if idx >= 0 and not bool(ea.direct_safe_customer_mask[idx]):
                    cont_margin -= 1e6

            ranks.append(_CustomerRank(
                slack_bucket=_slack_bucket(slack, horizon),
                due_time=float(due[jj]),
                neg_serviceable_after=-float(serviceable_after),
                neg_continuation_margin=-cont_margin,
                unserviceable_risk=float(risk),
                service_start=service_start,
                travel_distance=float(dist[ev.current_node, jj]),
                node=jj,
            ))
        # Tight-TW pressure → earliest-due inside slack bucket (state features, not names).
        from src.schneider.search.state_features import compute_construction_pressure

        pressure = compute_construction_pressure(env)
        if pressure.prefer_earliest_due:
            ranks.sort(key=lambda r: (r.slack_bucket, r.due_time, r.service_start, r.travel_distance, r.node))
        else:
            ranks.sort()
        return [r.node for r in ranks]

    def _rank_recharges(
        self,
        env,
        ev: EVState,
        analysis: DecisionStateAnalysis,
    ) -> list[PolicyAction]:
        inst = env.instance
        mask = analysis.final_node_mask
        travel = inst.travel_time_matrix
        ea = analysis.active_ev_analysis
        node_results = (ea.recharge_node_results if ea is not None else {}) or {}
        ok_dep_direct = bool(ea.safe_depot_return_direct) if ea is not None else False

        candidates: list[tuple[_RechargeRank, PolicyAction]] = []

        # Stations
        for s in inst.charging_station_indices:
            if not bool(mask[int(s)]):
                continue
            portions = _legal_portions(env, analysis, int(s))
            if not portions:
                continue
            nr = node_results.get(int(s))
            enabled = int(nr.enabled_customer_count) if nr is not None else 0
            enables_dep = bool(nr.enables_depot_return) if nr is not None else False
            useful = bool(nr.useful) if nr is not None else (enabled > 0 or enables_dep)
            stranded = not ok_dep_direct and not bool(
                getattr(ea, "safe_depot_return_after_recharge", False)
            )
            all_served = env.num_served_customers >= inst.num_customers
            if not (useful or enabled > 0 or enables_dep or stranded or all_served):
                continue
            best_portion = self._pick_portion(
                env, analysis, int(s), portions, nr, ok_dep_direct,
            )
            rank = _RechargeRank(
                neg_enabled_customers=-float(enabled),
                neg_enables_depot_return=-1.0 if (enables_dep and not ok_dep_direct) else 0.0,
                neg_continuation_margin=-float(enabled) - (10.0 if enables_dep else 0.0),
                travel_time=float(travel[ev.current_node, int(s)]),
                node=int(s),
                portion=float(best_portion),
                depot_finish=False,
            )
            action = PolicyAction(node=int(s), charge_portion=float(best_portion), depot_finish=False)
            candidates.append((rank, action))

        # Depot recharge
        depot = inst.depot_index
        dmm = analysis.depot_mode_mask
        if bool(mask[depot]) and dmm is not None and bool(dmm[0]):
            portions = _legal_portions(env, analysis, depot)
            if portions:
                nr = node_results.get(int(depot))
                enabled = int(nr.enabled_customer_count) if nr is not None else 0
                enables_dep = bool(nr.enables_depot_return) if nr is not None else True
                best_portion = self._pick_portion(
                    env, analysis, depot, portions, nr, ok_dep_direct,
                )
                rank = _RechargeRank(
                    neg_enabled_customers=-float(enabled),
                    neg_enables_depot_return=-1.0 if (enables_dep and not ok_dep_direct) else 0.0,
                    neg_continuation_margin=-float(enabled) - (5.0 if enables_dep else 0.0),
                    travel_time=float(travel[ev.current_node, depot]),
                    node=int(depot),
                    portion=float(best_portion),
                    depot_finish=False,
                )
                action = PolicyAction(
                    node=depot, charge_portion=float(best_portion), depot_finish=False,
                )
                candidates.append((rank, action))

        candidates.sort(key=lambda x: x[0])
        actions = [a for _, a in candidates]
        if actions:
            return actions
        # Fallback: any legal recharge if usefulness filter emptied the set.
        fallback: list[tuple[_RechargeRank, PolicyAction]] = []
        for s in list(inst.charging_station_indices) + [inst.depot_index]:
            s = int(s)
            if not bool(mask[s]):
                continue
            if s == inst.depot_index:
                dmm = analysis.depot_mode_mask
                if dmm is None or not bool(dmm[0]):
                    continue
            portions = _legal_portions(env, analysis, s)
            if not portions:
                continue
            nr = node_results.get(s)
            enabled = int(nr.enabled_customer_count) if nr is not None else 0
            best_portion = self._pick_portion(env, analysis, s, portions, nr, ok_dep_direct)
            rank = _RechargeRank(
                neg_enabled_customers=-float(enabled),
                neg_enables_depot_return=0.0,
                neg_continuation_margin=-float(enabled),
                travel_time=float(travel[ev.current_node, s]),
                node=s,
                portion=float(best_portion),
                depot_finish=False,
            )
            fallback.append((
                rank,
                PolicyAction(node=s, charge_portion=float(best_portion), depot_finish=False),
            ))
        fallback.sort(key=lambda x: x[0])
        return [a for _, a in fallback]

    def _pick_portion(
        self,
        env,
        analysis: DecisionStateAnalysis,
        node: int,
        portions: list[float],
        nr,
        ok_dep_direct: bool,
    ) -> float:
        """Score legal charge portions; prefer useful serviceability, not always 100%."""
        if len(portions) == 1:
            return portions[0]

        # Prefer per-portion recharge labels when available (exact analysis).
        labels = list(getattr(nr, "labels", None) or []) if nr is not None else []
        if labels:
            scored: list[tuple] = []
            for p in portions:
                matches = [
                    lab for lab in labels
                    if lab.charge_portion is not None
                    and abs(float(lab.charge_portion) - float(p)) < 1e-9
                ]
                if not matches:
                    # Keep legal portion as weak fallback.
                    scored.append((0, 0, 1e9, -float(p), float(p)))
                    continue
                en = max(int(getattr(lab, "enabled_customer_count", 0) or 0) for lab in matches)
                dep_ok = any(bool(getattr(lab, "enables_depot_return", False)) for lab in matches)
                # Prefer more enabled customers, depot return if needed, then shorter charge.
                charge_t = min(float(getattr(lab, "cumulative_charging_time", 0.0) or 0.0) for lab in matches)
                scored.append((
                    -en,
                    0 if (dep_ok or ok_dep_direct) else 1,
                    charge_t,
                    # Larger charge only if it adds serviceability (tie-break prefer smaller).
                    float(p) if en == 0 else -float(p),
                    float(p),
                ))
            scored.sort()
            return float(scored[0][-1])

        enabled = int(nr.enabled_customer_count) if nr is not None else 0
        enables_dep = bool(nr.enables_depot_return) if nr is not None else False
        if enabled > 0 or (enables_dep and not ok_dep_direct):
            preferred = [0.75, 0.5, 1.0, 0.25]
            for p in preferred:
                for q in portions:
                    if abs(q - p) < 1e-9:
                        return q
            return portions[-1]
        return portions[0]

    def _depot_finish_allowed(self, env, analysis: DecisionStateAnalysis) -> bool:
        inst = env.instance
        if env.num_served_customers >= inst.num_customers:
            return True
        # No legal customers and no useful recharge left for this EV.
        mask = analysis.final_node_mask
        if any(bool(mask[c]) for c in inst.customer_indices):
            return False
        ea = analysis.active_ev_analysis
        if ea is not None:
            for nr in (ea.recharge_node_results or {}).values():
                if nr is not None and (nr.enabled_customer_count > 0 or nr.useful):
                    # Useful recharge still available — do not finish.
                    node = int(nr.node)
                    if bool(mask[node]) and _legal_portions(env, analysis, node):
                        return False
        gs = analysis.global_serviceability
        if gs is not None and int(getattr(gs, "globally_serviceable_customers_count", 0) or 0) > 0:
            # Prefer keeping this EV alive when the fleet still has work.
            # Only finish if this EV itself has no useful recharge that enables customers.
            ea = analysis.active_ev_analysis
            if ea is not None:
                for nr in (ea.recharge_node_results or {}).values():
                    if nr is not None and int(getattr(nr, "enabled_customer_count", 0) or 0) > 0:
                        return False
            return True
        return True

    def bootstrap_type1_seeds(self, env, *, max_seeds: int | None = None) -> int:
        """Send distinct idle EVs to the earliest-due customers (Solomon seed).

        Returns number of customers seeded.
        """
        inst = env.instance
        from src.schneider.search.state_features import compute_construction_pressure

        pressure = compute_construction_pressure(env)
        if not (pressure.prefer_earliest_due or pressure.prefer_fresh_ev_release):
            return 0
        depot = int(inst.depot_index)
        due = inst.due_time_array()
        idle = [
            e for e in env.evs
            if (not e.finished)
            and int(e.current_node) == depot
            and len(getattr(e, "route_nodes", []) or []) <= 1
        ]
        unserved = [int(c) for c in inst.customer_indices if not env.served_mask[int(c)]]
        # Prefer earliest-due customers that are NOT directly legal from depot
        # (these need a station preposition and are the ones left unserved historically).
        hard = []
        easy = []
        if idle:
            probe = env.analyze_decision(idle[0])
            for c in unserved:
                if bool(probe.final_node_mask[int(c)]):
                    easy.append(int(c))
                else:
                    hard.append(int(c))
        else:
            hard, easy = [], list(unserved)
        hard.sort(key=lambda j: (float(due[int(j)]), int(j)))
        easy.sort(key=lambda j: (float(due[int(j)]), int(j)))
        unserved = hard + easy
        idle.sort(key=lambda e: int(e.ev_id))
        # More seeds when fleet pressure / depot-illegal customers are high.
        if pressure.fleet_pressure > 0.45 or pressure.customers_depot_illegal > 3:
            default_cap = max(22, env.m // 2)
        else:
            default_cap = max(12, env.m // 3)
        n_seed = int(max_seeds if max_seeds is not None else min(len(idle), len(unserved), default_cap))
        # Ensure all hard (station-needed) early customers get a seed slot.
        n_seed = min(len(idle), max(n_seed, min(len(hard) + 8, len(unserved))))
        seeded = 0
        for ev, cust in zip(idle[:n_seed], unserved[:n_seed]):
            try:
                cust = int(cust)
                analysis = env.analyze_decision(ev)
                if not bool(analysis.final_node_mask[cust]):
                    # Need a station stop before a distant early-due customer.
                    stations = list(inst.charging_station_indices)
                    energy = inst.energy_matrix
                    # Prefer stations that leave enough battery for cust → depot/station.
                    stations.sort(
                        key=lambda s: float(energy[int(s), cust]) + float(energy[cust, depot])
                    )
                    placed = False
                    for s in stations[:12]:
                        analysis = env.analyze_decision(ev)
                        if not bool(analysis.final_node_mask[int(s)]):
                            continue
                        portions = []
                        cm = (
                            analysis.final_charge_masks.get(int(s))
                            if analysis.final_charge_masks
                            else None
                        )
                        if cm is not None:
                            portions = [
                                float(p)
                                for p, ok in zip(
                                    list(getattr(env, "charging_portions", None) or [0.25, 0.5, 0.75, 1.0]),
                                    cm,
                                )
                                if ok
                            ]
                        if not portions:
                            continue
                        apply_policy_action(
                            env,
                            ev,
                            PolicyAction(node=int(s), charge_portion=max(portions), depot_finish=False),
                            mask_result=analysis.mask_result,
                        )
                        analysis = env.analyze_decision(ev)
                        if bool(analysis.final_node_mask[cust]):
                            apply_policy_action(
                                env, ev, self._customer_action(cust), mask_result=analysis.mask_result,
                            )
                            seeded += 1
                            placed = True
                            break
                        # Wrong station — this EV is now mid-route; stop seeding it.
                        break
                    if not placed:
                        continue
                    continue
                apply_policy_action(
                    env, ev, self._customer_action(cust), mask_result=analysis.mask_result,
                )
                seeded += 1
            except Exception:
                continue
        return seeded

    def select_acting_ev_for_type1(self, env) -> EVState | None:
        """Prefer an idle depot EV that can serve the most urgent unserved customer.

        Falls back to the env's default event-queue selection.
        """
        inst = env.instance
        from src.schneider.search.state_features import compute_construction_pressure

        pressure = compute_construction_pressure(env)
        if not (pressure.prefer_earliest_due or pressure.prefer_fresh_ev_release):
            return env.select_acting_ev()

        due = inst.due_time_array()
        unserved = [int(c) for c in inst.customer_indices if not env.served_mask[int(c)]]
        if not unserved:
            return env.select_acting_ev()
        unserved.sort(key=lambda j: (float(due[int(j)]), int(j)))
        urgent = int(unserved[0])
        depot = int(inst.depot_index)

        # Prefer unused/idle depot vans that can serve `urgent` now.
        idle = [
            e for e in env.evs
            if (not e.finished)
            and int(e.current_node) == depot
            and len(getattr(e, "route_nodes", []) or []) <= 1
        ]
        for ev in idle:
            try:
                analysis = env.analyze_decision(ev)
                if bool(analysis.final_node_mask[urgent]):
                    return ev
            except Exception:
                continue

        # Else any non-finished EV that can serve urgent.
        for ev in env.evs:
            if ev.finished:
                continue
            try:
                analysis = env.analyze_decision(ev)
                if bool(analysis.final_node_mask[urgent]):
                    return ev
            except Exception:
                continue
        return env.select_acting_ev()

    def repair_unserved(self, env, *, max_attempts: int | None = None) -> int:
        """Multi-EV earliest-due reinsertion of leftover customers.

        For each attempt, pick the most urgent unserved customer and try every
        reopenable EV (regret-lite). Falls back to recharge / open fresh depot
        vans when no direct insertion exists.
        """
        if not bool(self.cfg.repair_unserved):
            return 0
        attempts = int(max_attempts if max_attempts is not None else self.cfg.repair_max_attempts)
        served0 = int(env.num_served_customers)
        depot = int(env.instance.depot_index)
        depot_due = float(env.instance.nodes[depot].due_time)
        due = env.instance.due_time_array()
        travel = env.instance.travel_time_matrix
        ready = env.instance.ready_time_array()

        def _reopen_or_idle() -> list:
            cands = []
            for cand in env.evs:
                at_depot = int(cand.current_node) == depot
                if float(cand.current_time) + 1e-6 >= depot_due:
                    continue
                if cand.finished and at_depot:
                    cands.append(cand)
                elif (not cand.finished) and at_depot:
                    cands.append(cand)
            cands.sort(key=lambda e: (float(e.current_time), -float(e.remaining_battery)))
            return cands

        for _ in range(max(0, attempts)):
            if env.num_served_customers >= env.instance.num_customers:
                break
            unserved = [
                int(c) for c in env.instance.customer_indices if not env.served_mask[int(c)]
            ]
            if not unserved:
                break
            unserved.sort(key=lambda j: (float(due[int(j)]), int(j)))
            target = int(unserved[0])

            # Evaluate all reopenable / idle EVs for a legal insertion of `target`.
            best = None  # (service_start, ev, analysis)
            for cand in _reopen_or_idle():
                was_finished = bool(cand.finished)
                if was_finished:
                    cand.finished = False
                try:
                    analysis = env.analyze_decision(cand)
                    mask = analysis.final_node_mask
                    if bool(mask[target]):
                        arrival = float(cand.current_time + travel[cand.current_node, target])
                        service_start = float(max(arrival, ready[target]))
                        if best is None or service_start < best[0]:
                            best = (service_start, cand, analysis)
                    elif was_finished:
                        cand.finished = True
                except Exception:
                    if was_finished:
                        cand.finished = True

            if best is not None:
                _, ev, analysis = best
                action = self._customer_action(target)
                apply_policy_action(env, ev, action, mask_result=analysis.mask_result)
                continue

            # No EV can take the urgent customer: recharge the earliest idle EV,
            # else finish one EV to rotate the acting set.
            idle = _reopen_or_idle()
            if not idle:
                break
            ev = idle[0]
            ev.finished = False
            analysis = env.analyze_decision(ev)
            mask = analysis.final_node_mask
            # Any other legal customer (next-best urgency).
            legal = [
                int(c) for c in unserved if bool(mask[int(c)])
            ]
            if legal:
                legal.sort(key=lambda j: (float(due[int(j)]), int(j)))
                apply_policy_action(
                    env, ev, self._customer_action(legal[0]), mask_result=analysis.mask_result,
                )
                continue
            recharge_actions = self._rank_recharges(env, ev, analysis)
            if recharge_actions:
                apply_policy_action(
                    env, ev, recharge_actions[0], mask_result=analysis.mask_result,
                )
                continue
            if env.try_auto_close_from_analysis(ev, analysis) is None:
                if bool(mask[depot]):
                    apply_policy_action(
                        env,
                        ev,
                        PolicyAction(node=depot, charge_portion=None, depot_finish=True),
                        mask_result=analysis.mask_result,
                    )
                else:
                    ev.finished = True
        return int(env.num_served_customers) - served0

    def _solomon_type1_should_release_ev(
        self,
        env,
        active_ev: EVState,
        analysis: DecisionStateAnalysis,
        legal_customers: list[int],
    ) -> bool:
        """True when current EV should finish so a fresher van can take urgent work."""
        inst = env.instance
        from src.schneider.search.state_features import compute_construction_pressure

        pressure = compute_construction_pressure(env)
        if not (pressure.prefer_earliest_due or pressure.prefer_fresh_ev_release):
            return False
        unused = [
            e for e in env.evs
            if (not e.finished)
            and int(e.current_node) == int(inst.depot_index)
            and int(e.ev_id) != int(active_ev.ev_id)
            and len(getattr(e, "route_nodes", []) or []) <= 1
        ]
        if not unused:
            return False
        due = inst.due_time_array()
        unserved = [int(c) for c in inst.customer_indices if not env.served_mask[int(c)]]
        if not unserved:
            return False
        unserved.sort(key=lambda j: (float(due[int(j)]), int(j)))
        urgent = int(unserved[0])
        if urgent in set(int(c) for c in legal_customers):
            return False
        # Depot finish must be legal.
        if analysis.depot_mode_mask is None or len(analysis.depot_mode_mask) < 2:
            return False
        return bool(analysis.depot_mode_mask[1])
    def _lookahead_pick(
        self,
        env,
        ev: EVState,
        analysis: DecisionStateAnalysis,
        candidates: list[PolicyAction],
    ) -> PolicyAction | None:
        if not candidates:
            return None
        best_action = candidates[0]
        best_score = -1e30
        for action in candidates:
            try:
                score = self._lookahead_score(env, ev, action)
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _lookahead_score(self, env, ev: EVState, action: PolicyAction) -> float:
        """Cheap 1-step score without deepcopy (critical for type-1 runtime).

        Deepcopy+re-analyze was correct but too slow on 100-customer / high-m
        instances. Approximate post-action state from travel/energy matrices and
        follow-on counts already used by the ranking heuristics.
        """
        inst = env.instance
        j = int(action.node)
        travel = inst.travel_time_matrix
        energy = inst.energy_matrix
        dist_m = inst.distance_matrix
        ready = inst.ready_time_array()
        due = inst.due_time_array()
        svc = inst.service_time_array()
        demands = inst.demand_array()
        depot = int(inst.depot_index)

        served0 = float(env.num_served_customers)
        unserved0 = float(inst.num_customers - env.num_served_customers)

        # Depot finish / recharge: prefer enabling remaining work, not early close.
        if bool(action.depot_finish):
            return (
                1000.0 * served0
                - 200.0 * unserved0
                - 0.001 * float(sum(e.route_distance for e in env.evs))
            )

        # Customer visit approximation.
        if j in set(int(c) for c in inst.customer_indices):
            arrival = float(ev.current_time + travel[ev.current_node, j])
            service_start = float(max(arrival, ready[j]))
            service_end = float(service_start + svc[j])
            slack = float(due[j] - service_start)
            batt_after = float(ev.remaining_battery - energy[ev.current_node, j])
            cap_after = float(ev.remaining_capacity - demands[j])
            legal = [
                int(c) for c in inst.customer_indices
                if (not env.served_mask[int(c)]) and int(c) != j
            ]
            follow_on, risk = self._count_follow_on_customers(
                env, ev, j,
                service_end=service_end,
                batt_after=batt_after,
                cap_after=cap_after,
                legal_customers=legal,
            )
            e_dep = float(energy[j, depot])
            t_dep = float(travel[j, depot])
            depot_ok = (batt_after >= e_dep - 1e-6) and (
                service_end + t_dep <= float(inst.nodes[depot].due_time) + 1e-6
            )
            return (
                1000.0 * (served0 + 1.0)
                + 10.0 * float(follow_on)
                + 0.05 * float(slack)
                + 0.01 * batt_after
                - 50.0 * float(risk)
                - 0.001 * float(dist_m[ev.current_node, j])
                - (0.0 if depot_ok else 100.0)
                - 5.0 * max(0.0, unserved0 - 1.0)
            )

        # Station / depot recharge: prefer actions that enable more customers.
        arrival = float(ev.current_time + travel[ev.current_node, j])
        batt_after = float(ev.remaining_battery - energy[ev.current_node, j])
        portion = 0.0 if action.charge_portion is None else float(action.charge_portion)
        # Soft battery fill proxy (exact charge dynamics live in the env).
        batt_fill = batt_after + portion * float(getattr(inst, "battery_capacity", batt_after) or batt_after)
        return (
            1000.0 * served0
            + 0.02 * float(batt_fill)
            - 5.0 * unserved0
            - 0.01 * float(arrival)
            - 0.001 * float(dist_m[ev.current_node, j])
        )


__all__ = ["ExpertConfig", "FeasibilityFirstExpert"]
