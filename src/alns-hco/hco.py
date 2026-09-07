"""Heterogeneous Charging Oracle (HCO) — detour-only and heterogeneous modes.

Deterministic route-level charging heuristic. Not an exact solver.

Explicit charging plans
-----------------------
Every charging stop carries an explicit fraction alpha in [0, 1] encoding
e_s = alpha * (B_max - B_arr). Missing/None labels are never treated as full
recharge inside HCO (no None -> 1.0).

Battery propagation
-------------------
B_arr(j) = B_dep(i) - E_ij, and at every station B_dep(s) = B_arr(s) + e_s.
Propagation continues through *all* prior charging events.

Modes
-----
- detour: rank/select by (detour, station_id) only among feasible candidates.
  Never uses P_bar, W_bar, r, C_ch, T_wait, T_ch for ranking or tie-breaks.
- heterogeneous: top-k union over detour/price/wait/rate, dominate, then
  compare feasible alternatives by lex (C_ch, T_wait, T_ch, D).

Fixed-route non-worsening
------------------------
Return R_hco only if fully feasible and lex-better than R_in; else R_in.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

from src.alns_hco.hco_activity import CHARGE_QTY_TOL, HcoActivity, OracleCache  # noqa: F401
from src.hanif.model import charging_rate, expected_price, expected_wait
from src.hanif.replay import customer_sequence, replay_route, replay_solution

EPS = 1e-9
HcoMode = Literal["detour", "heterogeneous"]


@dataclass
class ChargingStop:
    """One explicit charging decision (s, e_s) encoded via alpha."""

    position: int
    station: int
    alpha: float
    energy: float = 0.0


@dataclass
class OraclePlan:
    feasible: bool
    nodes: list[int]
    labels: list[float | None]
    distance: float
    duration: float
    charging_expense: float
    charging_time: float
    station_waiting: float
    charged_energy: float
    charging_visits: int
    failure: str | None = None
    selection_mode: str | None = None
    n_insertions: int = 0
    n_replacements: int = 0
    n_removals: int = 0
    n_charge_qty_changes: int = 0
    improved_vs_input: bool = False
    tied_vs_input: bool = False
    rejected_worse: bool = False
    plan_changed: bool = False
    nonempty_candidate_sets: int = 0
    candidates_evaluated: int = 0
    candidates_feasible: int = 0


def _empty(reason: str | None = None, mode: str | None = None) -> OraclePlan:
    return OraclePlan(
        feasible=False, nodes=[], labels=[], distance=1e18, duration=1e18,
        charging_expense=1e18, charging_time=0.0, station_waiting=0.0,
        charged_energy=0.0, charging_visits=0, failure=reason, selection_mode=mode,
    )


def _from_replay(plan, mode: str | None = None, **extra) -> OraclePlan:
    if plan is None or not plan.feasible:
        return _empty(getattr(plan, "failure", None), mode=mode)
    return OraclePlan(
        feasible=True,
        nodes=list(plan.nodes),
        labels=list(plan.labels),
        distance=float(plan.distance),
        duration=float(plan.duration),
        charging_expense=float(plan.charging_expense),
        charging_time=float(plan.charging_time),
        station_waiting=float(plan.station_waiting),
        charged_energy=float(plan.charged_energy),
        charging_visits=int(plan.charging_visits),
        selection_mode=mode,
        **extra,
    )


def _is_station(instance, node: int) -> bool:
    return bool(instance.nodes[int(node)].is_charging_station)


def _stations(instance, nodes: list[int]) -> tuple[int, ...]:
    return tuple(int(n) for n in nodes if _is_station(instance, n))


def charging_plan_stops(instance, nodes: list[int], labels: list[float | None] | None) -> list[ChargingStop]:
    """Extract explicit (position, station, alpha) stops. Missing alpha → 0.0 (never 1.0)."""
    labs = list(labels) if labels is not None else [None] * len(nodes)
    if len(labs) < len(nodes):
        labs = labs + [None] * (len(nodes) - len(labs))
    stops: list[ChargingStop] = []
    batt_cap = float(instance.vehicle.battery_capacity)
    energy = instance.energy_matrix
    batt = batt_cap
    for i in range(1, len(nodes)):
        batt -= float(energy[nodes[i - 1], nodes[i]])
        if not _is_station(instance, nodes[i]):
            continue
        alpha = 0.0 if labs[i] is None else float(labs[i])
        alpha = min(1.0, max(0.0, alpha))
        missing = max(0.0, batt_cap - batt)
        e_s = alpha * missing
        stops.append(ChargingStop(position=i, station=int(nodes[i]), alpha=alpha, energy=e_s))
        batt = min(batt_cap, batt + e_s)
    return stops


def plan_fingerprint(instance, nodes: list[int], labels: list[float | None] | None) -> tuple:
    """Canonical plan signature for change detection."""
    stops = charging_plan_stops(instance, nodes, labels)
    return tuple((s.position, s.station, round(s.alpha, 12)) for s in stops)


def diff_charging_plans(
    instance,
    nodes_in: list[int],
    labels_in: list[float | None] | None,
    nodes_out: list[int],
    labels_out: list[float | None] | None,
) -> dict:
    """Compare input vs output charging plans; count structural / quantity changes."""
    stops_in = charging_plan_stops(instance, nodes_in, labels_in)
    stops_out = charging_plan_stops(instance, nodes_out, labels_out)
    # Align by ordered station visits along the customer sequence context.
    # Structural ops: compare multisets of station ids in order + alphas.
    seq_in = [s.station for s in stops_in]
    seq_out = [s.station for s in stops_out]
    insertions = max(0, len(seq_out) - len(seq_in))
    removals = max(0, len(seq_in) - len(seq_out))
    replacements = 0
    qty_changes = 0
    n = min(len(stops_in), len(stops_out))
    for i in range(n):
        if stops_in[i].station != stops_out[i].station:
            replacements += 1
        elif abs(stops_in[i].alpha - stops_out[i].alpha) > CHARGE_QTY_TOL:
            qty_changes += 1
        elif abs(stops_in[i].energy - stops_out[i].energy) > CHARGE_QTY_TOL:
            qty_changes += 1
    # Position change with same stations still counts as changed fingerprint.
    fp_in = plan_fingerprint(instance, nodes_in, labels_in)
    fp_out = plan_fingerprint(instance, nodes_out, labels_out)
    changed = fp_in != fp_out
    return {
        "changed": changed,
        "insertions": insertions,
        "removals": removals,
        "replacements": replacements,
        "charge_quantity_changes": qty_changes,
        "fp_in": fp_in,
        "fp_out": fp_out,
    }


def ensure_explicit_labels(instance, nodes: list[int], labels: list[float | None] | None) -> list[float | None]:
    """Return labels where every charging stop has an explicit float (default 0.0, never 1.0)."""
    labs = list(labels) if labels is not None else [None] * len(nodes)
    if len(labs) < len(nodes):
        labs = labs + [None] * (len(nodes) - len(labs))
    labs = labs[: len(nodes)]
    out: list[float | None] = []
    for i, n in enumerate(nodes):
        if _is_station(instance, n):
            out.append(0.0 if labs[i] is None else float(labs[i]))
        else:
            out.append(None)
    return out


def replay_explicit(instance, nodes: list[int], labels: list[float | None] | None):
    """Replay with explicit station labels. Missing station label → 0 charge (never full)."""
    labs = ensure_explicit_labels(instance, nodes, labels)
    return replay_route(instance, nodes, labs, default_portion=0.0)


def just_enough_energy(instance, station: int, batt_arr: float, next_node: int, slack: float = 1.0) -> float:
    batt_cap = float(instance.vehicle.battery_capacity)
    missing = max(0.0, batt_cap - float(batt_arr))
    if missing <= EPS:
        return 0.0
    e_next = float(instance.energy_matrix[station, next_node])
    depot = int(instance.depot_index)
    onward = [float(instance.energy_matrix[next_node, depot])]
    for s in instance.charging_station_indices:
        onward.append(float(instance.energy_matrix[next_node, int(s)]))
    need = e_next + min(onward) + float(slack)
    return min(missing, max(0.0, need - float(batt_arr)))


def just_enough_alpha(instance, station: int, batt_arr: float, next_node: int) -> float:
    batt_cap = float(instance.vehicle.battery_capacity)
    missing = max(0.0, batt_cap - float(batt_arr))
    if missing <= EPS:
        return 0.0
    e = just_enough_energy(instance, station, batt_arr, next_node)
    return min(1.0, max(0.0, e / missing))


def just_enough_labels(instance, seq: list[int]) -> list[float | None]:
    """Compute explicit just-enough alpha for *every* charging stop (full sequential walk)."""
    labs: list[float | None] = [None] * len(seq)
    batt_cap = float(instance.vehicle.battery_capacity)
    energy = instance.energy_matrix
    batt = batt_cap
    depot = int(instance.depot_index)
    for i in range(1, len(seq)):
        batt -= float(energy[seq[i - 1], seq[i]])
        if not _is_station(instance, seq[i]):
            continue
        nxt = next(
            (n for n in seq[i + 1:] if instance.nodes[n].is_customer or n == depot),
            depot,
        )
        alpha = just_enough_alpha(instance, seq[i], max(0.0, batt), nxt)
        labs[i] = min(1.0, max(float(alpha), 0.0))
        missing = max(0.0, batt_cap - max(0.0, batt))
        e_s = labs[i] * missing
        batt = min(batt_cap, max(0.0, batt) + e_s)
    return labs


def battery_arrival_at(
    instance,
    nodes: list[int],
    labels: list[float | None] | None,
    pos: int,
) -> float:
    """Arrival battery at nodes[pos], applying charge at every earlier station."""
    batt_cap = float(instance.vehicle.battery_capacity)
    energy = instance.energy_matrix
    labs = ensure_explicit_labels(instance, nodes, labels)
    batt = batt_cap
    for i in range(1, pos + 1):
        batt -= float(energy[nodes[i - 1], nodes[i]])
        if i < pos and _is_station(instance, nodes[i]):
            alpha = float(labs[i] if labs[i] is not None else 0.0)
            missing = max(0.0, batt_cap - max(0.0, batt))
            e_s = min(missing, max(0.0, alpha * missing))
            batt = min(batt_cap, max(0.0, batt) + e_s)
    return batt


# Backward-compatible name used by tests / callers
def _battery_before(instance, nodes: list[int], pos: int, labels: list[float | None] | None = None) -> float:
    return battery_arrival_at(instance, nodes, labels, pos)


def insertion_detour(instance, prev: int, station: int, nxt: int) -> float:
    dist = instance.distance_matrix
    return float(dist[prev, station]) + float(dist[station, nxt]) - float(dist[prev, nxt])


def _attrs(instance, s: int, prev: int, nxt: int) -> tuple[float, float, float, float]:
    return (
        insertion_detour(instance, prev, s, nxt),
        float(expected_price(instance, s)),
        float(expected_wait(instance, s)),
        float(charging_rate(instance, s)),
    )


def _dominates(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    better_or_eq = a[0] <= b[0] + EPS and a[1] <= b[1] + EPS and a[2] <= b[2] + EPS and a[3] + EPS >= b[3]
    strictly = a[0] < b[0] - EPS or a[1] < b[1] - EPS or a[2] < b[2] - EPS or a[3] > b[3] + EPS
    return better_or_eq and strictly


def station_candidates(
    instance,
    prev: int,
    nxt: int,
    *,
    k: int = 5,
    mode: HcoMode = "heterogeneous",
) -> list[int]:
    stations = [int(s) for s in instance.charging_station_indices]
    if not stations:
        return []
    k = max(1, int(k))
    if mode == "detour":
        scored = [(s, insertion_detour(instance, prev, s, nxt)) for s in stations]
        scored.sort(key=lambda t: (t[1], t[0]))  # lower detour, then lower station id
        return [t[0] for t in scored[:k]]
    scored = [(s, *_attrs(instance, s, prev, nxt)) for s in stations]
    by_detour = sorted(scored, key=lambda t: (t[1], t[0]))
    by_price = sorted(scored, key=lambda t: (t[2], t[1], t[0]))
    by_wait = sorted(scored, key=lambda t: (t[3], t[1], t[0]))
    by_rate = sorted(scored, key=lambda t: (-t[4], t[1], t[0]))
    union: list[int] = []
    seen: set[int] = set()
    for pool in (by_detour[:k], by_price[:k], by_wait[:k], by_rate[:k]):
        for t in pool:
            if t[0] not in seen:
                seen.add(t[0])
                union.append(t[0])
    attr = {s: _attrs(instance, s, prev, nxt) for s in union}
    kept = []
    for s in union:
        if any(_dominates(attr[o], attr[s]) for o in union if o != s):
            continue
        kept.append(s)
    return kept if kept else union


def _lex_charge_score(plan: OraclePlan) -> tuple:
    """Heterogeneous / Phase-II charging criterion: Cch → Twait → Tch → D."""
    return (
        float(plan.charging_expense),
        float(plan.station_waiting),
        float(plan.charging_time),
        float(plan.distance),
    )


def _geometric_score(instance, plan: OraclePlan) -> tuple:
    """Detour-mode acceptance / ranking: distance → N_ch → station-id sequence.

    Uses only geometry/feasibility structure — never P_bar, W_bar, r, Cch, Twait, Tch.
    """
    stations = _stations(instance, plan.nodes)
    return (float(plan.distance), len(stations), stations)


def _accept_score(instance, plan: OraclePlan, mode: str | None) -> tuple:
    if mode == "detour":
        return _geometric_score(instance, plan)
    return _lex_charge_score(plan)


def _detour_tie(instance, prev: int, station: int, nxt: int) -> tuple:
    return (insertion_detour(instance, prev, station, nxt), int(station))


def plan_min_charge_stations(
    instance,
    seq: list[int],
    station_pos: int,
    next_node: int,
    base_labels: list[float | None] | None = None,
):
    """Set just-enough charge at station_pos; preserve all other explicit charges."""
    labs = ensure_explicit_labels(instance, seq, base_labels)
    batt = battery_arrival_at(instance, seq, labs, station_pos)
    s = seq[station_pos]
    alpha = just_enough_alpha(instance, s, batt, next_node)
    labs[station_pos] = float(0.0 if alpha is None else alpha)
    # Feasibility: try just-enough, then larger explicit alphas (still explicit).
    for try_alpha in (labs[station_pos], 0.5, 1.0):
        labs[station_pos] = float(try_alpha)
        plan = replay_explicit(instance, seq, labs)
        if plan.feasible:
            return plan
    return plan


def _remap_labels(instance, old_nodes, old_labs, new_nodes) -> list[float | None]:
    """Copy explicit charges for stations that keep the same index identity when possible."""
    old = ensure_explicit_labels(instance, old_nodes, old_labs)
    # Map by walking both sequences: match station occurrences in order for shared customers.
    # Simpler: if lengths equal and only one index differs, copy with that slot unset.
    if len(old_nodes) == len(new_nodes):
        labs = list(old)
        for i, (a, b) in enumerate(zip(old_nodes, new_nodes)):
            if a != b:
                labs[i] = 0.0 if _is_station(instance, b) else None
            elif not _is_station(instance, b):
                labs[i] = None
        return ensure_explicit_labels(instance, new_nodes, labs)
    # Insertion: new_nodes has one extra station relative to old_nodes
    if len(new_nodes) == len(old_nodes) + 1:
        labs: list[float | None] = [None] * len(new_nodes)
        oi = 0
        for ni in range(len(new_nodes)):
            if oi < len(old_nodes) and new_nodes[ni] == old_nodes[oi]:
                labs[ni] = old[oi]
                oi += 1
            elif _is_station(instance, new_nodes[ni]):
                labs[ni] = 0.0  # placeholder; caller overwrites evaluated stop
            else:
                labs[ni] = None
        return ensure_explicit_labels(instance, new_nodes, labs)
    # Removal
    if len(new_nodes) + 1 == len(old_nodes):
        labs = [None] * len(new_nodes)
        ni = 0
        for oi in range(len(old_nodes)):
            if ni < len(new_nodes) and old_nodes[oi] == new_nodes[ni]:
                labs[ni] = old[oi]
                ni += 1
        return ensure_explicit_labels(instance, new_nodes, labs)
    return ensure_explicit_labels(instance, new_nodes, None)


def _seed_stations(instance, customers: list[int], *, k: int, mode: HcoMode) -> OraclePlan:
    depot = int(instance.depot_index)
    energy = instance.energy_matrix
    batt_cap = float(instance.vehicle.battery_capacity)
    custs = [int(c) for c in customers if instance.nodes[int(c)].is_customer]
    nodes = [depot] + custs + [depot]
    labels: list[float | None] = [None] * len(nodes)
    n_ins = n_cand = n_feas = n_sets = 0
    plan = replay_explicit(instance, nodes, labels)
    if plan.feasible:
        return _from_replay(plan, mode=mode)
    for _ in range(8):
        batt = batt_cap
        placed = False
        labels = ensure_explicit_labels(instance, nodes, labels)
        for i in range(1, len(nodes)):
            prev, nxt = nodes[i - 1], nodes[i]
            need = float(energy[prev, nxt])
            if batt + EPS < need and not _is_station(instance, prev):
                cands = station_candidates(instance, prev, nxt, k=k, mode=mode)
                if cands:
                    n_sets += 1
                best = None
                for s in cands:
                    n_cand += 1
                    trial = nodes[:i] + [s] + nodes[i:]
                    base = _remap_labels(instance, nodes, labels, trial)
                    cand_raw = plan_min_charge_stations(instance, trial, i, nxt, base_labels=base)
                    cand = _from_replay(cand_raw, mode=mode)
                    if not cand.feasible:
                        continue
                    n_feas += 1
                    if mode == "detour":
                        rank = _detour_tie(instance, prev, s, nxt)
                    else:
                        rank = _lex_charge_score(cand)
                    if best is None or rank < best[0]:
                        best = (rank, cand)
                if best is None:
                    return _empty("hco_insert_failed", mode=mode)
                nodes, labels = list(best[1].nodes), list(best[1].labels)
                n_ins += 1
                placed = True
                break
            batt -= need
            if _is_station(instance, nxt):
                missing = max(0.0, batt_cap - batt)
                alpha = float(labels[i] if i < len(labels) and labels[i] is not None else 0.0)
                batt = min(batt_cap, batt + alpha * missing)
        plan = replay_explicit(instance, nodes, labels)
        if plan.feasible:
            return _from_replay(
                plan, mode=mode, n_insertions=n_ins,
                nonempty_candidate_sets=n_sets, candidates_evaluated=n_cand,
                candidates_feasible=n_feas,
            )
        if not placed:
            return _empty(plan.failure, mode=mode)
    return _empty("hco_budget", mode=mode)


def _select_best_replace(
    instance, nodes, labels, *, k: int, mode: HcoMode,
) -> OraclePlan:
    """Cleaner replacement pass: evaluate remove + replace with mode-correct ranking."""
    labels = ensure_explicit_labels(instance, nodes, labels)
    seed = _from_replay(replay_explicit(instance, nodes, labels), mode=mode)
    if not seed.feasible:
        return _empty("hco_seed_infeasible", mode=mode)

    best = seed
    best_key: tuple
    if mode == "detour":
        # Keep seed unless a feasible remove/replace has strictly smaller total distance,
        # with station-id / detour used only for choosing among station alternatives.
        best_key = (seed.distance, len(_stations(instance, seed.nodes)), 0)
    else:
        best_key = _lex_charge_score(seed)

    n_rep = n_rem = n_cand = n_feas = n_sets = 0
    action = "keep"
    station_pos = [i for i, n in enumerate(nodes) if _is_station(instance, n)]

    for pos in station_pos:
        nxt = next(
            (n for n in nodes[pos + 1:] if instance.nodes[n].is_customer or n == instance.depot_index),
            instance.depot_index,
        )
        prev = nodes[pos - 1] if pos else nodes[0]

        # Removal
        dropped = nodes[:pos] + nodes[pos + 1:]
        drop_labs = _remap_labels(instance, nodes, labels, dropped)
        cand = _from_replay(replay_explicit(instance, dropped, drop_labs), mode=mode)
        n_cand += 1
        if cand.feasible:
            n_feas += 1
            if mode == "detour":
                key = (cand.distance, len(_stations(instance, cand.nodes)), 0)
            else:
                key = _lex_charge_score(cand)
            if key < best_key:
                best, best_key, action = cand, key, "remove"

        cands = station_candidates(instance, prev, nxt, k=k, mode=mode)
        if cands:
            n_sets += 1
        for s in cands:
            n_cand += 1
            trial = list(nodes)
            trial[pos] = int(s)
            base = _remap_labels(instance, nodes, labels, trial)
            raw = plan_min_charge_stations(instance, trial, pos, nxt, base_labels=base)
            cand = _from_replay(raw, mode=mode)
            if not cand.feasible:
                continue
            n_feas += 1
            if mode == "detour":
                # Feasible station choice ranked by detour then id; acceptance vs seed by distance.
                det = _detour_tie(instance, prev, int(s), nxt)
                key = (cand.distance, len(_stations(instance, cand.nodes)), det[0], det[1])
            else:
                key = _lex_charge_score(cand)
            if key < best_key:
                best, best_key, action = cand, key, ("replace" if int(s) != nodes[pos] else "keep")

    if action == "remove":
        n_rem = 1
    elif action == "replace":
        n_rep = 1
    best.n_replacements = n_rep
    best.n_removals = n_rem
    best.nonempty_candidate_sets = n_sets
    best.candidates_evaluated = n_cand
    best.candidates_feasible = n_feas
    best.improved_vs_input = _accept_score(instance, best, mode) < _accept_score(instance, seed, mode)
    best.tied_vs_input = _accept_score(instance, best, mode) == _accept_score(instance, seed, mode)
    return best


def hco(
    instance,
    customers: list[int],
    cache: OracleCache | None = None,
    *,
    mode: HcoMode = "heterogeneous",
    candidates_per_criterion: int = 5,
    input_nodes: list[int] | None = None,
    input_labels: list[float | None] | None = None,
    non_worsening: bool = False,
) -> OraclePlan:
    """Build a charging plan for the given customer sequence.

    If input_nodes/labels are provided, activity diffs and optional non-worsening
    compare against that input plan.
    """
    cust_key = tuple(int(c) for c in customers)
    key = (cust_key, str(mode), int(candidates_per_criterion))
    act = cache.activity if cache is not None else None
    t0 = time.perf_counter()

    if cache is not None:
        cache.calls += 1
        hit = cache.plans.get(key)
        if hit is not None:
            cache.hits += 1
            if act is not None:
                act.record_cache_hit()
                act.eval_time_s += time.perf_counter() - t0
                # Re-apply non-worsening / activity vs *this* call's input
                return _finalize_vs_input(
                    instance, hit, act, input_nodes, input_labels,
                    mode=mode, non_worsening=non_worsening, from_cache=True, t0=t0,
                )
        if act is not None:
            act.record_cache_miss()

    seed = _seed_stations(instance, list(cust_key), k=candidates_per_criterion, mode=mode)
    if not seed.feasible:
        plan = seed
    else:
        improved = _select_best_replace(
            instance, seed.nodes, seed.labels, k=candidates_per_criterion, mode=mode,
        )
        if improved.feasible:
            plan = improved
            plan.n_insertions = seed.n_insertions
            plan.nonempty_candidate_sets += seed.nonempty_candidate_sets
            plan.candidates_evaluated += seed.candidates_evaluated
            plan.candidates_feasible += seed.candidates_feasible
        else:
            plan = seed

    if cache is not None:
        # Store the raw reconstructed plan (before per-call non-worsening).
        cache.plans[key] = plan

    return _finalize_vs_input(
        instance, plan, act, input_nodes, input_labels,
        mode=mode, non_worsening=non_worsening, from_cache=False, t0=t0,
    )


def _reset_call_outcome_flags(plan: OraclePlan) -> None:
    """Clear per-call outcome flags before final input/output accounting.

    Internal seed/replace comparisons may set improved_vs_input / tied_vs_input;
    those must not leak into HCO activity counters.
    """
    plan.improved_vs_input = False
    plan.tied_vs_input = False
    plan.rejected_worse = False
    plan.plan_changed = False
    plan.n_insertions = 0
    plan.n_removals = 0
    plan.n_replacements = 0
    plan.n_charge_qty_changes = 0


def _finalize_vs_input(
    instance,
    plan: OraclePlan,
    act: HcoActivity | None,
    input_nodes,
    input_labels,
    *,
    mode: str | None,
    non_worsening: bool,
    from_cache: bool,
    t0: float,
) -> OraclePlan:
    mode = mode or plan.selection_mode
    # Default input: empty charging plan on the customer sequence (depot-customers-depot).
    if input_nodes is None:
        in_nodes = list(plan.nodes) if plan.feasible else []
        in_labs = [None] * len(in_nodes)
        if plan.feasible:
            custs = list(customer_sequence(instance, plan.nodes))
            depot = int(instance.depot_index)
            in_nodes = [depot] + custs + [depot]
            in_labs = [None] * len(in_nodes)
    else:
        in_nodes = list(input_nodes)
        in_labs = list(input_labels) if input_labels is not None else [None] * len(in_nodes)

    in_plan = _from_replay(replay_explicit(instance, in_nodes, in_labs), mode=mode)

    # Drop construction/seed flags before the *final* input comparison.
    _reset_call_outcome_flags(plan)
    _reset_call_outcome_flags(in_plan)

    out = plan
    if non_worsening:
        if not plan.feasible:
            out = in_plan if in_plan.feasible else plan
            _reset_call_outcome_flags(out)
            out.rejected_worse = True
        elif in_plan.feasible:
            score_new = _accept_score(instance, plan, mode)
            score_in = _accept_score(instance, in_plan, mode)
            if score_new < score_in:
                out = plan
                _reset_call_outcome_flags(out)
                out.improved_vs_input = True
            elif score_new == score_in:
                out = in_plan
                _reset_call_outcome_flags(out)
                out.tied_vs_input = True
            else:
                out = in_plan
                _reset_call_outcome_flags(out)
                out.rejected_worse = True
        # else: input infeasible — keep feasible HCO plan (no improve/tie/reject vs input)

    # Structural activity counts come *only* from the final input→output plan diff.
    if out.feasible and in_nodes:
        diff = diff_charging_plans(instance, in_nodes, in_labs, out.nodes, out.labels)
        out.plan_changed = bool(diff["changed"])
        out.n_insertions = int(diff["insertions"])
        out.n_removals = int(diff["removals"])
        out.n_replacements = int(diff["replacements"])
        out.n_charge_qty_changes = int(diff["charge_quantity_changes"])
    else:
        out.plan_changed = False
        out.n_insertions = 0
        out.n_removals = 0
        out.n_replacements = 0
        out.n_charge_qty_changes = 0

    if act is not None:
        act.eval_time_s += time.perf_counter() - t0
        if out.feasible:
            act.feasible_hco_calls += 1
        if not from_cache:
            act.nonempty_candidate_sets += int(plan.nonempty_candidate_sets)
            act.candidates_evaluated += int(plan.candidates_evaluated)
            act.candidates_feasible += int(plan.candidates_feasible)
        act.station_insertions += int(out.n_insertions)
        act.station_removals += int(out.n_removals)
        act.station_replacements += int(out.n_replacements)
        act.charge_quantity_changes += int(out.n_charge_qty_changes)
        if out.feasible:
            if out.plan_changed:
                act.changed_calls += 1
            else:
                act.unchanged_calls += 1
            # Mutually exclusive outcome buckets relative to the call input.
            if out.improved_vs_input:
                act.objective_improvements += 1
            elif out.tied_vs_input:
                act.objective_ties += 1
            elif out.rejected_worse:
                act.rejected_worse_plans += 1

    return out


def apply_hco_to_routes(
    instance, routes: list, cache: OracleCache | None = None, *,
    mode: HcoMode = "heterogeneous", candidates_per_criterion: int = 5,
    non_worsening: bool = False,
) -> list:
    out = []
    for nodes, labs in routes:
        custs = list(customer_sequence(instance, nodes))
        if not custs:
            continue
        plan = hco(
            instance, custs, cache, mode=mode,
            candidates_per_criterion=candidates_per_criterion,
            input_nodes=list(nodes), input_labels=list(labs),
            non_worsening=non_worsening,
        )
        if not plan.feasible:
            return []
        out.append((list(plan.nodes), list(plan.labels)))
    return out


def solution_from_hco(
    instance, routes: list, cache: OracleCache | None = None, *,
    mode: HcoMode = "heterogeneous", candidates_per_criterion: int = 5,
    non_worsening: bool = True,
):
    """Fixed-route HCO. Default non_worsening=True.

    Heterogeneous: lex (C_ch, T_wait, T_ch, D).
    Detour: geometric (D, N_ch, station-id sequence) — never uses P/W/r/Cch/Twait/Tch.
    """
    planned = apply_hco_to_routes(
        instance, routes, cache=cache, mode=mode,
        candidates_per_criterion=candidates_per_criterion,
        non_worsening=non_worsening,
    )
    if not planned:
        return None, replay_solution(instance, [])
    return planned, replay_solution(instance, planned)


heterogeneous_charging_oracle = hco
