"""Phase 0 — environment, matrix and mask sanity checks (no learning).

Run a full battery of correctness checks before any training. Every check is a
small, deterministic probe of the loader, the distance/time/energy matrices, the
hard feasibility masks and the environment physics. Nothing here trains or
mutates a policy; the goal is to fail fast if the simulator is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data.evrptw_dataset import EVRPTWDataset
from src.env.action_masks import compute_action_masks
from src.env.evrptw_env import EVRPTWEnv
from src.env.masks import customer_feasible, station_feasible
from src.evaluation.baselines import NearestFeasiblePolicy, run_heuristic_episode

EPS = 1e-4

# Canonical + legacy customer infeasibility reason names.
SERVED_MASK_REASONS = frozenset({
    "already_served",
    "customer_already_served",
    "hard_customer_already_served",
})
RESERVED_MASK_REASONS = frozenset({
    "reserved_by_other",
    "customer_reserved",
    "hard_customer_reserved",
})
BATTERY_UNREACHABLE_REASONS = frozenset({
    "battery_cannot_reach",
    "customer_battery_unreachable",
    "hard_customer_battery_unreachable",
})
ANTI_STRANDING_REASONS = frozenset({
    "battery_would_strand",
    "customer_infeasible_no_safe_continuation",
    "customer_infeasible_no_reachable_station_or_depot",
    "customer_infeasible_after_service_battery",
    "customer_infeasible_after_service_time",
    "customer_no_safe_continuation",
    "customer_no_safe_direct_depot",
    "customer_no_reachable_station",
    "customer_station_has_no_feasible_charge",
    "hard_customer_no_safe_continuation",
    "hard_customer_no_safe_direct_depot",
})
CAPACITY_INFEASIBLE_REASONS = frozenset({
    "capacity_exceeded",
    "customer_capacity_infeasible",
    "hard_customer_capacity_infeasible",
})
TIME_WINDOW_INFEASIBLE_REASONS = frozenset({
    "time_window_late",
    "horizon_exceeded",
    "customer_time_window_infeasible",
    "hard_customer_time_window_infeasible",
})


def _customer_feasible(env, instance, ev, j):
    return customer_feasible(
        instance, ev, j, env.served_mask, env.reserved_mask, env.use_horizon_check,
        env.use_depot_charging, env.charging_portions,
    )


def _reason_in(reason: str, allowed: frozenset[str]) -> bool:
    if reason in allowed:
        return True
    return any(reason.endswith(r) or r in reason for r in allowed)


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


def _ok(name: str, cond: bool, detail: str = "") -> CheckResult:
    return CheckResult(name=name, passed=bool(cond), detail=detail)


def pick_check_instance(dataset_path: str) -> object:
    """Pick a small instance (fewest customers) that has at least one station."""
    ds = EVRPTWDataset(dataset_path)
    best = None
    best_n = None
    for i in range(len(ds)):
        inst = ds[i]
        if inst.num_charging_stations < 1:
            continue
        if best_n is None or inst.num_customers < best_n:
            best, best_n = inst, inst.num_customers
    if best is None:
        best = ds[0]
    return best


def check_loader(instance) -> CheckResult:
    cond = (
        instance.num_nodes > 0
        and instance.num_customers > 0
        and instance.depot_index >= 0
        and len(instance.customer_indices) == instance.num_customers
    )
    return _ok(
        "dataset_loader",
        cond,
        f"{instance.instance_name}: N={instance.num_nodes}, "
        f"customers={instance.num_customers}, stations={instance.num_charging_stations}",
    )


def check_vehicle_params(instance) -> CheckResult:
    vp = instance.vehicle
    cap_ok = vp.battery_capacity > 0 and vp.load_capacity > 0
    norm_ok = True
    bad_norm = 0
    for idx in instance.customer_indices:
        node = instance.nodes[idx]
        expected = node.demand / vp.load_capacity
        if abs(node.demand_norm - expected) > 1e-5:
            norm_ok = False
            bad_norm += 1
    rate_detail = ""
    if instance.station_charging_rate is not None:
        rates = instance.station_charging_rate[instance.charging_station_indices]
        rates = rates[~np.isnan(rates)]
        if len(rates):
            rate_detail = (
                f", station_rates=[{float(rates.min()):.3f},{float(rates.max()):.3f}]"
            )
    return _ok(
        "vehicle_params",
        cap_ok and norm_ok,
        f"battery={vp.battery_capacity}, load={vp.load_capacity}, h={vp.energy_consumption_rate}"
        f"{rate_detail}, demand_norm_ok={norm_ok}"
        + (f", mismatches={bad_norm}" if bad_norm else ""),
    )


def check_matrices(instance) -> CheckResult:
    d, t, e = instance.distance_matrix, instance.travel_time_matrix, instance.energy_matrix
    n = instance.num_nodes
    shape_ok = d.shape == (n, n) and t.shape == (n, n) and e.shape == (n, n)
    diag_ok = (
        np.allclose(np.diag(d), 0, atol=EPS)
        and np.allclose(np.diag(t), 0, atol=EPS)
        and np.allclose(np.diag(e), 0, atol=EPS)
    )
    nonneg_ok = bool((d >= -EPS).all() and (t >= -EPS).all() and (e >= -EPS).all())
    sym_ok = np.allclose(d, d.T, atol=1e-3)
    return _ok(
        "matrices_valid",
        shape_ok and diag_ok and nonneg_ok and sym_ok,
        f"shape_ok={shape_ok}, zero_diag={diag_ok}, nonneg={nonneg_ok}, dist_symmetric={sym_ok}",
    )


def check_unit_consistency(instance) -> CheckResult:
    d = instance.distance_matrix
    t = instance.travel_time_matrix
    e = instance.energy_matrix
    speed = instance.vehicle.speed
    rate = instance.vehicle.energy_consumption_rate
    off = ~np.eye(d.shape[0], dtype=bool)
    mask = off & (d > EPS)
    if not mask.any():
        return _ok("unit_consistency", False, "no positive off-diagonal distances")
    time_ok = np.allclose(t[mask] * speed, d[mask], rtol=1e-2, atol=1e-2)
    energy_ok = np.allclose(e[mask], d[mask] * rate, rtol=1e-2, atol=1e-2)
    identity = abs(speed - 1.0) < 1e-6 and abs(rate - 1.0) < 1e-6
    return _ok(
        "unit_consistency",
        time_ok and energy_ok,
        f"v={speed}, r={rate}, time=dist/v:{time_ok}, energy=dist*r:{energy_ok}, "
        f"unit_identity(v=1,r=1)={identity}",
    )


def check_served_mask(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=2)
    ev = env.evs[0]
    j = instance.customer_indices[0]
    mask_before, _ = env.feasible_mask(ev)
    env.served_mask[j] = True
    env.num_served_customers += 1
    env.bump_state_version()
    mask_after, reasons = env.feasible_mask(ev)
    reason = reasons.get(j, "")
    return _ok(
        "served_customer_masked",
        (not mask_after[j]) and _reason_in(reason, SERVED_MASK_REASONS),
        f"feasible_before={bool(mask_before[j])}, after_served={bool(mask_after[j])}, reason={reason}",
    )


def check_reserved_mask(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=2)
    ev0, ev1 = env.evs[0], env.evs[1]
    j = instance.customer_indices[0]
    env.reserve_customer(ev0, j)
    env.bump_state_version()
    mask1, reasons = env.feasible_mask(ev1)
    reason = reasons.get(j, "")
    return _ok(
        "reserved_customer_masked",
        (not mask1[j]) and _reason_in(reason, RESERVED_MASK_REASONS),
        f"ev1 sees customer {j} feasible={bool(mask1[j])}, reason={reason}",
    )


def check_battery_reach(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=1)
    ev = env.evs[0]
    j = instance.customer_indices[0]
    need = float(instance.energy_matrix[ev.current_node, j])
    ev.remaining_battery = max(0.0, need - max(need * 0.5, 1.0))
    ok, reason = _customer_feasible(env, instance, ev, j)
    return _ok(
        "battery_cannot_reach",
        (not ok) and _reason_in(reason, BATTERY_UNREACHABLE_REASONS),
        f"battery={ev.remaining_battery:.3f} < energy_to_j={need:.3f}, reason={reason}",
    )


def check_anti_stranding(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=1)
    ev = env.evs[0]
    j = instance.customer_indices[0]
    energy_to_j = float(instance.energy_matrix[ev.current_node, j])
    min_after = float(min(instance.energy_matrix[j, r] for r in instance.rechargeable_indices))
    if min_after <= EPS:
        return _ok("anti_stranding", True, "customer co-located with a recharge node; n/a")
    ev.remaining_battery = energy_to_j + 0.5 * min_after
    ok, reason = _customer_feasible(env, instance, ev, j)
    return _ok(
        "anti_stranding_masked",
        (not ok) and _reason_in(reason, ANTI_STRANDING_REASONS),
        f"battery={ev.remaining_battery:.3f}, reach={energy_to_j:.3f}, escape={min_after:.3f}, reason={reason}",
    )


def check_capacity(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=1)
    ev = env.evs[0]
    j = next(
        (c for c in instance.customer_indices if instance.nodes[c].demand > 0),
        instance.customer_indices[0],
    )
    demand = instance.nodes[j].demand
    ev.remaining_capacity = max(0.0, demand - max(demand * 0.5, 0.5))
    ok, reason = _customer_feasible(env, instance, ev, j)
    return _ok(
        "capacity_exceeded_masked",
        (not ok) and _reason_in(reason, CAPACITY_INFEASIBLE_REASONS | BATTERY_UNREACHABLE_REASONS | ANTI_STRANDING_REASONS),
        f"remaining_cap={ev.remaining_capacity:.3f} < demand={demand:.3f}, reason={reason}",
    )


def check_time_window_late(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=1)
    ev = env.evs[0]
    j = instance.customer_indices[0]
    ev.remaining_battery = instance.vehicle.battery_capacity
    ev.remaining_capacity = instance.vehicle.load_capacity
    ev.current_time = instance.nodes[j].due_time + 100.0
    ok, reason = _customer_feasible(env, instance, ev, j)
    return _ok(
        "late_arrival_infeasible",
        (not ok) and _reason_in(reason, TIME_WINDOW_INFEASIBLE_REASONS),
        f"current_time>{instance.nodes[j].due_time:.1f} (due), reason={reason}",
    )


def check_early_waiting(instance) -> CheckResult:
    j = next(
        (c for c in instance.customer_indices if instance.nodes[c].ready_time > 1.0),
        None,
    )
    if j is None:
        return _ok("early_arrival_waiting", True, "no customer with positive ready_time; n/a")
    env = EVRPTWEnv(instance, m=1)
    ev = env.evs[0]
    ev.remaining_battery = instance.vehicle.battery_capacity
    ev.remaining_capacity = instance.vehicle.load_capacity
    ev.current_time = 0.0
    waiting_before = env.total_waiting_time
    env.apply_action(ev, j)
    arrival = float(instance.travel_time_matrix[instance.depot_index, j])
    expected_wait = max(0.0, instance.nodes[j].ready_time - arrival)
    accrued = env.total_waiting_time - waiting_before
    return _ok(
        "early_arrival_waiting",
        abs(accrued - expected_wait) < 1e-2,
        f"arrival={arrival:.2f}, ready={instance.nodes[j].ready_time:.2f}, "
        f"waiting accrued={accrued:.2f} (expected {expected_wait:.2f})",
    )


def check_station_reusable(instance) -> CheckResult:
    if instance.num_charging_stations == 0:
        return _ok("station_reusable", True, "instance has no stations; n/a")
    env = EVRPTWEnv(instance, m=2)
    s = instance.charging_station_indices[0]
    ev0, ev1 = env.evs[0], env.evs[1]
    ev0.remaining_battery = instance.vehicle.battery_capacity
    ev1.remaining_battery = instance.vehicle.battery_capacity
    ok0, r0 = station_feasible(
        instance, ev0, s, env.served_mask, env.reserved_mask,
        env.use_horizon_check, env.use_depot_charging, env.charging_portions,
    )
    mask0 = compute_action_masks(env, ev0)
    if s in mask0.charge_masks and mask0.charge_masks[s].any():
        portion = env.charging_portions[int(np.argmax(mask0.charge_masks[s]))]
    else:
        portion = 1.0
    # Physical reuse: bypass pruning mask for the first EV visit.
    env.apply_action(ev0, s, charge_portion=portion)
    ok1, r1 = station_feasible(
        instance, ev1, s, env.served_mask, env.reserved_mask,
        env.use_horizon_check, env.use_depot_charging, env.charging_portions,
    )
    return _ok(
        "station_reusable",
        ok0 and ok1,
        f"ev0 physical={ok0} ({r0}), ev1 after reuse physical={ok1} ({r1})",
    )


def check_depot_return(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=1)
    ev = env.evs[0]
    j = instance.customer_indices[0]
    ev.remaining_battery = instance.vehicle.battery_capacity
    ev.remaining_capacity = instance.vehicle.load_capacity
    env.apply_action(ev, j)
    depot = instance.depot_index
    ev.remaining_battery = instance.vehicle.battery_capacity
    env.auto_close_route(ev)
    return _ok(
        "depot_return_works",
        ev.finished and ev.route_nodes[-1] == depot,
        f"finished={ev.finished}, last_node={ev.route_nodes[-1]} (depot={depot})",
    )


def check_all_evs_return(instance) -> CheckResult:
    env = EVRPTWEnv(instance, m=max(instance.m_min + 3, 3))
    run_heuristic_episode(env, NearestFeasiblePolicy())
    depot = instance.depot_index
    used = [ev for ev in env.evs if ev.used()]
    all_home = all(ev.route_nodes and ev.route_nodes[-1] == depot for ev in used)
    return _ok(
        "active_evs_return_to_depot",
        all_home,
        f"used_evs={len(used)}, all ended at depot={all_home}",
    )


ALL_CHECKS = [
    check_loader,
    check_vehicle_params,
    check_matrices,
    check_unit_consistency,
    check_served_mask,
    check_reserved_mask,
    check_battery_reach,
    check_anti_stranding,
    check_capacity,
    check_time_window_late,
    check_early_waiting,
    check_station_reusable,
    check_depot_return,
    check_all_evs_return,
]


def run_phase0_checks(instance) -> list[CheckResult]:
    results: list[CheckResult] = []
    for check in ALL_CHECKS:
        try:
            results.append(check(instance))
        except Exception as exc:
            results.append(CheckResult(name=check.__name__, passed=False, detail=f"exception: {exc!r}"))
    return results


def format_report(results: list[CheckResult]) -> str:
    lines = ["Phase 0 — environment & mask sanity checks", "=" * 52]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        lines.append(f"[{status}] {r.name}")
        if r.detail:
            lines.append(f"        {r.detail}")
    n_pass = sum(1 for r in results if r.passed)
    lines.append("-" * 52)
    lines.append(f"{n_pass}/{len(results)} checks passed")
    return "\n".join(lines)


def phase0_passed(results: list[CheckResult]) -> bool:
    return all(r.passed for r in results)
