"""Stage 2: station + portion refinement on frozen customer sequences.

One operation at each existing charging position: choose (s, α) or delete the stop.
Φ = D/D0 + T/T0 + C_ch/C_ch0 with T = total route duration.
"""
from __future__ import annotations

from src.hanif.model import PORTIONS, PhiReference, charging_rate, expected_price, expected_wait
from src.hanif.replay import customer_sequence, replay_route, replay_solution


def _station_positions(instance, nodes: list[int]) -> list[int]:
    return [i for i in range(1, len(nodes) - 1) if instance.nodes[int(nodes[i])].is_charging_station]


def _anchor(instance, nodes: list[int], pos: int, step: int) -> int:
    i = pos + step
    while 0 <= i < len(nodes):
        nd = instance.nodes[int(nodes[i])]
        if nd.is_customer or nd.is_depot:
            return int(nodes[i])
        i += step
    return int(nodes[0] if step < 0 else nodes[-1])


def candidate_stations(instance, nodes: list[int], pos: int) -> list[int]:
    """{S_current, S_near, S_cheap, S_low-wait, S_fast}."""
    prev = _anchor(instance, nodes, pos, -1)
    nxt = _anchor(instance, nodes, pos, 1)
    current = int(nodes[pos])
    dist = instance.distance_matrix
    stations = list(instance.charging_station_indices)
    if not stations:
        return [current]
    near = min(stations, key=lambda s: (float(dist[prev, s]) + float(dist[s, nxt]), s))
    cheap = min(stations, key=lambda s: (expected_price(instance, s), s))
    low_wait = min(stations, key=lambda s: (expected_wait(instance, s), s))
    fast = min(stations, key=lambda s: (-charging_rate(instance, s), s))
    picked: list[int] = []
    for s in (current, near, cheap, low_wait, fast):
        if s not in picked:
            picked.append(int(s))
    return picked


def _phi_route(plan, ref: PhiReference) -> float:
    return ref.phi(plan.distance, plan.duration, plan.charging_expense)


def refine_route(
    instance,
    nodes: list[int],
    labels: list[float | None],
    ref: PhiReference,
    *,
    max_passes: int = 20,
) -> tuple[list[int], list[float | None]]:
    cur_nodes = list(nodes)
    cur_labs = list(labels) if labels else [None] * len(nodes)
    frozen = customer_sequence(instance, cur_nodes)
    cur = replay_route(instance, cur_nodes, cur_labs)
    if not cur.feasible:
        return nodes, labels
    best_phi = _phi_route(cur, ref)

    for _ in range(max(1, max_passes)):
        improved = False
        for pos in list(_station_positions(instance, cur_nodes)):
            if pos >= len(cur_nodes) or not instance.nodes[cur_nodes[pos]].is_charging_station:
                continue
            best_local = None
            deleted = cur_nodes[:pos] + cur_nodes[pos + 1 :]
            deleted_labs = cur_labs[:pos] + cur_labs[pos + 1 :]
            plan = replay_route(instance, deleted, deleted_labs)
            if plan.feasible and customer_sequence(instance, plan.nodes) == frozen:
                phi = _phi_route(plan, ref)
                if phi + 1e-12 < best_phi:
                    best_local = (phi, plan.nodes, plan.labels)
            for s in candidate_stations(instance, cur_nodes, pos):
                for alpha in PORTIONS:
                    trial_n = list(cur_nodes)
                    trial_l = list(cur_labs)
                    trial_n[pos] = s
                    trial_l[pos] = float(alpha)
                    plan = replay_route(instance, trial_n, trial_l)
                    if not plan.feasible:
                        continue
                    if customer_sequence(instance, plan.nodes) != frozen:
                        continue
                    phi = _phi_route(plan, ref)
                    if phi + 1e-12 < best_phi and (best_local is None or phi < best_local[0]):
                        best_local = (phi, plan.nodes, plan.labels)
            if best_local is not None:
                best_phi, cur_nodes, cur_labs = best_local
                improved = True
        if not improved:
            break
    return cur_nodes, cur_labs


def refine_charging(instance, routes: list, *, max_passes: int = 20) -> tuple[list, dict]:
    classical = replay_solution(instance, routes)
    ref = PhiReference(classical.distance, classical.duration, max(classical.charging_expense, 1e-9))
    out = []
    for nodes, labels in routes:
        if not customer_sequence(instance, nodes):
            continue
        n2, l2 = refine_route(instance, list(nodes), list(labels or [None] * len(nodes)), ref, max_passes=max_passes)
        out.append((n2, l2))
    final = replay_solution(instance, out)
    phi0 = ref.phi(classical.distance, classical.duration, classical.charging_expense)
    phi1 = ref.phi(final.distance, final.duration, final.charging_expense)
    if not final.feasible or phi1 > phi0 + 1e-9:
        return [(list(n), list(l)) for n, l in routes], {
            "accepted": False, "phi_classical": phi0, "phi_proposed": phi0,
        }
    return out, {"accepted": True, "phi_classical": phi0, "phi_proposed": phi1, "reference": {
        "distance": ref.distance, "duration": ref.duration, "charge_cost": ref.charge_cost,
    }}
