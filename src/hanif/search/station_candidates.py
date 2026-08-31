"""Charging-station candidate generation driven by operational attributes.

Fleet construction still uses geographic nearest-station repair. Stage-2
refinement must also propose stations that are attractive on charging rate,
expected waiting, and expected price — not only Euclidean proximity.
"""
from __future__ import annotations

from src.hanif.evaluation.paper_objective import charging_rate_s, expected_price_s, expected_wait_s

EPS = 1e-12


def station_nodes(instance) -> list[int]:
    idx = getattr(instance, "charging_station_indices", None)
    if idx is not None:
        return [int(s) for s in idx]
    return [i for i, n in enumerate(instance.nodes) if n.is_charging_station]


def detour_distance(instance, prev: int, nxt: int, station: int) -> float:
    dist = instance.distance_matrix
    p, n, s = int(prev), int(nxt), int(station)
    via = float(dist[p, s]) + float(dist[s, n])
    if p == n:
        return via
    direct = float(dist[p, n])
    return max(0.0, via - direct)


def geo_nearest_stations(instance, prev: int, nxt: int, k: int = 5) -> list[int]:
    stations = station_nodes(instance)
    if not stations:
        return []
    ranked = sorted(stations, key=lambda s: detour_distance(instance, prev, nxt, s))
    return ranked[: max(1, int(k))]


def _minmax(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    span = hi - lo
    if span <= EPS:
        return [0.0] * len(values)
    return [(v - lo) / span for v in values]


def operational_rank_key(instance, station: int, *, prev: int, nxt: int) -> tuple[float, float, float, float]:
    rate = max(charging_rate_s(instance, station), EPS)
    return (
        detour_distance(instance, prev, nxt, station),
        expected_wait_s(instance, station),
        expected_price_s(instance, station),
        1.0 / rate,
    )


def balanced_union(sequences: list[list[int]], k_total: int, *, quota: int | None = None) -> list[int]:
    """Round-robin unique merge so later criteria are not truncated away.

    Sequential concat-then-slice can spend the entire ``k_total`` on the first
    lists (wait, price, rate) and drop detour/combined. Interleaving with a
    per-list quota keeps every criterion represented.
    """
    limit = max(1, int(k_total))
    if not sequences:
        return []
    q = int(quota) if quota is not None else max(1, (limit + len(sequences) - 1) // len(sequences))
    ordered: list[int] = []
    seen: set[int] = set()
    pointers = [0] * len(sequences)
    taken = [0] * len(sequences)
    progressed = True
    while len(ordered) < limit and progressed:
        progressed = False
        for i, seq in enumerate(sequences):
            if len(ordered) >= limit:
                break
            while pointers[i] < len(seq) and taken[i] < q:
                s = int(seq[pointers[i]])
                pointers[i] += 1
                if s in seen:
                    continue
                seen.add(s)
                ordered.append(s)
                taken[i] += 1
                progressed = True
                break
    if len(ordered) < limit:
        for seq in sequences:
            for s in seq:
                if len(ordered) >= limit:
                    break
                s = int(s)
                if s in seen:
                    continue
                seen.add(s)
                ordered.append(s)
            if len(ordered) >= limit:
                break
    return ordered[:limit]


def operational_station_pool(
    instance,
    prev: int,
    nxt: int,
    *,
    k_each: int = 4,
    k_total: int = 12,
) -> list[int]:
    """Balanced union of wait, price, rate, detour, and combined-score stations.

    A low-wait / low-price / fast station can enter the pool even when it is not
    among the k nearest by detour. Exact replay later rejects infeasible detours.
    """
    stations = station_nodes(instance)
    if not stations:
        return []
    k = max(1, int(k_each))
    limit = max(1, int(k_total))
    by_detour = sorted(stations, key=lambda s: detour_distance(instance, prev, nxt, s))
    by_wait = sorted(stations, key=lambda s: (expected_wait_s(instance, s), detour_distance(instance, prev, nxt, s)))
    by_price = sorted(stations, key=lambda s: (expected_price_s(instance, s), detour_distance(instance, prev, nxt, s)))
    by_rate = sorted(
        stations,
        key=lambda s: (-charging_rate_s(instance, s), detour_distance(instance, prev, nxt, s)),
    )
    nearest_d = detour_distance(instance, prev, nxt, by_detour[0])
    cap = max(3.0 * nearest_d, nearest_d + 40.0)
    eligible = [s for s in stations if detour_distance(instance, prev, nxt, s) <= cap] or by_detour[: max(k * 3, 12)]
    dets = [detour_distance(instance, prev, nxt, s) for s in eligible]
    waits = [expected_wait_s(instance, s) for s in eligible]
    prices = [expected_price_s(instance, s) for s in eligible]
    inv_rates = [1.0 / max(charging_rate_s(instance, s), EPS) for s in eligible]
    nd, nw, np_, nr = _minmax(dets), _minmax(waits), _minmax(prices), _minmax(inv_rates)
    combined = [eligible[i] for i in sorted(
        range(len(eligible)),
        key=lambda i: nd[i] + nw[i] + np_[i] + nr[i],
    )]
    return balanced_union(
        [by_wait[:k], by_price[:k], by_rate[:k], by_detour[:k], combined[:k]],
        limit,
        quota=k,
    )
