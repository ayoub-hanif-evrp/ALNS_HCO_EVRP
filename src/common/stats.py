"""Paired statistics for ALNS-PR vs ALNS-HCO."""
from __future__ import annotations

import math
import random

TIE_TOL = 1e-6


def _finite(xs: list[float]) -> list[float]:
    return [float(x) for x in xs if x is not None and math.isfinite(float(x))]


def mean(xs: list[float]) -> float | None:
    vals = _finite(xs)
    return sum(vals) / len(vals) if vals else None


def stdev(xs: list[float]) -> float | None:
    vals = _finite(xs)
    if len(vals) < 2:
        return 0.0 if len(vals) == 1 else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((x - m) ** 2 for x in vals) / (len(vals) - 1))


def median(xs: list[float]) -> float | None:
    vals = sorted(_finite(xs))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    if n % 2:
        return vals[mid]
    return 0.5 * (vals[mid - 1] + vals[mid])


def pct_change(new: float, old: float, *, tol: float = TIE_TOL) -> float | None:
    if old is None or new is None:
        return None
    if abs(old) <= tol:
        return 0.0 if abs(new - old) <= tol else None
    return 100.0 * (new - old) / old


def wins_ties_losses(new: list[float], old: list[float], *, tol: float = TIE_TOL) -> tuple[int, int, int]:
    w = t = l = 0
    for a, b in zip(new, old):
        if a is None or b is None:
            continue
        if a < b - tol:
            w += 1
        elif a > b + tol:
            l += 1
        else:
            t += 1
    return w, t, l


def bootstrap_mean_ci(xs: list[float], *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float | None, float | None]:
    vals = _finite(xs)
    if not vals:
        return None, None
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return lo, hi


def wilcoxon_p(diff: list[float], *, tol: float = TIE_TOL) -> float | None:
    """Two-sided Wilcoxon signed-rank p-value. Returns None if scipy is missing or n is too small."""
    vals = [x for x in _finite(diff) if abs(x) > tol]
    if len(vals) < 6:
        return None
    try:
        from scipy.stats import wilcoxon
    except ImportError:
        return _wilcoxon_normal(vals)
    try:
        res = wilcoxon(vals, zero_method="wilcox", alternative="two-sided", mode="auto")
        return float(res.pvalue)
    except ValueError:
        return _wilcoxon_normal(vals)


def _wilcoxon_normal(vals: list[float]) -> float | None:
    n = len(vals)
    ranks = _rank_abs(vals)
    w_pos = sum(r for x, r in zip(vals, ranks) if x > 0)
    mean_w = n * (n + 1) / 4.0
    var_w = n * (n + 1) * (2 * n + 1) / 24.0
    if var_w <= 0:
        return None
    z = (w_pos - mean_w) / math.sqrt(var_w)
    return 2.0 * (1.0 - _phi(abs(z)))


def _rank_abs(vals: list[float]) -> list[float]:
    order = sorted(range(len(vals)), key=lambda i: abs(vals[i]))
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and abs(vals[order[j + 1]]) == abs(vals[order[i]]):
            j += 1
        avg = 0.5 * (i + 1 + j + 1)
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
