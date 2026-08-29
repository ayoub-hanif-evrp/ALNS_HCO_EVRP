"""Log-normal station-attribute helpers (price and waiting).

CSV fields ``price_mu``, ``price_sigma``, ``wait_mu``, ``wait_sigma`` are
LogNormal parameters, not raw means.

For X ~ LogNormal(mu, sigma):
    E[X]   = exp(mu + 0.5 * sigma^2)
    Std[X] = E[X] * sqrt(exp(sigma^2) - 1)
    CVaR_alpha(X) = E[X] * Phi(sigma - z_alpha) / (1 - alpha)
where Phi is the standard-normal CDF and z_alpha = Phi^{-1}(alpha).
"""
from __future__ import annotations

import math
from statistics import NormalDist

_UNIT = NormalDist()


def expected_lognormal(mu: float, sigma: float) -> float:
    s = float(sigma)
    if s <= 0.0:
        return float(math.exp(float(mu)))
    return float(math.exp(float(mu) + 0.5 * s * s))


def std_lognormal(mu: float, sigma: float) -> float:
    mean = expected_lognormal(mu, sigma)
    s = float(sigma)
    if s <= 0.0:
        return 0.0
    return float(mean * math.sqrt(math.exp(s * s) - 1.0))


def cvar_lognormal(mu: float, sigma: float, alpha: float = 0.95) -> float:
    """Upper-tail CVaR of a log-normal random variable."""
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    mean = expected_lognormal(mu, sigma)
    s = float(sigma)
    if s <= 0.0:
        return mean
    z = _UNIT.inv_cdf(a)
    return float(mean * _UNIT.cdf(s - z) / (1.0 - a))


def risk_blend(mu: float, sigma: float, *, rho: float, alpha: float = 0.95) -> float:
    """(1-rho) * E[X] + rho * CVaR_alpha(X)."""
    r = min(1.0, max(0.0, float(rho)))
    mean = expected_lognormal(mu, sigma)
    if r <= 0.0:
        return mean
    return (1.0 - r) * mean + r * cvar_lognormal(mu, sigma, alpha)


# Back-compat alias used by older loaders.
expected_lognormal_mean = expected_lognormal
