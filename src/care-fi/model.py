"""Hanif CARE-FI constants, expected station attributes, and Stage-2 Φ."""
from __future__ import annotations

from dataclasses import dataclass

from src.hanif.env.lognormal import expected_lognormal

PORTIONS = (0.25, 0.50, 0.75, 1.00)
EPS = 1e-12


def expected_wait(instance, station: int) -> float:
    arr = getattr(instance, "station_wait_expected", None)
    if arr is not None:
        v = float(arr[int(station)])
        if v == v:
            return max(0.0, v)
    mu = getattr(instance, "station_wait_mu", None)
    sg = getattr(instance, "station_wait_sigma", None)
    if mu is None:
        return 0.0
    return max(0.0, expected_lognormal(float(mu[int(station)]), float(sg[int(station)] if sg is not None else 0.0)))


def expected_price(instance, station: int) -> float:
    arr = getattr(instance, "station_price_expected", None)
    if arr is not None:
        v = float(arr[int(station)])
        if v == v and v > 0.0:
            return v
    mu = getattr(instance, "station_price_mu", None)
    sg = getattr(instance, "station_price_sigma", None)
    if mu is None:
        return 0.0
    return max(0.0, expected_lognormal(float(mu[int(station)]), float(sg[int(station)] if sg is not None else 0.0)))


def charging_rate(instance, station: int) -> float:
    arr = getattr(instance, "station_charging_rate", None)
    if arr is not None:
        v = float(arr[int(station)])
        if v > EPS:
            return v
    inv = float(instance.vehicle.inverse_charging_rate or 0.0)
    return (1.0 / inv) if inv > EPS else 1.0


@dataclass(frozen=True)
class PhiReference:
    distance: float
    duration: float
    charge_cost: float

    def phi(self, distance: float, duration: float, charge_cost: float) -> float:
        d0 = max(abs(self.distance), EPS)
        t0 = max(abs(self.duration), EPS)
        c0 = max(abs(self.charge_cost), EPS)
        return float(distance) / d0 + float(duration) / t0 + float(charge_cost) / c0
