"""Fleet-first heterogeneous-station operational objective Φ.

Ranking (all minimized):
    hard violations, unserved, K, Φ, distance, station time, duration

Classical mode omits Φ and ranks distance after K.

Feasibility timing always uses E[W_s], never CVaR. Risk blend ρ only
affects search preference (effective price / wait in Φ).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable

from src.hanif.env.lognormal import cvar_lognormal

EPS = 1e-9

MODE_CLASSICAL = "classical_distance"
MODE_EXPECTED = "expected_heterogeneous"
MODE_RISK_AWARE = "risk_aware_heterogeneous"


class ObjectiveMode(str, Enum):
    CLASSICAL_DISTANCE = MODE_CLASSICAL
    EXPECTED_HETEROGENEOUS = MODE_EXPECTED
    RISK_AWARE_HETEROGENEOUS = MODE_RISK_AWARE


def parse_objective_mode(value: Any) -> ObjectiveMode:
    raw = str(value or MODE_CLASSICAL).strip().lower().replace("-", "_")
    aliases = {
        "classical": ObjectiveMode.CLASSICAL_DISTANCE,
        "classical_distance": ObjectiveMode.CLASSICAL_DISTANCE,
        "expected": ObjectiveMode.EXPECTED_HETEROGENEOUS,
        "expected_heterogeneous": ObjectiveMode.EXPECTED_HETEROGENEOUS,
        "risk": ObjectiveMode.RISK_AWARE_HETEROGENEOUS,
        "risk_aware": ObjectiveMode.RISK_AWARE_HETEROGENEOUS,
        "risk_aware_heterogeneous": ObjectiveMode.RISK_AWARE_HETEROGENEOUS,
    }
    if raw not in aliases:
        raise ValueError(f"Unknown objective mode {value!r}")
    return aliases[raw]


def objective_spec(config: dict | None) -> dict[str, Any]:
    cfg = dict((config or {}).get("paper_objective") or {})
    mode = parse_objective_mode(cfg.get("mode") or MODE_CLASSICAL)
    alpha = float(cfg.get("risk_alpha", 0.95))
    if mode is ObjectiveMode.EXPECTED_HETEROGENEOUS:
        rho = 0.0
    elif mode is ObjectiveMode.CLASSICAL_DISTANCE:
        rho = 0.0
    else:
        rho = float(cfg.get("risk_blend", 0.25))
    if "risk_blend" in cfg and mode is ObjectiveMode.EXPECTED_HETEROGENEOUS:
        rho = 0.0
    if mode is ObjectiveMode.RISK_AWARE_HETEROGENEOUS and "risk_blend" in cfg:
        rho = float(cfg["risk_blend"])
    return {
        "mode": mode.value,
        "risk_alpha": alpha,
        "risk_blend": rho,
        "density_in_objective": False,
    }


@dataclass
class ObjectiveReference:
    distance: float
    tstation: float
    ccharge: float

    def as_dict(self) -> dict[str, float]:
        return {
            "D_ref": float(self.distance),
            "Tstation_ref": float(self.tstation),
            "Ccharge_ref": float(self.ccharge),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "ObjectiveReference | None":
        if not raw:
            return None
        return cls(
            distance=float(raw.get("D_ref") or raw.get("distance") or 0.0),
            tstation=float(raw.get("Tstation_ref") or raw.get("tstation") or 0.0),
            ccharge=float(raw.get("Ccharge_ref") or raw.get("ccharge") or 0.0),
        )


def _finite(value: float, default: float = 0.0) -> float:
    x = float(value)
    if x != x or x < 0.0:
        return default
    return x


def station_param(instance, node: int, field: str) -> float:
    arr = getattr(instance, field, None)
    if arr is None:
        return 0.0
    v = float(arr[int(node)])
    if v != v:
        return 0.0
    return v


def expected_wait_s(instance, node: int) -> float:
    return station_param(instance, node, "station_wait_expected")


def expected_price_s(instance, node: int) -> float:
    p = station_param(instance, node, "station_price_expected")
    if p > 0.0:
        return p
    return station_param(instance, node, "station_price_per_energy")


def charging_rate_s(instance, node: int) -> float:
    r = station_param(instance, node, "station_charging_rate")
    if r > 1e-12:
        return r
    inv = float(instance.vehicle.inverse_charging_rate or 0.0)
    return (1.0 / inv) if inv > 1e-12 else 1.0


def effective_price_s(instance, node: int, *, rho: float, alpha: float) -> float:
    """Risk-adjusted price. ρ=0 uses stored E[P_s], never CVaR."""
    mean = expected_price_s(instance, node)
    r = min(1.0, max(0.0, float(rho)))
    if r <= 0.0:
        return mean
    sigma = station_param(instance, node, "station_price_sigma")
    if sigma <= 0.0:
        return mean
    mu = station_param(instance, node, "station_price_mu")
    return (1.0 - r) * mean + r * cvar_lognormal(mu, sigma, alpha)


def effective_wait_s(instance, node: int, *, rho: float, alpha: float) -> float:
    """Risk-adjusted wait. Feasibility still uses E[W_s], not this quantity."""
    mean = expected_wait_s(instance, node)
    r = min(1.0, max(0.0, float(rho)))
    if r <= 0.0:
        return mean
    sigma = station_param(instance, node, "station_wait_sigma")
    if sigma <= 0.0:
        return mean
    mu = station_param(instance, node, "station_wait_mu")
    return (1.0 - r) * mean + r * cvar_lognormal(mu, sigma, alpha)


def _is_charge_visit(instance, node: int, charged_energy: float) -> bool:
    if charged_energy <= 1e-12:
        return False
    nd = instance.nodes[int(node)]
    return bool(nd.is_charging_station or nd.is_depot)


def operational_components(
    instance,
    *,
    distance: float,
    charging_visits: Iterable[dict[str, Any]],
    rho: float,
    alpha: float,
) -> dict[str, float]:
    tchg = 0.0
    wrisk = 0.0
    crisk = 0.0
    w_expected = 0.0
    c_expected = 0.0
    cvar_cost = 0.0
    cvar_wait = 0.0
    n_visits = 0
    for ev in charging_visits:
        node = int(ev["station_id"])
        energy = float(ev.get("energy_added") or ev.get("charged_energy") or 0.0)
        if energy <= 1e-12 and not _is_charge_visit(instance, node, energy):
            continue
        n_visits += 1
        rate = charging_rate_s(instance, node)
        tchg += energy / max(rate, EPS)
        wrisk += effective_wait_s(instance, node, rho=rho, alpha=alpha)
        crisk += energy * effective_price_s(instance, node, rho=rho, alpha=alpha)
        w_expected += expected_wait_s(instance, node)
        c_expected += energy * expected_price_s(instance, node)
        mu_p = station_param(instance, node, "station_price_mu")
        sg_p = station_param(instance, node, "station_price_sigma")
        mu_w = station_param(instance, node, "station_wait_mu")
        sg_w = station_param(instance, node, "station_wait_sigma")
        cvar_cost += energy * cvar_lognormal(mu_p, sg_p, alpha) if (mu_p or sg_p) else energy * expected_price_s(instance, node)
        cvar_wait += cvar_lognormal(mu_w, sg_w, alpha) if (mu_w or sg_w) else expected_wait_s(instance, node)
    tstation = tchg + wrisk
    return {
        "distance": float(distance),
        "charging_time": tchg,
        "wrisk": wrisk,
        "crisk": crisk,
        "tstation": tstation,
        "station_waiting_expected": w_expected,
        "expected_charging_cost": c_expected,
        "cvar95_charging_cost_exposure": cvar_cost,
        "cvar95_wait_exposure": cvar_wait,
        "charging_visits": float(n_visits),
    }


def _safe_norm(value: float, ref: float) -> tuple[float, float]:
    """Zero candidate stays zero. If the Classical reference is ~0, use |value|."""
    v = float(value)
    if abs(v) <= EPS:
        return 0.0, max(float(ref), EPS)
    denom = float(ref)
    if denom <= EPS:
        denom = abs(v)
    return v / denom, denom


def normalize_phi(
    *,
    distance: float,
    tstation: float,
    crisk: float,
    reference: ObjectiveReference | None,
) -> tuple[float, dict[str, float]]:
    if reference is None:
        d_ref = max(abs(distance), 1.0)
        t_ref = max(abs(tstation), 1.0)
        c_ref = max(abs(crisk), 1.0)
    else:
        d_ref = float(reference.distance)
        t_ref = float(reference.tstation)
        c_ref = float(reference.ccharge)
    d_norm, d_ref = _safe_norm(distance, d_ref)
    t_norm, t_ref = _safe_norm(tstation, t_ref)
    c_norm, c_ref = _safe_norm(crisk, c_ref)
    phi = d_norm + t_norm + c_norm
    return phi, {
        "D_norm": d_norm,
        "Tstation_norm": t_norm,
        "Ccharge_norm": c_norm,
        "Phi": phi,
        "D_ref": d_ref,
        "Tstation_ref": t_ref,
        "Ccharge_ref": c_ref,
        "zero_reference_rule": "zero candidate component stays 0; if Classical ref is 0 use |value| as denom",
    }


def charging_visits_from_events(instance, route_results: Iterable[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for route in route_results or []:
        for ev in getattr(route, "route_events", None) or []:
            energy = float(getattr(ev, "charged_energy", 0.0) or 0.0)
            node = int(getattr(ev, "node", -1))
            if energy <= 1e-12:
                continue
            if node < 0 or not _is_charge_visit(instance, node, energy):
                continue
            rate = charging_rate_s(instance, node)
            out.append({
                "station_id": node,
                "energy_added": energy,
                "battery_on_arrival": float(getattr(ev, "battery_before", 0.0) or 0.0),
                "battery_after_charge": float(getattr(ev, "battery_after", 0.0) or 0.0),
                "charging_time": energy / max(rate, EPS),
            })
    return out


def phi_from_replay(instance, replay, config: dict | None) -> dict[str, Any]:
    spec = objective_spec(config)
    rho, alpha = float(spec["risk_blend"]), float(spec["risk_alpha"])
    visits = charging_visits_from_events(instance, getattr(replay, "route_results", None) or [])
    comps = operational_components(
        instance,
        distance=float(getattr(replay, "distance", 0.0) or 0.0),
        charging_visits=visits,
        rho=rho,
        alpha=alpha,
    )
    ref = ObjectiveReference.from_dict((config or {}).get("objective_reference"))
    phi, norms = normalize_phi(
        distance=comps["distance"],
        tstation=comps["tstation"],
        crisk=comps["crisk"],
        reference=ref,
    )
    return {**spec, **comps, **norms, "charging_visit_records": visits}


def ranking_tuple(
    *,
    hard: int,
    unserved: int,
    used_k: int,
    phi: float,
    distance: float,
    station_time: float,
    duration: float,
    charging_time: float,
    paper_j: float,
    mode: str,
) -> tuple:
    if mode == MODE_CLASSICAL:
        return (
            int(hard),
            int(unserved),
            int(used_k),
            float(distance),
            float(duration),
        )
    return (
        int(hard),
        int(unserved),
        int(used_k),
        float(phi),
        float(distance),
        float(duration),
    )


def audit_payload(config: dict | None) -> dict[str, Any]:
    spec = objective_spec(config)
    mode = spec["mode"]
    return {
        "mode": mode,
        "formula": (
            "lex(hard, unserved, K, distance, duration)"
            if mode == MODE_CLASSICAL
            else "lex(hard, unserved, K, Phi, distance, duration); "
            "Phi = D/D_class + Tstation/Tstation_class + Ccharge/Ccharge_class; "
            "Tstation = Tchg + Wstation; Ccharge = sum energy * E[price]; "
            "Wstation = sum E[wait] over charging visits"
        ),
        "alpha": spec["risk_alpha"],
        "rho": spec["risk_blend"],
        "normalizers": "frozen Classical solution on the same instance (not a fast constructor)",
        "ranking_tuple": (
            "(hard, unserved, K, distance, duration)"
            if mode == MODE_CLASSICAL
            else "(hard, unserved, K, Phi, distance, duration)"
        ),
        "station_wait_semantics": "feasibility uses E[W]=exp(wait_mu + 0.5 wait_sigma^2); CVaR only in Φ",
        "price_semantics": "E[P]=exp(price_mu + 0.5 price_sigma^2); CVaR only in Φ",
        "feasibility_semantics": "hard TW/battery/capacity/depot; expected wait only; not chance-constrained",
        "density_in_objective": False,
        "unsupported_terms_excluded": [
            "carbon", "battery_degradation", "driver_wage", "parking",
            "station_capacity", "renewable_fraction", "density_score",
        ],
    }


def independent_recompute_phi(instance, replay, config: dict | None) -> dict[str, Any]:
    """Recompute Φ from raw charging events; used to audit phi_from_replay."""
    spec = objective_spec(config)
    visits = []
    for route in getattr(replay, "route_results", None) or []:
        for ev in getattr(route, "route_events", None) or []:
            energy = float(getattr(ev, "charged_energy", 0.0) or 0.0)
            if energy <= 1e-12:
                continue
            visits.append({
                "station_id": int(getattr(ev, "node", -1)),
                "energy_added": energy,
            })
    comps = operational_components(
        instance,
        distance=float(getattr(replay, "distance", 0.0) or 0.0),
        charging_visits=visits,
        rho=float(spec["risk_blend"]),
        alpha=float(spec["risk_alpha"]),
    )
    ref = ObjectiveReference.from_dict((config or {}).get("objective_reference"))
    phi, norms = normalize_phi(
        distance=comps["distance"],
        tstation=comps["tstation"],
        crisk=comps["crisk"],
        reference=ref,
    )
    primary = phi_from_replay(instance, replay, config)
    return {
        "independent_phi": phi,
        "replay_phi": float(primary["Phi"]),
        "agree": abs(phi - float(primary["Phi"])) < 1e-8,
        "independent_crisk": comps["crisk"],
        "replay_crisk": float(primary["crisk"]),
        "independent_tchg": comps["charging_time"],
        "replay_tchg": float(primary["charging_time"]),
        **norms,
    }
