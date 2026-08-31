"""Post-optimization stochastic robustness evaluation (common random numbers)."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from dataclasses import replace

from src.hanif.evaluation.paper_objective import station_param
from src.hanif.search.route_replay import replay_solution


@dataclass
class StochasticSummary:
    n_scenarios: int
    mean_charging_expense: float
    median_charging_expense: float
    p95_charging_expense: float
    cvar95_charging_expense: float
    mean_station_waiting: float
    p95_station_waiting: float
    cvar95_station_waiting: float
    tw_reliability: float
    depot_reliability: float
    mean_late_customers: float
    p95_completion_delay: float

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def _cvar(samples: np.ndarray, alpha: float = 0.95) -> float:
    if samples.size == 0:
        return 0.0
    var = float(np.quantile(samples, alpha))
    tail = samples[samples >= var]
    return float(np.mean(tail)) if tail.size else var


def _station_ids(instance) -> list[int]:
    return [int(i) for i in instance.charging_station_indices]


def draw_scenario_fields(instance, *, scenario: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """CRN: one wait/price field per station for this scenario index."""
    rng = np.random.default_rng((int(seed) + 1) * 1_000_003 + int(scenario))
    n = int(instance.num_nodes)
    waits = np.zeros(n, dtype=float)
    prices = np.zeros(n, dtype=float)
    for node in _station_ids(instance):
        mu_w = station_param(instance, node, "station_wait_mu")
        sg_w = max(station_param(instance, node, "station_wait_sigma"), 0.0)
        mu_p = station_param(instance, node, "station_price_mu")
        sg_p = max(station_param(instance, node, "station_price_sigma"), 0.0)
        waits[node] = float(rng.lognormal(mu_w, sg_w)) if sg_w > 1e-12 else float(np.exp(mu_w))
        prices[node] = float(rng.lognormal(mu_p, sg_p)) if sg_p > 1e-12 else float(np.exp(mu_p))
    return waits, prices


def evaluate_solution_stochastic(
    instance,
    routes,
    *,
    n_scenarios: int = 500,
    seed: int = 0,
    config: dict | None = None,
) -> StochasticSummary:
    """Fixed-route simulation; sampled W and P; not a chance-constraint claim."""
    det = replay_solution(instance, routes, config=config, m=max(len(routes), 1))
    charge_events: list[tuple[int, float]] = []
    for route in det.route_results:
        for ev in route.route_events:
            energy = float(ev.charged_energy or 0.0)
            if energy > 1e-12 and instance.nodes[ev.node].is_charging_station:
                charge_events.append((int(ev.node), energy))

    expenses = np.zeros(n_scenarios)
    waits = np.zeros(n_scenarios)
    late = np.zeros(n_scenarios)
    tw_ok = np.zeros(n_scenarios, dtype=bool)
    depot_ok = np.zeros(n_scenarios, dtype=bool)
    delay = np.zeros(n_scenarios)
    det_dur = float(det.duration)

    for s in range(int(n_scenarios)):
        w_field, p_field = draw_scenario_fields(instance, scenario=s, seed=seed)
        sampled = replace(instance, station_wait_expected=w_field)
        sim = replay_solution(
            sampled, routes, config=config, m=max(len(routes), 1),
            hard_time_windows=False, stop_on_hard=False,
        )
        exp = sum(energy * float(p_field[node]) for node, energy in charge_events)
        wst = float(sim.station_waiting_time) if sim.station_waiting_time else sum(
            float(w_field[node]) for node, _energy in charge_events
        )
        expenses[s] = exp
        waits[s] = wst
        tw_ok[s] = int(getattr(sim, "time_window_violations", 0) or 0) == 0 and int(sim.late_customers) == 0
        depot_ok[s] = int(getattr(sim, "depot_return_violations", 0) or 0) == 0
        late[s] = float(sim.late_customers)
        delay[s] = max(0.0, float(sim.duration) - det_dur)

    return StochasticSummary(
        n_scenarios=int(n_scenarios),
        mean_charging_expense=float(np.mean(expenses)),
        median_charging_expense=float(np.median(expenses)),
        p95_charging_expense=float(np.quantile(expenses, 0.95)),
        cvar95_charging_expense=_cvar(expenses),
        mean_station_waiting=float(np.mean(waits)),
        p95_station_waiting=float(np.quantile(waits, 0.95)),
        cvar95_station_waiting=_cvar(waits),
        tw_reliability=float(np.mean(tw_ok)),
        depot_reliability=float(np.mean(depot_ok)),
        mean_late_customers=float(np.mean(late)),
        p95_completion_delay=float(np.quantile(delay, 0.95)),
    )
