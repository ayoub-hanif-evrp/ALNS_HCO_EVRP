"""Heterogeneous Charging Oracle — compatibility wrapper.

There is one oracle: ``src.alns_hco.hco.hco``. Names ``fast_hco`` / ``exact_hco``
are aliases of that function so archived tests keep importing.
"""
from src.alns_hco.hco import (  # noqa: F401
    EPS,
    OracleCache,
    OraclePlan,
    apply_hco_to_routes as apply_oracle_to_routes,
    hco,
    hco as exact_hco,
    hco as fast_hco,
    just_enough_alpha,
    just_enough_energy,
    just_enough_labels,
    plan_min_charge_stations,
    solution_from_hco as solution_from_oracle,
    station_candidates,
)


def promising_for_exact(hat_cost: float, best_cost: float, theta: float) -> bool:
    return hat_cost < (1.0 + float(theta)) * max(best_cost, EPS)
