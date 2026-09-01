"""ALNS-HCO unit checks for the paper method."""
from __future__ import annotations

from src.alns_hco.hco import hco
from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.solver import MODES, solve_alns_hco


def test_one_hco_and_frozen_fleet_settings():
    assert HYPERPARAMETERS["abbreviation"] == "ALNS-HCO"
    assert "fleet_compress_s" not in HYPERPARAMETERS
    assert "exact_oracle_threshold_theta" not in HYPERPARAMETERS
    assert "exactly K=K*" in HYPERPARAMETERS["fleet_constraint"]
    assert "expensive_charge" in HYPERPARAMETERS["phase2_destroy_operators"]
    assert callable(hco)


def test_modes_defined():
    assert "alns_hco" in MODES
    assert "alns_pr" in MODES
    assert "alns_hco_post" in MODES


def test_short_solve_if_converted():
    from src.hanif.loader import load_hanif, list_instances

    names = list_instances()
    if not names:
        return
    inst = load_hanif(names[0] if "C1_10_1" not in names else "C1_10_1", "original")
    result = solve_alns_hco(inst, mode="alns_hco", phase1_s=2.0, phase2_s=2.0, seed=0)
    assert result.metrics.feasible
    assert result.metrics.unserved == 0
    assert result.metrics.hard_violations == 0
    assert result.validation["k_frozen_ok"]
    assert int(result.metrics.used_k) == int(result.validation["k_star"])
