"""ALNS-HCO unit checks: imports, guardrails, and a short live solve if data exist."""
from __future__ import annotations

from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.alns_hco.oracle import promising_for_exact
from src.alns_hco.solver import MODES, solve_alns_hco


def test_hyperparameters_match_spec():
    assert HYPERPARAMETERS["epsilon_distance"] == 0.01
    assert HYPERPARAMETERS["epsilon_duration"] == 0.0
    assert HYPERPARAMETERS["fleet_compress_s"] == 8.0
    assert HYPERPARAMETERS["exact_oracle_threshold_theta"] == 0.02
    assert "expensive_charge" in HYPERPARAMETERS["phase2_destroy_operators"]
    assert "slow_charge_wait" in HYPERPARAMETERS["phase2_destroy_operators"]


def test_modes_defined():
    assert MODES == ("alns", "alns_hco_post", "alns_hco_fast", "alns_hco")


def test_promising_threshold():
    assert promising_for_exact(100.0, 100.0, 0.02) is True
    assert promising_for_exact(103.0, 100.0, 0.02) is False
    assert promising_for_exact(101.0, 100.0, 0.02) is True


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
    assert result.validation["k_ok"]
    assert result.metrics.used_k <= int(result.validation["k_star"])
