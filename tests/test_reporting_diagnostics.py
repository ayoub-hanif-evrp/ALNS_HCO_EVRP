"""Round-2 reporting diagnostics: iterations, runtime split, CSV survival."""
from __future__ import annotations

from pathlib import Path

from src.common.energy_accounting import (
    attach_search_diagnostics,
    sum_optional_ints,
)
from src.common.reporting_io import assert_reporting_fields_present, write_master_csv


def test_sum_optional_ints_distinguishes_missing_from_zero():
    assert sum_optional_ints(None, None) is None
    assert sum_optional_ints(0, None) == 0
    assert sum_optional_ints(None, 0) == 0
    assert sum_optional_ints(3, 5) == 8
    assert sum_optional_ints(3, None) == 3


def test_shared_phase1_iterations_not_overwritten_by_empty_stage():
    row = {
        "alns_phase1_iterations": 42,
        "runtime_s": 15.0,
        "solver_runtime_s": 15.0,
    }
    # Phase-II style stages: phase1 block from start_routes has no iterations
    stages = {
        "phase1": {"runtime_s": 0.0, "from_shared": True},
        "phase2": {"iterations": 17, "runtime_s": 15.2},
    }
    attach_search_diagnostics(row, stages=stages)
    assert row["alns_phase1_iterations"] == 42
    assert row["alns_phase2_iterations"] == 17
    assert row["alns_iterations"] == 59


def test_shared_phase1_iterations_kwarg_preferred_over_empty_stage():
    row = {"runtime_s": 30.0}
    stages = {"phase1": {"from_shared": True}, "phase2": {"iterations": 10}}
    attach_search_diagnostics(row, stages=stages, shared_phase1_iterations=100)
    assert row["alns_phase1_iterations"] == 100
    assert row["alns_iterations"] == 110


def test_missing_both_phases_preserves_single_loop_total():
    row = {"alns_iterations": 77, "runtime_s": 30.0}
    attach_search_diagnostics(row, stages={})
    assert row["alns_iterations"] == 77
    assert row["alns_phase1_iterations"] is None
    assert row["alns_phase2_iterations"] is None


def test_zero_iterations_not_treated_as_missing():
    row = {}
    stages = {"phase1": {"iterations": 0}, "phase2": {"iterations": 0}}
    attach_search_diagnostics(row, stages=stages)
    assert row["alns_phase1_iterations"] == 0
    assert row["alns_phase2_iterations"] == 0
    assert row["alns_iterations"] == 0


def test_master_csv_keeps_reporting_fields(tmp_path: Path):
    rows = [{
        "instance": "C1_10_1",
        "family": "C1",
        "tw": "original",
        "method": "ALNS-PR",
        "seed": 0,
        "feasible": True,
        "served": 1000,
        "unserved": 0,
        "hard_violations": 0,
        "k": 100,
        "distance": 1.0,
        "duration": 1.0,
        "customer_waiting": 0.0,
        "charging_visits": 1,
        "charged_energy": 1.0,
        "station_waiting": 0.0,
        "charging_time": 1.0,
        "charging_expense": 1.0,
        "runtime_s": 30.0,
        "termination_reason": "time_limit",
        "phase1_s": 30.0,
        "phase2_s": None,
        "phase1_k": 100,
        "phase1_distance": 1.0,
        "phase1_charging_expense": 1.0,
        "k_star": 100,
        "k_frozen_ok": True,
        "phase1_checksum": "abc",
        "hco_mode": None,
        "eps": None,
        "error": None,
        "propulsion_energy": 10.0,
        "enroute_purchased_energy": 1.0,
        "residual_battery_at_depot": 5.0,
        "depot_replenishment_energy": 5.0,
        "mean_energy_per_charging_visit": 1.0,
        "initial_battery_energy": 14.0,
        "energy_balance_ok": True,
        "energy_balance_residual": 0.0,
        "alns_iterations": 12,
        "alns_phase1_iterations": 12,
        "alns_phase2_iterations": None,
        "hco_calls": None,
        "feasible_hco_calls": None,
        "hco_eval_time_s": None,
        "execution_time_s": 30.1,
        "solver_runtime_s": 30.0,
        "reporting_overhead_s": 0.1,
        "end_to_end_runtime_s": 30.1,
        "phase1_runtime_s": 30.0,
        "phase2_runtime_s": None,
        "station_visits": [{"should": "not_appear"}],
    }]
    path = tmp_path / "master.csv"
    write_master_csv(path, rows)
    missing = assert_reporting_fields_present(path)
    assert missing == []
    text = path.read_text(encoding="utf-8")
    assert "propulsion_energy" in text
    assert "solver_runtime_s" in text
    assert "should" not in text  # nested payload stripped
