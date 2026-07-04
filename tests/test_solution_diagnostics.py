"""Tests for solution-behavior diagnostic plots and episode selection."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.training.solution_diagnostics import (
    SOLUTION_SECTION,
    build_final_cycle_table_html,
    generate_solution_diagnostic_figures,
    select_representative_episodes,
    TraceSelection,
)
from tests.test_training_dashboard import _synthetic_training_df


def _synthetic_trace() -> dict:
    return {
        "meta": {
            "instance_name": "rc102_21",
            "epoch": 99,
            "cycle": 5,
            "objective": 420000.0,
            "served_ratio": 0.75,
            "unserved_customers": 12,
            "feasible": False,
            "used_evs": 2,
            "total_distance": 5100.0,
            "total_charging_time": 45.0,
            "customer_lateness": 1.5,
            "depot_lateness": 0.0,
            "battery_violation": 0.0,
        },
        "nodes": [
            {"id": 0, "type": "depot", "x": 0.0, "y": 0.0},
            {"id": 1, "type": "customer", "x": 10.0, "y": 5.0},
            {"id": 2, "type": "customer", "x": 20.0, "y": 0.0},
            {"id": 3, "type": "station", "x": 15.0, "y": 10.0},
        ],
        "served_mask": [False, True, False, False],
        "routes": [
            {
                "ev_id": 0,
                "route_nodes": [0, 1, 3, 0],
                "arrival_times": [0.0, 5.0, 20.0, 35.0],
                "departure_times": [0.0, 8.0, 25.0, 40.0],
                "battery_trace": [100.0, 80.0, 60.0, 90.0],
                "capacity_trace": [100.0, 90.0, 90.0, 90.0],
                "route_distance": 1200.0,
                "total_charging_time": 15.0,
                "total_waiting_time": 2.0,
                "charging_actions": [{"departure": 25.0, "charging_time": 5.0}],
                "visits": [
                    {"node": 0, "arrival": 0.0, "departure": 0.0},
                    {"node": 1, "arrival": 5.0, "departure": 8.0},
                    {"node": 3, "arrival": 20.0, "departure": 25.0},
                    {"node": 0, "arrival": 35.0, "departure": 40.0},
                ],
            },
        ],
        "decisions": [
            {
                "decision_index": i,
                "action_type": t,
                "customer_served": t == "customer",
                "service_slack": 10.0 - i if t == "customer" else None,
                "unserved_after": max(0, 2 - i),
                "expired_after": 0,
                "feasible_customers": max(0, 2 - i),
                "feasible_stations": 1,
                "feasible_depot": 1,
                "masked_by_time_window": 0,
                "masked_by_battery": 0,
                "masked_by_capacity": 0,
            }
            for i, t in enumerate(["customer", "station", "depot_recharge", "depot_finish"])
        ],
    }


@pytest.fixture
def trace_selections() -> list[TraceSelection]:
    trace = _synthetic_trace()
    tags = [
        "random_final_cycle",
        "best_final_cycle",
        "worst_final_cycle",
        "best_overall",
    ]
    return [TraceSelection(tag=tag, row={"epoch": 1, "cycle": 5}, trace=trace) for tag in tags]


def test_select_representative_episodes():
    df = _synthetic_training_df(100, 20, 5)
    cfg = {
        "num_random_solution_maps": 1,
        "include_best_final_cycle": True,
        "include_worst_final_cycle": True,
        "include_best_overall": True,
    }
    sels = select_representative_episodes(df, cfg, seed=0)
    tags = {s.tag for s in sels}
    assert "random_final_cycle" in tags
    assert "best_final_cycle" in tags
    assert "worst_final_cycle" in tags
    assert len(sels) >= 3


def test_generate_solution_diagnostic_figures(tmp_path, trace_selections):
    df = _synthetic_training_df(100, 20, 5)
    figures, table_html = generate_solution_diagnostic_figures(
        trace_selections, tmp_path, df, save_svg=False,
    )
    generated = [f for f in figures if f.generated]
    assert (tmp_path / "20_solution_map_random_final_cycle.png").exists()
    assert (tmp_path / "21_solution_map_best_final_cycle.png").exists()
    assert (tmp_path / "24_route_timeline_random_final_cycle.png").exists()
    assert (tmp_path / "27_battery_trajectory_random_final_cycle.png").exists()
    assert (tmp_path / "33_action_sequence_random.png").exists()
    assert (tmp_path / "44_final_cycle_instance_table.html").exists()
    assert "Final-cycle instances" in table_html
    assert any(f.section.startswith(SOLUTION_SECTION) for f in figures)
    assert len(generated) >= 10
    names = {f.filename for f in generated}
    assert "20_solution_map_random_final_cycle.png" in names
    assert "44_final_cycle_instance_table.html" in names


def test_missing_trace_skips_without_crash(tmp_path):
    df = _synthetic_training_df(20, 20, 1)
    figures, _ = generate_solution_diagnostic_figures([], tmp_path, df, save_svg=False)
    skipped = [f for f in figures if not f.generated]
    assert skipped
    assert all(f.skip_reason for f in skipped)
    assert (tmp_path / "44_final_cycle_instance_table.html").exists()


def test_build_final_cycle_table_html():
    df = _synthetic_training_df(40, 20, 2)
    html = build_final_cycle_table_html(df)
    assert "<table" in html
    assert "rc" in html
