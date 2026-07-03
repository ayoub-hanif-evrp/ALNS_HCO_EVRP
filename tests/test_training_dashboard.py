"""Tests for focused training diagnostic dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.training_dashboard import (
    REQUIRED_RC21_TRAIN_COUNT,
    generate_training_dashboard,
    load_training_dataframe,
    resolve_dashboard_ma_window,
)


def _synthetic_training_df(n_epochs: int = 100, n_instances: int = 20, n_cycles: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    instances = [f"rc10{i}_21" for i in range(1, 9)] + [f"rc20{i}_21" for i in range(1, 13)]
    instances = instances[:n_instances]
    rows = []
    for epoch in range(n_epochs):
        inst = instances[epoch % n_instances]
        cycle = epoch // n_instances + 1
        rows.append({
            "epoch": epoch,
            "cycle": cycle,
            "pass": cycle,
            "instance_name": inst,
            "final_objective": 500000 - epoch * 500 + rng.normal(0, 1000),
            "cost": 500000 - epoch * 500,
            "raw_objective_return": -500 + epoch * 0.5,
            "shaped_training_return": -400 + epoch * 0.6,
            "feasibility_rate": min(1.0, 0.05 + epoch * 0.01),
            "served_rate": min(1.0, 0.4 + epoch * 0.006),
            "unserved_customers": max(0, 50 - epoch // 2),
            "served_bonus_total": epoch * 0.1,
            "early_depot_penalty_total": max(0, 5 - epoch * 0.05),
            "terminal_unserved_penalty_scaled": max(0, 400000 - epoch * 2000),
            "mean_pen_infeasible": 100000.0,
            "step_penalty_total": 0.01,
            "terminal_penalty_coef": min(1.0, 0.25 + cycle * 0.04),
            "relative_objective_mismatch": 0.01,
            "fraction_customer_actions": min(0.7, 0.3 + epoch * 0.004),
            "fraction_station_actions": 0.2,
            "fraction_depot_actions": max(0.05, 0.3 - epoch * 0.003),
            "early_depot_returns": max(0, 10 - epoch // 5),
            "actor_loss": 1.5 - epoch * 0.005,
            "critic_loss": 1000 - epoch * 2,
            "critic_explained_variance": 0.001 * epoch,
            "advantage": rng.normal(0, 1),
            "advantage_std": 1.0,
            "actor_grad_norm": 0.5,
            "critic_grad_norm": 0.3,
            "actor_param_delta_norm": 0.01,
            "critic_param_delta_norm": 0.008,
            "avg_used_evs": 5,
            "avg_distance": 5000,
            "avg_charging_time": 100,
            "charging_station_visits": 3,
        })
    return pd.DataFrame(rows)


def test_load_training_dataframe_filters_holdout(tmp_path):
    csv = tmp_path / "metrics.csv"
    df = _synthetic_training_df(40, 20, 2)
    holdout = ["rc101_21"]
    extra = df.iloc[:2].copy()
    extra["instance_name"] = "rc101_21"
    full = pd.concat([df, extra], ignore_index=True)
    full.to_csv(csv, index=False)
    loaded = load_training_dataframe(csv_path=csv, holdout_instances=holdout)
    assert "rc101_21" not in loaded["instance_name"].values


def test_resolve_ma_window_requires_20_instances():
    cfg = {"dataset": {"train_instances": [f"rc{i}_21" for i in range(13)]}}
    df = _synthetic_training_df(20, 13, 1)
    with pytest.raises(ValueError, match="20"):
        resolve_dashboard_ma_window(cfg, df)


def test_generate_focused_dashboard(tmp_path):
    df = _synthetic_training_df(100, REQUIRED_RC21_TRAIN_COUNT, 5)
    report = generate_training_dashboard(
        df, tmp_path, moving_avg_window=REQUIRED_RC21_TRAIN_COUNT,
        n_train_instances=REQUIRED_RC21_TRAIN_COUNT, save_svg=False,
    )
    assert report.html_path
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "figures_index.md").exists()
    assert (tmp_path / "01_objective.png").exists()
    assert (tmp_path / "03_cycle_objective.png").exists()
    assert sum(1 for f in report.figures if f.generated) >= 12
    assert sum(1 for f in report.figures if f.required) >= 10


def test_empty_dashboard_does_not_crash(tmp_path):
    report = generate_training_dashboard(
        pd.DataFrame(), tmp_path, moving_avg_window=20, n_train_instances=20,
    )
    assert (tmp_path / "index.html").exists()
    assert report.figures == []
