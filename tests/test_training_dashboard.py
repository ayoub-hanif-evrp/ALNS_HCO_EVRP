"""Tests for static training dashboard figure generation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.training.training_dashboard import (
    generate_training_dashboard,
    load_training_dataframe,
)


def _synthetic_training_df(n_epochs: int = 40, n_instances: int = 4, n_cycles: int = 10) -> pd.DataFrame:
  rng = np.random.default_rng(0)
  instances = [f"r10{i}_21" for i in range(1, n_instances + 1)]
  rows = []
  for epoch in range(n_epochs):
      inst = instances[epoch % n_instances]
      cycle = epoch // n_instances + 1
      rows.append({
          "epoch": epoch,
          "pass": cycle,
          "instance_name": inst,
          "cost": 50000 - epoch * 200 + rng.normal(0, 500),
          "reward": -50 + epoch * 0.2,
          "feasibility_rate": min(1.0, 0.05 + epoch * 0.02),
          "served_rate": min(1.0, 0.3 + epoch * 0.015),
          "actor_loss": 1.5 - epoch * 0.01,
          "critic_loss": 1000 - epoch * 5,
          "total_loss": 1001,
          "entropy": 5.0 - epoch * 0.02,
          "critic_explained_variance": 0.01 * epoch,
          "avg_used_evs": 3 + epoch * 0.01,
          "avg_distance": 1000 - epoch,
          "avg_duration": 500 - epoch * 0.5,
          "avg_charging_time": 100 + rng.normal(0, 5),
          "avg_charging_price": 50,
          "avg_waiting_time": 20,
          "unserved_customers": max(0, 10 - epoch // 4),
          "avg_lateness": max(0, 5 - epoch * 0.1),
          "avg_battery_violation": max(0, 2 - epoch * 0.05),
          "avg_capacity_violation": 0.0,
          "avg_depot_violation": 0.0,
          "avg_duplicate_visits": 0.0,
          "route_balance_cv": 0.5 - epoch * 0.005,
          "charging_station_visits": 2 + epoch * 0.05,
          "min_remaining_battery": 10 + epoch * 0.1,
          "mean_feasible_actions": 50,
          "num_nodes": 121,
          "feasible_action_ratio": 50 / 121,
          "grad_norm": 100 - epoch,
          "cuda_max_mem_allocated_mb": 1200,
      })
  return pd.DataFrame(rows)


def test_load_training_dataframe_filters_holdout(tmp_path):
    csv = tmp_path / "metrics.csv"
    df = _synthetic_training_df(8, 4, 2)
    holdout = ["r104_21"]
    extra = df.iloc[:2].copy()
    extra["instance_name"] = "r104_21"
    full = pd.concat([df, extra], ignore_index=True)
    full.to_csv(csv, index=False)
    loaded = load_training_dataframe(csv_path=csv, holdout_instances=holdout)
    assert "r104_21" not in loaded["instance_name"].values


def test_generate_training_dashboard(tmp_path):
    df = _synthetic_training_df(40, 4, 10)
    report = generate_training_dashboard(df, tmp_path, moving_avg_window=5, save_svg=False)
    assert report.html_path
    assert (tmp_path / "index.html").exists()
    assert (tmp_path / "figures_index.md").exists()
    assert sum(1 for f in report.figures if f.generated) >= 10
    assert (tmp_path / "01_training_objective.png").exists()


def test_empty_dashboard_does_not_crash(tmp_path):
    report = generate_training_dashboard(pd.DataFrame(), tmp_path)
    assert (tmp_path / "index.html").exists()
    assert report.figures == [] or all(not f.generated for f in report.figures)
