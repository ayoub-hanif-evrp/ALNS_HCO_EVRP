"""Tests for live NiceGUI dashboard data prep and chart specs."""
from __future__ import annotations

import pandas as pd

from src.webapp.dashboard_charts import build_chart_figure, build_flat_training_sections, get_training_chart_sections
from src.webapp.dashboard_data import filter_training_instances, prepare_metrics_df, training_plan_from_config


def _sample_row(epoch: int, instance: str, pass_num: int) -> dict:
    return {
        "epoch": epoch,
        "pass": pass_num,
        "instance_name": instance,
        "cost": 100000 - epoch * 100,
        "reward": -100 + epoch * 0.1,
        "feasibility_rate": 0.1,
        "served_rate": 0.5,
        "entropy": 3.0,
        "actor_loss": 1.0,
        "critic_loss": 500.0,
        "avg_used_evs": 5.0,
        "route_balance_cv": 0.3,
        "unserved_customers": 10.0,
    }


def test_filter_training_instances_excludes_holdout():
    df = pd.DataFrame([
        _sample_row(0, "r102_21", 1),
        _sample_row(1, "r104_21", 1),
    ])
    cfg = {
        "dataset": {
            "train_instances": ["r102_21"],
            "eval_instances": ["r104_21"],
        }
    }
    filtered = filter_training_instances(df, cfg)
    assert list(filtered["instance_name"]) == ["r102_21"]


def test_flat_training_sections_and_charts():
    rows = [_sample_row(i, f"r10{i % 3 + 2}_21", i // 3 + 1) for i in range(12)]
    df = prepare_metrics_df(pd.DataFrame(rows), window=3)
    cfg = {
        "training": {"passes": 10, "dashboard_ma_window": 3, "method": "actor_critic"},
        "dataset": {"train_instances": [f"r10{i}_21" for i in range(2, 5)]},
    }
    plan = training_plan_from_config(cfg)
    assert plan["training_mode"] == "flat"
    sections = get_training_chart_sections(df, plan)
    assert sections[0]["id"] == "sample_efficiency"
    assert any(s["id"] == "solution_quality" for s in sections)
    assert any(s["id"] == "rl_optimization" for s in sections)
    objective_spec = next(
        c for s in sections for c in s["charts"]
        if c.get("title") in {"Objective J over epochs", "Training objective over epochs"}
    )
    fig = build_chart_figure(df, plan, objective_spec)
    assert fig is not None
    assert len(fig.data) >= 1


def test_reward_moving_average_line():
    rows = [_sample_row(i, "r102_21", i // 3 + 1) for i in range(12)]
    cfg = {"training": {"dashboard_ma_window": 3}}
    plan = {"training_mode": "flat", "ma_window": 3}
    df = prepare_metrics_df(pd.DataFrame(rows), window=3, cfg=cfg)
    spec = {"type": "series_ma", "col": "reward", "title": "Return", "ylabel": "R"}
    fig = build_chart_figure(df, plan, spec)
    assert fig is not None
    assert len(fig.data) == 2
    assert any(getattr(t, "mode", "") == "lines" for t in fig.data)


def test_build_flat_sections_empty_df():
    plan = training_plan_from_config({"training": {"passes": 10}, "dataset": {"train_instances": ["r102_21"]}})
    sections = build_flat_training_sections(pd.DataFrame(), plan)
    assert len(sections) == 10
    assert {s["id"] for s in sections} >= {
        "sample_efficiency",
        "solution_quality",
        "rl_optimization",
        "per_instance",
    }
