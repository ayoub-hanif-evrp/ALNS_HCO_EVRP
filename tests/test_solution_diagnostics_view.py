"""Tests for live dashboard solution-behavior section builder."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.webapp.solution_diagnostics_data import (
    build_solution_behavior_section,
    figures_dir_from_config,
)
from tests.test_training_dashboard import _synthetic_training_df


def test_build_solution_behavior_section_with_table(tmp_path):
    df = _synthetic_training_df(40, 20, 2)
    figures_dir = tmp_path / "training"
    figures_dir.mkdir()
    section = build_solution_behavior_section(
        figures_dir,
        df,
        traces_dir=tmp_path / "traces",
        cfg={
            "dashboard": {
                "save_representative_traces": True,
                "live_random_diagnostics": False,
            },
        },
    )
    assert section is not None
    assert section["type"] == "solution_diagnostics"
    assert section["table_html"] is not None
    assert len(section["groups"]) >= 8
    assert section.get("figures_pending") in (True, False)


def test_build_solution_behavior_section_live_enabled(tmp_path):
    df = _synthetic_training_df(10, 20, 1)
    figures_dir = tmp_path / "training"
    figures_dir.mkdir()
    cfg = {
        "dashboard": {"live_random_diagnostics": True, "save_representative_traces": False},
        "outputs": {"checkpoints": str(tmp_path / "ckpt")},
        "seed": 0,
    }
    section = build_solution_behavior_section(figures_dir, df, cfg=cfg)
    assert section is not None
    assert section["live"]["enabled"] is True


def test_build_solution_behavior_section_with_figures(tmp_path):
    df = _synthetic_training_df(20, 20, 1)
    figures_dir = tmp_path / "training"
    figures_dir.mkdir()
    png = figures_dir / "20_solution_map_random_final_cycle.png"
    png.write_bytes(b"fake")
    section = build_solution_behavior_section(figures_dir, df, cfg={"dashboard": {}})
    assert section is not None
    assert section["any_figure"] is True
    assert section["figures_pending"] is False
    maps = next(g for g in section["groups"] if g["id"] == "route_maps")
    assert maps["items"][0]["exists"] is True


def test_figures_dir_from_config():
    cfg = {"outputs": {"figures": "outputs/figures", "root": "outputs"}}
    assert figures_dir_from_config(cfg).as_posix().endswith("figures/training")
