"""Tests for live random-instance dashboard diagnostics helpers."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.webapp.live_solution_diagnostics import (
    build_live_groups,
    read_live_state,
    should_refresh_live,
    write_live_state,
)


def test_should_refresh_live_empty_state():
    cfg = {"dashboard": {"live_random_diagnostics": True, "live_random_interval_sec": 45}}
    assert should_refresh_live(cfg, pd.DataFrame(), {}) is True


def test_should_refresh_respects_interval():
    cfg = {"dashboard": {"live_random_diagnostics": True, "live_random_interval_sec": 9999}}
    state = {"generated_at": 1e12, "epoch": 2}
    df = pd.DataFrame([{"epoch": 2}])
    assert should_refresh_live(cfg, df, state) is False


def test_build_live_groups(tmp_path):
    live_dir = tmp_path / "live"
    live_dir.mkdir(parents=True)
    png = live_dir / "live_solution_map.png"
    png.write_bytes(b"x")
    write_live_state(tmp_path, {"generated_at": 1.0, "instance_name": "rc102_21"})
    groups = build_live_groups(tmp_path)
    maps = next(g for g in groups if g["id"] == "live_snapshot")
    first = maps["items"][0]
    assert first["file"] == "live/live_solution_map.png"
    assert first["exists"] is True
    assert read_live_state(tmp_path)["instance_name"] == "rc102_21"
