"""Tests for maps dashboard data helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.webapp.maps_dashboard_data import (
    instance_progression_records,
    list_training_instances,
    resolve_checkpoint_for_epoch,
    snapshot_dir,
    snapshot_is_cached,
)
from tests.test_training_dashboard import _synthetic_training_df


def test_list_instances_and_progression():
    df = _synthetic_training_df(40, 5, 2)
    inst = list_training_instances(df)[0]
    recs = instance_progression_records(df, inst)
    assert len(recs) >= 2
    assert "epoch" in recs[0]


def test_resolve_checkpoint_for_epoch(tmp_path):
    hist = tmp_path / "history"
    hist.mkdir(parents=True)
    (hist / "epoch_10.pt").write_bytes(b"x")
    (hist / "epoch_20.pt").write_bytes(b"x")
    cfg = {"outputs": {"checkpoints": str(tmp_path)}}
    path, ep, note = resolve_checkpoint_for_epoch(cfg, 15)
    assert path is not None
    assert ep == 10
    assert "10" in note


def test_snapshot_cache(tmp_path):
    inst = "rc102_21"
    out = snapshot_dir(tmp_path, inst, 5)
    out.mkdir(parents=True)
    (out / "live_solution_map.png").write_bytes(b"x")
    (out / "snapshot_meta.json").write_text("{}", encoding="utf-8")
    assert snapshot_is_cached(out) is False
    for name, _, _ in __import__(
        "src.training.solution_diagnostics", fromlist=["LIVE_PLOT_SPECS"]
    ).LIVE_PLOT_SPECS:
        (out / name).write_bytes(b"x")
    assert snapshot_is_cached(out) is True
