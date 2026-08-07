"""Regression: frozen full-9 baseline aggregates must recompute from per-instance rows."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "outputs" / "baselines" / "full9_20260731"
SUMMARIES = BASELINE / "evaluation" / "holdout_full9" / "all_summaries.json"
MANIFEST = BASELINE / "manifest.json"


def _recompute(rows: list[dict]) -> tuple[float, float, float]:
    n = max(len(rows), 1)
    served = sum(float(r["served_ratio"]) for r in rows) / n
    feas = sum(1.0 if r["feasible"] else 0.0 for r in rows) / n
    obj = sum(float(r["objective"]) for r in rows) / n
    return served, feas, obj


@pytest.mark.skipif(not SUMMARIES.is_file(), reason="frozen full-9 baseline missing")
def test_baseline_directory_immutable_marker():
    assert MANIFEST.is_file()
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert man.get("immutable") is True
    assert man.get("baseline_id") == "full9_20260731"


@pytest.mark.skipif(not SUMMARIES.is_file(), reason="frozen full-9 baseline missing")
def test_full9_aggregates_match_per_instance_rows():
    data = json.loads(SUMMARIES.read_text(encoding="utf-8"))
    assert set(data) >= {"Expert", "BC", "RL-best"}
    for method, summary in data.items():
        rows = summary["per_instance"]
        assert len(rows) == 9
        served, feas, obj = _recompute(rows)
        assert abs(served - float(summary["mean_served"])) < 1e-9
        assert abs(feas - float(summary["feasibility_rate"])) < 1e-9
        assert abs(obj - float(summary["mean_objective"])) < 1e-6
    # Paper ceiling snapshot (Expert)
    exp = data["Expert"]
    assert abs(float(exp["feasibility_rate"]) - 2.0 / 3.0) < 1e-9
    assert float(exp["mean_served"]) > 0.98
    assert float(exp["mean_objective"]) < 100_000.0


@pytest.mark.skipif(not MANIFEST.is_file(), reason="frozen full-9 baseline missing")
def test_manifest_matches_summaries():
    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    data = json.loads(SUMMARIES.read_text(encoding="utf-8"))
    for method, summary in data.items():
        m = man["methods"][method]
        assert abs(m["mean_served"] - summary["mean_served"]) < 1e-12
        assert abs(m["feasibility_rate"] - summary["feasibility_rate"]) < 1e-12
        assert abs(m["mean_objective"] - summary["mean_objective"]) < 1e-6
        assert len(m["per_instance"]) == 9
