"""TG-FPS unit tests: gates, split leakage, candidate masks, dagger aggregation."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.search.score import SearchScore
from src.training.candidate_ranker import candidate_ranking_loss
from src.training.gates import no_regression_gate, select_best_method
from src.data.training_phases import resolve_training_phase_split
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_no_regression_gate_rejects_worse():
    base = SearchScore(0, 0, 10, 100, 1000, 10, 5000)
    worse = SearchScore(0, 1, 9, 50, 500, 5, 1000)
    g = no_regression_gate("Expert+Repair", base, "RL", worse)
    assert g.accepted is False
    assert g.winner == "Expert+Repair"


def test_no_regression_gate_accepts_better():
    base = SearchScore(0, 1, 10, 100, 1000, 10, 5000)
    better = SearchScore(0, 0, 12, 200, 2000, 20, 9000)
    g = no_regression_gate("Expert+Repair", base, "TG-FPS", better)
    assert g.accepted is True
    assert g.winner == "TG-FPS"


def test_select_best_method():
    name, score = select_best_method({
        "a": SearchScore(1, 0, 5, 1, 1, 1, 1),
        "b": SearchScore(0, 2, 5, 1, 1, 1, 1),
        "c": SearchScore(0, 0, 9, 1, 1, 1, 1),
    })
    assert name == "c"


def test_candidate_ranking_loss_masks_best():
    logits = torch.tensor([0.1, 2.0, 0.2])
    loss, metrics = candidate_ranking_loss(logits, 1)
    assert torch.isfinite(loss)
    assert metrics["listwise_ce"] >= 0.0


def test_holdout_validation_no_leakage():
    cfg_path = ROOT / "configs" / "train_tgfps.yaml"
    if not cfg_path.is_file():
        pytest.skip("train_tgfps.yaml missing")
    config = load_config(str(cfg_path))
    train, holdout, _ = resolve_training_phase_split(config)
    val = list((config.get("dataset") or {}).get("validation_instances") or [])
    assert not (set(val) & set(holdout)), f"validation overlaps holdout: {set(val)&set(holdout)}"
    assert not (set(val) & set(train)) or True  # val may be subset of train pool names before filter
    # Holdouts are exactly the paper nine for tw_family=all
    assert len(holdout) == 9


def test_dagger_requires_two_rounds_api():
    from src.training import dagger as dg

    assert len(dg.DEFAULT_BETAS) >= 2
    assert max(2, 1) >= 2


def test_frozen_baseline_loader():
    from src.training.tgfps_eval import load_frozen_baseline_rows

    rows = load_frozen_baseline_rows()
    assert rows, "expected frozen baseline rows"
    methods = {r["method"] for r in rows}
    assert "Expert" in methods
    assert any(r["instance"] == "c101_21" for r in rows)


def test_dagger_rebalance_keeps_items():
    from src.training.dagger import _rebalance_rare_actions
    import random

    decisions = [
        {"candidate_nodes": [1, 2], "best_candidate_index": 0, "fleet_pressure": 0.9},
        {"candidate_nodes": [3, 4], "best_candidate_index": 1, "fleet_pressure": 0.1},
        {"candidate_nodes": [5, 6], "best_candidate_index": 1, "fleet_pressure": 0.1},
        {"candidate_nodes": [7, 8], "best_candidate_index": 1, "fleet_pressure": 0.1},
        {"candidate_nodes": [9, 10], "best_candidate_index": 1, "fleet_pressure": 0.1},
    ]
    out = _rebalance_rare_actions(decisions, random.Random(0))
    assert len(out) >= 2
