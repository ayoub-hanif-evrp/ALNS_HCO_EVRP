"""Tests for Schneider RC _21 split discovery and validation."""
from __future__ import annotations

import pytest

from src.data.rc21_split import discover_rc21_instances, resolve_rc21_split
from tests.conftest import DATASET_PATH

ALL_RC21 = [
    "rc101_21", "rc102_21", "rc103_21", "rc104_21", "rc105_21", "rc106_21", "rc107_21", "rc108_21",
    "rc201_21", "rc202_21", "rc203_21", "rc204_21", "rc205_21", "rc206_21", "rc207_21", "rc208_21",
]
EVAL = ["rc101_21", "rc104_21", "rc208_21"]
TRAIN = [n for n in ALL_RC21 if n not in EVAL]


@pytest.fixture(scope="module")
def rc21_names():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    return discover_rc21_instances(DATASET_PATH)


def test_discover_rc21_count(rc21_names):
    assert len(rc21_names) == 16
    assert rc21_names == sorted(ALL_RC21)


def test_resolve_auto_split_13_train():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    cfg = {
        "dataset": {
            "path": str(DATASET_PATH),
            "auto_discover_rc21": True,
            "expected_train_count": 13,
            "expected_eval_count": 3,
            "fail_if_train_count_mismatch": True,
            "eval_instances": EVAL,
        }
    }
    train, eval_names, meta = resolve_rc21_split(cfg)
    assert train == TRAIN
    assert eval_names == EVAL
    assert meta["num_train"] == 13


def test_fails_when_expecting_20_train():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    cfg = {
        "dataset": {
            "path": str(DATASET_PATH),
            "auto_discover_rc21": True,
            "expected_train_count": 20,
            "expected_eval_count": 3,
            "fail_if_train_count_mismatch": True,
            "eval_instances": EVAL,
        }
    }
    with pytest.raises(ValueError, match="SCHNEIDER RC _21 SPLIT ERROR"):
        resolve_rc21_split(cfg)
