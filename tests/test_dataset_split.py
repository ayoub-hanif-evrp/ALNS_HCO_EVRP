"""Tests for explicit train/eval splits and flat sequential schedules."""
from __future__ import annotations

import pytest

from src.data.evrptw_dataset import split_dataset
from src.data.instance_class import instance_class
from src.data.rc21_split import resolve_rc21_split
from src.training.config_validation import validate_training_config
from src.training.flat_schedule import build_sequential_schedule
from tests.conftest import DATASET_PATH

SCHNEIDER_RC21_EVAL = ["rc101_21", "rc104_21", "rc208_21"]
ALL_RC21 = [
    "rc101_21", "rc102_21", "rc103_21", "rc104_21", "rc105_21", "rc106_21", "rc107_21", "rc108_21",
    "rc201_21", "rc202_21", "rc203_21", "rc204_21", "rc205_21", "rc206_21", "rc207_21", "rc208_21",
]
SCHNEIDER_RC21_TRAIN = [n for n in ALL_RC21 if n not in SCHNEIDER_RC21_EVAL]


@pytest.fixture(scope="module")
def schneider_rc21_split():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    cfg = {
        "dataset": {
            "path": str(DATASET_PATH),
            "auto_discover_rc21": True,
            "expected_train_count": 13,
            "expected_eval_count": 3,
            "fail_if_train_count_mismatch": True,
            "eval_instances": SCHNEIDER_RC21_EVAL,
        }
    }
    train_names, eval_names, _ = resolve_rc21_split(cfg)
    return split_dataset(
        DATASET_PATH,
        train_ratio=0.8,
        seed=42,
        train_filter=["_21"],
        eval_filter=["_21"],
        class_filter=["RC"],
        train_instances=train_names,
        eval_instances=eval_names,
        load_capacity=100.0,
    )


def test_explicit_rc_split_counts_and_disjoint(schneider_rc21_split):
    train_ds, eval_ds = schneider_rc21_split
    assert len(train_ds) == 13
    assert len(eval_ds) == 3
    assert set(train_ds.names) == set(SCHNEIDER_RC21_TRAIN)
    assert set(eval_ds.names) == set(SCHNEIDER_RC21_EVAL)
    assert not set(train_ds.names) & set(eval_ds.names)


def test_explicit_split_all_rc_class(schneider_rc21_split):
    train_ds, eval_ds = schneider_rc21_split
    assert all(instance_class(n) == "RC" for n in train_ds.names)
    assert all(instance_class(n) == "RC" for n in eval_ds.names)


def test_rejects_non_rc_instance():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    with pytest.raises(ValueError, match="not in allowed"):
        split_dataset(
            DATASET_PATH,
            train_ratio=0.8,
            seed=42,
            class_filter=["RC"],
            train_instances=["r102_21"],
            eval_instances=["rc101_21"],
            load_capacity=100.0,
        )


def test_sequential_schedule_length_and_order():
    schedule = build_sequential_schedule(13, 50, shuffle_within_pass=False)
    assert len(schedule) == 650
    assert schedule[:13] == list(range(13))


def test_validate_safe_config():
    cfg = {
        "dataset": {
            "train_instances": SCHNEIDER_RC21_TRAIN,
            "eval_instances": SCHNEIDER_RC21_EVAL,
            "expected_train_count": 13,
        },
        "training": {"cycles": 50},
        "rl": {"use_counterfactual_critic": False, "use_pomo": False},
        "model": {"use_edge_aware_encoder": False},
    }
    validate_training_config(cfg)


def test_validate_rejects_all_advanced_modules():
    cfg = {
        "dataset": {"train_instances": SCHNEIDER_RC21_TRAIN, "eval_instances": SCHNEIDER_RC21_EVAL},
        "rl": {"use_counterfactual_critic": True, "use_pomo": True},
        "model": {"use_edge_aware_encoder": True},
        "debug": {},
    }
    with pytest.raises(ValueError, match="Refusing"):
        validate_training_config(cfg)
