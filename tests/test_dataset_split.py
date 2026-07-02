"""Tests for explicit train/eval splits and flat sequential schedules."""
from __future__ import annotations

import pytest

from src.data.evrptw_dataset import split_dataset
from src.data.instance_class import instance_class
from src.training.flat_schedule import build_sequential_schedule
from tests.conftest import DATASET_PATH

SCHNEIDER_R21_TRAIN = [
    "r102_21", "r103_21", "r105_21", "r106_21", "r107_21", "r108_21", "r109_21",
    "r110_21", "r111_21", "r112_21", "r201_21", "r202_21", "r203_21", "r204_21",
    "r205_21", "r206_21", "r207_21", "r208_21", "r210_21", "r211_21",
]
SCHNEIDER_R21_EVAL = ["r101_21", "r104_21", "r209_21"]


@pytest.fixture(scope="module")
def schneider_r21_split():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    return split_dataset(
        DATASET_PATH,
        train_ratio=0.8,
        seed=42,
        train_filter=["_21"],
        eval_filter=["_21"],
        train_instances=SCHNEIDER_R21_TRAIN,
        eval_instances=SCHNEIDER_R21_EVAL,
        load_capacity=100.0,
    )


def test_explicit_split_counts_and_disjoint(schneider_r21_split):
    train_ds, eval_ds = schneider_r21_split
    assert len(train_ds) == 20
    assert len(eval_ds) == 3
    assert set(train_ds.names) == set(SCHNEIDER_R21_TRAIN)
    assert set(eval_ds.names) == set(SCHNEIDER_R21_EVAL)
    assert not set(train_ds.names) & set(eval_ds.names)


def test_explicit_split_all_r_class(schneider_r21_split):
    train_ds, eval_ds = schneider_r21_split
    assert all(instance_class(n) == "R" for n in train_ds.names)
    assert all(instance_class(n) == "R" for n in eval_ds.names)


def test_sequential_schedule_length_and_order():
    schedule = build_sequential_schedule(20, 10, shuffle_within_pass=False)
    assert len(schedule) == 200
    assert schedule[:20] == list(range(20))
    assert schedule[20:40] == list(range(20))


def test_explicit_split_rejects_unknown_name():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    with pytest.raises(ValueError, match="train instance not found"):
        split_dataset(
            DATASET_PATH,
            train_ratio=0.8,
            seed=42,
            train_instances=["not_a_real_instance_21"],
            load_capacity=100.0,
        )


def test_explicit_split_rejects_overlap():
    if not DATASET_PATH.is_dir():
        pytest.skip("Schneider dataset not available")
    with pytest.raises(ValueError, match="overlap"):
        split_dataset(
            DATASET_PATH,
            train_ratio=0.8,
            seed=42,
            train_instances=SCHNEIDER_R21_TRAIN[:5],
            eval_instances=SCHNEIDER_R21_TRAIN[:3],
            load_capacity=100.0,
        )
