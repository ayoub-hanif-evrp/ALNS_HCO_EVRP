"""Deterministic C→R→RC interleaving for full training."""
from __future__ import annotations

from src.data.instance_class import instance_class
from src.data.interleaved_schedule import build_interleaved_class_schedule


def test_interleave_order_and_wrap():
    train_by_class = {
        "C": ["c102_21", "c103_21"],
        "R": ["r102_21", "r103_21", "r105_21"],
        "RC": ["rc102_21"],
    }
    flat = sorted({n for v in train_by_class.values() for n in v})
    name_to_idx = {n: i for i, n in enumerate(flat)}
    sched, meta = build_interleaved_class_schedule(train_by_class, name_to_idx, cycles=2)

    assert meta["updates_per_cycle"] == 9  # 3 * max(2,3,1)
    assert meta["steps_per_class_per_cycle"] == 3
    assert meta["total_updates"] == 18
    assert meta["shuffle"] is False

    cycle1 = [flat[i] for i in sched[:9]]
    cycle2 = [flat[i] for i in sched[9:18]]
    assert cycle1 == cycle2

    classes = [instance_class(n) for n in cycle1]
    assert classes == ["C", "R", "RC"] * 3
    assert classes.count("C") == classes.count("R") == classes.count("RC") == 3

    assert cycle1[0] == "c102_21"
    assert cycle1[3] == "c103_21"
    assert cycle1[6] == "c102_21"
    assert cycle1[2] == cycle1[5] == cycle1[8] == "rc102_21"


def test_interleave_never_blocks_one_class_first():
    train_by_class = {
        "C": ["c102_21", "c105_21", "c106_21"],
        "R": ["r102_21", "r103_21"],
        "RC": ["rc102_21", "rc103_21", "rc105_21", "rc106_21"],
    }
    flat = sorted({n for v in train_by_class.values() for n in v})
    name_to_idx = {n: i for i, n in enumerate(flat)}
    _, meta = build_interleaved_class_schedule(train_by_class, name_to_idx, cycles=20)
    names = meta["one_cycle_names"]
    assert [instance_class(n) for n in names[:3]] == ["C", "R", "RC"]
    # Not "all C then all R..."
    assert not all(instance_class(n) == "C" for n in names[:3])
    upc = meta["updates_per_cycle"]
    assert upc == 3 * 4  # max class size = 4
    assert meta["cycles"] == 20
    assert meta["total_updates"] == 20 * upc
