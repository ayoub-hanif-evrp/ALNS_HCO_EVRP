"""Deterministic C→R→RC class-balanced macro-update schedule."""
from __future__ import annotations

from src.data.instance_class import instance_class
from src.data.interleaved_schedule import build_interleaved_class_schedule


def test_macro_update_triplets_and_wrap():
    train_by_class = {
        "C": ["c102_21", "c103_21"],
        "R": ["r102_21", "r103_21", "r105_21"],
        "RC": ["rc102_21"],
    }
    flat = sorted({n for v in train_by_class.values() for n in v})
    name_to_idx = {n: i for i, n in enumerate(flat)}
    sched, meta = build_interleaved_class_schedule(
        train_by_class, name_to_idx, cycles=2, macro_updates=True,
    )

    assert meta["macro_updates"] is True
    assert meta["updates_per_cycle"] == 3  # max class size
    assert meta["steps_per_class_per_cycle"] == 3
    assert meta["total_updates"] == 6
    assert meta["shuffle"] is False
    assert meta["sampling"] == "macro_class_balanced"

    # Each update is a C/R/RC triplet
    assert len(sched) == 6
    for batch in sched:
        assert len(batch) == 3
        names = [flat[i] for i in batch]
        assert [instance_class(n) for n in names] == ["C", "R", "RC"]

    cycle1 = sched[:3]
    cycle2 = sched[3:6]
    assert cycle1 == cycle2

    first = [flat[i] for i in cycle1[0]]
    assert first[0] == "c102_21"
    assert first[2] == "rc102_21"


def test_legacy_flat_interleave_still_available():
    train_by_class = {
        "C": ["c102_21", "c103_21"],
        "R": ["r102_21", "r103_21", "r105_21"],
        "RC": ["rc102_21"],
    }
    flat = sorted({n for v in train_by_class.values() for n in v})
    name_to_idx = {n: i for i, n in enumerate(flat)}
    sched, meta = build_interleaved_class_schedule(
        train_by_class, name_to_idx, cycles=2, macro_updates=False,
    )
    assert meta["updates_per_cycle"] == 9
    assert meta["total_updates"] == 18
    assert all(isinstance(x, int) for x in sched)
    cycle1 = [flat[i] for i in sched[:9]]
    assert [instance_class(n) for n in cycle1] == ["C", "R", "RC"] * 3


def test_macro_never_blocks_one_class_first():
    train_by_class = {
        "C": ["c102_21", "c105_21", "c106_21"],
        "R": ["r102_21", "r103_21"],
        "RC": ["rc102_21", "rc103_21", "rc105_21", "rc106_21"],
    }
    flat = sorted({n for v in train_by_class.values() for n in v})
    name_to_idx = {n: i for i, n in enumerate(flat)}
    sched, meta = build_interleaved_class_schedule(
        train_by_class, name_to_idx, cycles=20, macro_updates=True,
    )
    assert meta["updates_per_cycle"] == 4
    assert meta["total_updates"] == 20 * 4
    first_batch = [flat[i] for i in sched[0]]
    assert [instance_class(n) for n in first_batch] == ["C", "R", "RC"]
