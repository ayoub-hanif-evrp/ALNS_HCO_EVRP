"""Tests for curriculum building blocks: class detection, batching, Phase 0."""
from __future__ import annotations

import random

from src.data.instance_class import group_indices_by_class, instance_class
from src.training.curriculum import sample_balanced_batch, sample_hard_batch
from src.training.phase0_checks import phase0_passed, run_phase0_checks


def test_instance_class_detection():
    assert instance_class("c101_21") == "C"
    assert instance_class("c101C5") == "C"
    assert instance_class("r201C10") == "R"
    assert instance_class("r110_21") == "R"
    assert instance_class("rc108C15") == "RC"
    assert instance_class("rc201_21") == "RC"
    # rc must win over the r prefix.
    assert instance_class("RC205") == "RC"


def test_group_indices_by_class():
    names = ["c101_21", "r201_21", "rc108C5", "c205C10"]
    groups = group_indices_by_class(names)
    assert groups["C"] == [0, 3]
    assert groups["R"] == [1]
    assert groups["RC"] == [2]


def test_balanced_batch_respects_proportions():
    rng = random.Random(0)
    groups = {"C": list(range(100)), "R": list(range(100, 200)), "RC": []}
    batch = sample_balanced_batch(rng, groups, ["C", "R"], {"C": 0.5, "R": 0.5}, batch_size=8)
    assert len(batch) == 8
    n_c = sum(1 for i in batch if i < 100)
    n_r = sum(1 for i in batch if i >= 100)
    assert n_c == 4 and n_r == 4


def test_balanced_batch_redistributes_empty_class():
    rng = random.Random(0)
    groups = {"C": list(range(50)), "R": [], "RC": []}
    batch = sample_balanced_batch(rng, groups, ["C", "R"], {"C": 0.5, "R": 0.5}, batch_size=6)
    assert len(batch) == 6
    assert all(i < 50 for i in batch)  # all drawn from C since R is empty


def test_hard_batch_split():
    rng = random.Random(0)
    hard = list(range(10))
    normal = list(range(100, 200))
    batch = sample_hard_batch(rng, hard, normal, batch_size=8, hard_share=0.5)
    assert len(batch) == 8
    n_hard = sum(1 for i in batch if i < 10)
    assert n_hard == 4


def test_cyclic_schedule_order_and_length():
    groups = {"C": [0, 1], "R": [10, 11, 12], "RC": [20]}
    from src.training.curriculum import build_cyclic_schedule, cyclic_total_steps

    sched = build_cyclic_schedule(groups, class_order=["C", "R", "RC"], cycles=5, rng=random.Random(0))
    assert len(sched) == cyclic_total_steps(groups, class_order=["C", "R", "RC"], cycles=5) == 30
    assert [s["instance_class"] for s in sched[:6]] == ["C", "C", "R", "R", "R", "RC"]
    assert sched[0]["cycle"] == 1 and sched[-1]["cycle"] == 5
    assert sched[6]["instance_idx"] == 0  # second cycle starts again with C


def test_rc_size_schedule():
    from src.data.instance_class import group_indices_by_class
    from src.training.curriculum import build_rc_size_schedule

    names = [
        "rc108C5", "rc201C10", "rc108_21", "rc201_21",
    ]
    groups = group_indices_by_class(names)
    sched = build_rc_size_schedule(names, groups, cycles_small=2, cycles_full=1, rng=random.Random(0))
    # small: idx 0,1 × 2 cycles = 4; full: idx 2,3 × 1 = 2 → total 6
    assert len(sched) == 6
    assert all(s["instance_class"] == "RC" for s in sched)
    assert sched[0]["size_tier"] == "small"
    assert sched[-1]["size_tier"] == "full"


def test_progressive_schedule_order_and_length():
    from src.data.instance_class import group_indices_by_class
    from src.training.curriculum import build_progressive_schedule, progressive_total_steps

    names = [
        "c101C5", "c101_21",
        "rc108C5", "rc108_21",
        "r201C10", "r201_21",
    ]
    groups = group_indices_by_class(names)
    sched = build_progressive_schedule(
        names,
        groups,
        class_order=["C", "RC", "R"],
        cycles_small=2,
        cycles_full=1,
        rng=random.Random(0),
    )
    expected = progressive_total_steps(
        names, groups, class_order=["C", "RC", "R"], cycles_small=2, cycles_full=1,
    )
    assert len(sched) == expected == 9
    assert [s["tier_label"] for s in sched] == [
        "C_small", "C_small",
        "C_full",
        "RC_small", "RC_small",
        "RC_full",
        "R_small", "R_small",
        "R_full",
    ]
    assert sched[0]["instance_class"] == "C" and sched[-1]["instance_class"] == "R"


def test_class_size_schedule():
    from src.data.instance_class import group_indices_by_class
    from src.training.curriculum import build_class_size_schedule

    names = ["c101C5", "c103C10", "c101_21"]
    groups = group_indices_by_class(names)
    sched = build_class_size_schedule(names, groups, "C", cycles_small=2, cycles_full=1, rng=random.Random(0))
    assert len(sched) == 5  # 2 small × 2 + 1 full × 1
    assert all(s["instance_class"] == "C" for s in sched)


def test_progressive_blocks():
    from src.data.instance_class import group_indices_by_class
    from src.training.curriculum import build_progressive_schedule, progressive_blocks

    names = [
        "c101C5", "c101_21",
        "rc108C5", "rc108_21",
        "r201C10", "r201_21",
    ]
    groups = group_indices_by_class(names)
    sched = build_progressive_schedule(
        names,
        groups,
        class_order=["C", "RC", "R"],
        cycles_small=2,
        cycles_full=1,
        rng=random.Random(0),
    )
    blocks = progressive_blocks(sched, class_order=["C", "RC", "R"])
    assert len(blocks) == 3
    assert blocks[0]["class"] == "C"
    assert len(blocks[0]["small"]) == 2
    assert len(blocks[0]["full"]) == 1

    hanif_names = ["C1_10_1", "C2_10_1", "RC1_10_1", "RC2_10_1"]
    hanif_groups = group_indices_by_class(hanif_names)
    hanif_sched = build_progressive_schedule(
        hanif_names, hanif_groups, class_order=["C", "RC"], cycles_small=1, cycles_full=1, rng=random.Random(0),
    )
    assert len(hanif_sched) == 4


def test_phase0_checks_pass(dataset_path):
    from src.training.phase0_checks import pick_check_instance

    instance = pick_check_instance(str(dataset_path))
    results = run_phase0_checks(instance)
    assert phase0_passed(results), [
        (r.name, r.detail) for r in results if not r.passed
    ]
