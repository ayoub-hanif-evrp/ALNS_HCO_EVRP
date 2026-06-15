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


def test_phase0_checks_pass(dataset_path):
    from src.training.phase0_checks import pick_check_instance

    instance = pick_check_instance(str(dataset_path))
    results = run_phase0_checks(instance)
    assert phase0_passed(results), [
        (r.name, r.detail) for r in results if not r.passed
    ]
