"""Tests for fixed mini curriculum (15 fixed instances)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from src.training.fixed_mini import (
    INSTANCES_PER_CLASS,
    all_fixed_names,
    assert_disjoint_train_eval,
    build_fixed_mini_schedule,
    fixed_train_set_hash,
    load_fixed_train_manifest,
    manifest_fixed_by_class,
    resolve_fixed_train_instances,
    total_expected_updates,
    write_fixed_train_manifest,
)


def _synthetic_schneider_names() -> list[str]:
    names = []
    for i in range(1, 8):
        names.append(f"c10{i}_21")
        names.append(f"rc10{i}_21")
        names.append(f"r10{i}_21")
    return names


def test_resolve_default_first_five():
    names = _synthetic_schneider_names()
    config = {"dataset": {"fixed_train_instances": {}}}
    selected = resolve_fixed_train_instances(names, config, seed=42)
    assert set(selected.keys()) == {"C", "RC", "R"}
    for cls in ("C", "RC", "R"):
        assert len(selected[cls]) == INSTANCES_PER_CLASS
        assert all(n.startswith(f"{cls[0]}" if cls != "RC" else "RC") for n in selected[cls])
    assert selected["C"] == ["c101_21", "c102_21", "c103_21", "c104_21", "c105_21"]


def test_schedule_same_set_every_cycle():
    names = _synthetic_schneider_names()
    config = {"dataset": {"fixed_train_instances": {}}}
    fixed = resolve_fixed_train_instances(names, config, seed=0)
    name_to_idx = {n: names.index(n) for n in all_fixed_names(fixed)}
    sched = build_fixed_mini_schedule(
        names,
        name_to_idx,
        fixed,
        class_order=["C", "RC", "R"],
        cycles_by_class={"C": 3, "RC": 3, "R": 3},
        shuffle_within_cycle=False,
        seed=42,
    )
    for cls in ("C", "RC", "R"):
        per_cycle_sets = []
        for cycle in (1, 2, 3):
            cycle_names = {
                s["instance_name"]
                for s in sched
                if s["instance_class"] == cls and s["cycle"] == cycle
            }
            per_cycle_sets.append(cycle_names)
        assert per_cycle_sets[0] == per_cycle_sets[1] == per_cycle_sets[2]
        assert per_cycle_sets[0] == set(fixed[cls])


def test_schedule_count_per_cycle():
    names = _synthetic_schneider_names()
    config = {"dataset": {"fixed_train_instances": {}}}
    fixed = resolve_fixed_train_instances(names, config, seed=0)
    name_to_idx = {n: names.index(n) for n in all_fixed_names(fixed)}
    sched = build_fixed_mini_schedule(
        names,
        name_to_idx,
        fixed,
        class_order=["C", "RC", "R"],
        cycles_by_class={"C": 3, "RC": 3, "R": 3},
        shuffle_within_cycle=False,
        seed=42,
    )
    for cls in ("C", "RC", "R"):
        for name in fixed[cls]:
            count = sum(1 for s in sched if s["instance_name"] == name)
            assert count == 3


def test_build_schedule_total_steps():
    names = _synthetic_schneider_names()
    config = {"dataset": {"fixed_train_instances": {}}}
    fixed = resolve_fixed_train_instances(names, config, seed=0)
    cycles = {"C": 200, "RC": 200, "R": 200}
    assert total_expected_updates(fixed, cycles) == 3000
    name_to_idx = {n: names.index(n) for n in all_fixed_names(fixed)}
    sched = build_fixed_mini_schedule(
        names,
        name_to_idx,
        fixed,
        class_order=["C", "RC", "R"],
        cycles_by_class=cycles,
        shuffle_within_cycle=False,
        seed=42,
    )
    assert len(sched) == 3000


def test_manifest_roundtrip(tmp_path: Path):
    fixed = {
        "C": [f"c10{i}_21" for i in range(1, 6)],
        "RC": [f"rc10{i}_21" for i in range(1, 6)],
        "R": [f"r10{i}_21" for i in range(1, 6)],
    }
    cycles = {"C": 2, "RC": 2, "R": 2}
    path = write_fixed_train_manifest(
        tmp_path,
        fixed_by_class=fixed,
        seed=99,
        dataset_path="dataset/Schneider/evrptw_instances",
        class_order=["C", "RC", "R"],
        cycles_by_class=cycles,
    )
    loaded = load_fixed_train_manifest(path)
    assert manifest_fixed_by_class(loaded) == fixed
    assert loaded["fixed_train_set_hash"] == fixed_train_set_hash(fixed)


def test_disjoint_train_eval_raises():
    fixed = {"c101_21", "rc101_21"}
    with pytest.raises(RuntimeError, match="overlap"):
        assert_disjoint_train_eval(fixed, ["c101_21", "c102_21"])


def test_resume_reloads_manifest(tmp_path: Path):
    fixed = {
        "C": [f"c10{i}_21" for i in range(1, 6)],
        "RC": [f"rc10{i}_21" for i in range(1, 6)],
        "R": [f"r10{i}_21" for i in range(1, 6)],
    }
    path = write_fixed_train_manifest(
        tmp_path,
        fixed_by_class=fixed,
        seed=1,
        dataset_path="dataset/x",
        class_order=["C", "RC", "R"],
        cycles_by_class={"C": 1, "RC": 1, "R": 1},
    )
    training_meta = {
        "mode": "fixed_mini",
        "global_step": 7,
        "manifest_path": str(path),
    }
    reloaded = load_fixed_train_manifest(Path(training_meta["manifest_path"]))
    assert manifest_fixed_by_class(reloaded) == fixed
    assert training_meta["global_step"] == 7


def test_filter_eval_groups_excludes_fixed():
    from src.training.fixed_mini import filter_eval_groups

    groups = {"C": [0, 1], "RC": [2], "R": [3]}
    names = ["C1_10_1", "C1_10_2", "RC1_10_1", "R1_10_1"]
    filtered = filter_eval_groups(groups, names, {"C1_10_1"})
    assert filtered["C"] == [1]
    assert filtered["RC"] == [2]
