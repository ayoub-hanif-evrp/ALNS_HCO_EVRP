"""Curriculum schedule, balanced batch sampling and hard-case selection.

The curriculum trains in phases rather than as one random mixed run:

* Phase 1 — C only          (basic feasible construction on clustered customers)
* Phase 2 — C + R           (more dispersed customers, stronger charging)
* Phase 3 — C + R + RC      (generalize across all distributions)
* Phase 4 — hard cases      (fine-tune on the instances the model fails on)

Each phase draws balanced batches from its classes according to fixed
proportions, so a phase never collapses onto a single easy class.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from src.data.instance_class import CLASSES, instance_class
from src.training.rollout import run_episode


@dataclass
class Phase:
    index: int
    name: str
    classes: list[str]
    epochs: int
    proportions: dict[str, float] = field(default_factory=dict)
    extra_margin: int = 2          # m sampled in [m_min, m_min + extra_margin]
    hard_finetune: bool = False    # Phase 4: mix hard + normal instances


DEFAULT_PHASES = [
    Phase(1, "C_only", ["C"], 50, {"C": 1.0}, extra_margin=2),
    Phase(2, "C_R", ["C", "R"], 70, {"C": 0.5, "R": 0.5}, extra_margin=2),
    Phase(3, "C_R_RC", ["C", "R", "RC"], 100, {"C": 1 / 3, "R": 1 / 3, "RC": 1 / 3}, extra_margin=5),
    Phase(4, "hard_finetune", ["C", "R", "RC"], 30, {"C": 1 / 3, "R": 1 / 3, "RC": 1 / 3},
          extra_margin=5, hard_finetune=True),
]


def phases_from_config(cfg: dict | None) -> list[Phase]:
    """Build the phase list from config, falling back to the default schedule."""
    if not cfg or not cfg.get("phases"):
        return DEFAULT_PHASES
    phases = []
    for i, p in enumerate(cfg["phases"], start=1):
        phases.append(
            Phase(
                index=int(p.get("index", i)),
                name=str(p.get("name", f"phase_{i}")),
                classes=[c.upper() for c in p.get("classes", ["C"])],
                epochs=int(p["epochs"]),
                proportions={k.upper(): float(v) for k, v in (p.get("proportions") or {}).items()},
                extra_margin=int(p.get("extra_margin", 2)),
                hard_finetune=bool(p.get("hard_finetune", False)),
            )
        )
    return phases


def sample_phase_m(instance, extra_margin: int, rng: random.Random) -> int:
    """Sample a fleet size in [m_min, m_min + extra_margin] (capped by #customers)."""
    m_min = instance.m_min
    m_max = min(instance.num_customers, m_min + extra_margin)
    return rng.randint(m_min, max(m_min, m_max))


def _normalize_proportions(classes: list[str], proportions: dict[str, float]) -> dict[str, float]:
    props = {c: float(proportions.get(c, 0.0)) for c in classes}
    total = sum(props.values())
    if total <= 0:
        return {c: 1.0 / len(classes) for c in classes}
    return {c: v / total for c, v in props.items()}


def sample_balanced_batch(
    rng: random.Random,
    train_groups: dict[str, list[int]],
    classes: list[str],
    proportions: dict[str, float],
    batch_size: int,
) -> list[int]:
    """Sample ``batch_size`` train indices honoring per-class proportions.

    Classes with no training instances are skipped and their share is
    redistributed across the remaining classes.
    """
    available = [c for c in classes if train_groups.get(c)]
    if not available:
        raise ValueError("No training instances available for this phase's classes.")
    props = _normalize_proportions(available, proportions)

    # Integer per-class counts that sum to batch_size.
    counts = {c: int(np.floor(props[c] * batch_size)) for c in available}
    remainder = batch_size - sum(counts.values())
    # Distribute the leftover to the classes with the largest fractional parts.
    fracs = sorted(available, key=lambda c: props[c] * batch_size - counts[c], reverse=True)
    for k in range(remainder):
        counts[fracs[k % len(fracs)]] += 1

    batch: list[int] = []
    for c in available:
        pool = train_groups[c]
        for _ in range(counts[c]):
            batch.append(rng.choice(pool))
    rng.shuffle(batch)
    return batch


@torch.no_grad()
def select_hard_indices(
    train_ds,
    train_indices: list[int],
    policy,
    weights,
    *,
    device="cpu",
    charging_portions=None,
    use_horizon: bool = True,
    max_steps_factor: int = 6,
    extra_margin: int = 5,
    hard_fraction: float = 0.5,
    rng: random.Random | None = None,
) -> list[int]:
    """Greedily evaluate each training instance and return the hardest indices.

    Hardness combines the failure signals listed in the spec: infeasibility,
    battery / time-window / capacity violations, unserved customers, excessive
    EVs, very high cost and excessive charging visits (each normalized per
    instance so classes of different size compare fairly).
    """
    rng = rng or random.Random(0)
    policy.eval()
    scored: list[tuple[float, int]] = []
    for idx in train_indices:
        instance = train_ds[idx]
        m = min(instance.num_customers, instance.m_min + 1)
        res = run_episode(
            instance, policy, m, weights, mode="greedy", device=device,
            charging_portions=charging_portions, use_horizon_check=use_horizon,
            max_steps_factor=max_steps_factor,
        )
        s = res.summary
        n_cust = max(instance.num_customers, 1)
        hardness = 0.0
        hardness += 0.0 if res.feasible else 5.0
        hardness += s.unserved_customers / n_cust
        hardness += min(s.total_battery_violation, 1e3) / 1e3
        hardness += min(s.total_capacity_violation, 1e3) / 1e3
        hardness += min(s.total_lateness, 1e3) / 1e3
        hardness += s.used_evs / n_cust
        hardness += len([1 for ev in res.env.evs for _ in ev.charging_actions]) / n_cust
        hardness += min(res.objective, 1e6) / 1e6
        scored.append((hardness, idx))

    scored.sort(key=lambda t: t[0], reverse=True)
    k = max(1, int(round(hard_fraction * len(scored))))
    return [idx for _, idx in scored[:k]]


def sample_hard_batch(
    rng: random.Random,
    hard_pool: list[int],
    normal_pool: list[int],
    batch_size: int,
    hard_share: float = 0.5,
) -> list[int]:
    """Phase 4 batch: ``hard_share`` from hard instances, the rest from normal."""
    n_hard = int(round(hard_share * batch_size))
    n_normal = batch_size - n_hard
    batch = [rng.choice(hard_pool) for _ in range(n_hard)] if hard_pool else []
    pool = normal_pool or hard_pool
    batch += [rng.choice(pool) for _ in range(n_normal + (0 if hard_pool else n_hard))]
    rng.shuffle(batch)
    return batch


def build_cyclic_schedule(
    train_groups: dict[str, list[int]],
    *,
    class_order: list[str] | None = None,
    cycles: int = 5,
    shuffle_within_class: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """Build a sequential schedule: all C, then all R, then all RC, repeat ``cycles`` times.

    Each entry is one training iteration (one instance, one REINFORCE update).
    """
    rng = rng or random.Random(0)
    order = [c.upper() for c in (class_order or ["C", "R", "RC"])]
    base: list[tuple[str, int]] = []
    for cls in order:
        indices = list(train_groups.get(cls, []))
        if not indices:
            continue
        if shuffle_within_class:
            rng.shuffle(indices)
        else:
            indices.sort()
        for idx in indices:
            base.append((cls, idx))
    if not base:
        raise ValueError("Cyclic schedule is empty — no training instances for the requested classes.")

    schedule: list[dict] = []
    for cycle in range(1, int(cycles) + 1):
        for step_in_cycle, (cls, idx) in enumerate(base, start=1):
            schedule.append(
                {
                    "instance_idx": idx,
                    "instance_class": cls,
                    "cycle": cycle,
                    "step_in_cycle": step_in_cycle,
                    "steps_per_cycle": len(base),
                }
            )
    return schedule


def cyclic_total_steps(
    train_groups: dict[str, list[int]],
    *,
    class_order: list[str] | None = None,
    cycles: int = 5,
) -> int:
    order = [c.upper() for c in (class_order or ["C", "R", "RC"])]
    per_cycle = sum(len(train_groups.get(c, [])) for c in order)
    return per_cycle * int(cycles)


def extra_margin_for_class(cfg: dict, instance_class: str, default: int = 2) -> int:
    by_class = (cfg.get("curriculum") or {}).get("extra_margin_by_class") or {}
    return int(by_class.get(instance_class.upper(), default))


def _cycles_for_class(
    cls: str,
    *,
    cycles_small: int,
    cycles_full: int,
    cycles_by_class: dict | None,
) -> tuple[int, int]:
    """Resolve small/full cycle counts for a class (optional per-class overrides)."""
    overrides = (cycles_by_class or {}).get(cls) or (cycles_by_class or {}).get(cls.upper())
    if overrides:
        return int(overrides.get("small", cycles_small)), int(overrides.get("full", cycles_full))
    return int(cycles_small), int(cycles_full)


def _append_size_tier_steps(
    schedule: list[dict],
    *,
    names: list[str],
    cls: str,
    indices: list[int],
    tier: str,
    cycles: int,
    tier_label: str,
    shuffle_within_tier: bool,
    rng: random.Random,
) -> None:
    if not indices:
        return
    tier_indices = list(indices)
    if shuffle_within_tier:
        rng.shuffle(tier_indices)
    for cycle in range(1, cycles + 1):
        for step_in_cycle, idx in enumerate(tier_indices, start=1):
            schedule.append(
                {
                    "instance_idx": idx,
                    "instance_class": cls,
                    "size_tier": tier,
                    "cycle": cycle,
                    "tier_label": tier_label,
                    "step_in_cycle": step_in_cycle,
                    "steps_per_cycle": len(tier_indices),
                }
            )


def build_class_size_schedule(
    names: list[str],
    train_groups: dict[str, list[int]],
    instance_class: str,
    *,
    cycles_small: int = 3,
    cycles_full: int = 5,
    shuffle_within_tier: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """One class: all small instances (C5/C10/C15), then all full (_21)."""
    from src.data.instance_class import split_size_tiers

    cls = instance_class.upper()
    indices = list(train_groups.get(cls, []))
    if not indices:
        raise ValueError(f"{cls} size schedule is empty — no {cls} training instances.")

    rng = rng or random.Random(0)
    small, full = split_size_tiers(names, indices)
    schedule: list[dict] = []
    for tier, tier_indices, cycles, tier_label in (
        ("small", small, int(cycles_small), f"{cls}_small"),
        ("full", full, int(cycles_full), f"{cls}_full"),
    ):
        _append_size_tier_steps(
            schedule,
            names=names,
            cls=cls,
            indices=tier_indices,
            tier=tier,
            cycles=cycles,
            tier_label=tier_label,
            shuffle_within_tier=shuffle_within_tier,
            rng=rng,
        )
    if not schedule:
        raise ValueError(f"{cls} size schedule is empty after tier split.")
    return schedule


def build_progressive_schedule(
    names: list[str],
    train_groups: dict[str, list[int]],
    *,
    class_order: list[str] | None = None,
    cycles_small: int = 3,
    cycles_full: int = 3,
    cycles_by_class: dict | None = None,
    shuffle_within_tier: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """Progressive curriculum: for each class, small instances then full.

    Default order C → RC → R (easiest to hardest). Each entry is one training
    iteration (one instance, one REINFORCE update).
    """
    from src.data.instance_class import split_size_tiers

    order = [c.upper() for c in (class_order or ["C", "RC", "R"])]
    rng = rng or random.Random(0)
    schedule: list[dict] = []

    for cls in order:
        indices = list(train_groups.get(cls, []))
        if not indices:
            continue
        c_small, c_full = _cycles_for_class(
            cls,
            cycles_small=cycles_small,
            cycles_full=cycles_full,
            cycles_by_class=cycles_by_class,
        )
        small, full = split_size_tiers(names, indices)
        for tier, tier_indices, cycles, tier_label in (
            ("small", small, c_small, f"{cls}_small"),
            ("full", full, c_full, f"{cls}_full"),
        ):
            _append_size_tier_steps(
                schedule,
                names=names,
                cls=cls,
                indices=tier_indices,
                tier=tier,
                cycles=cycles,
                tier_label=tier_label,
                shuffle_within_tier=shuffle_within_tier,
                rng=rng,
            )

    if not schedule:
        raise ValueError("Progressive schedule is empty — no training instances in class_order.")
    return schedule


def progressive_blocks(
    schedule: list[dict],
    class_order: list[str] | None = None,
) -> list[dict]:
    """Split a progressive schedule into per-class blocks with small/full tiers.

    Returns a list of dicts:
        {"class": "C", "small": [...steps...], "full": [...steps...]}
    """
    order = [c.upper() for c in (class_order or [])]
    if not order:
        seen: list[str] = []
        for step in schedule:
            cls = str(step.get("instance_class", "")).upper()
            if cls and cls not in seen:
                seen.append(cls)
        order = seen

    blocks: list[dict] = []
    for cls in order:
        cls_steps = [s for s in schedule if str(s.get("instance_class", "")).upper() == cls]
        if not cls_steps:
            continue
        blocks.append({
            "class": cls,
            "small": [s for s in cls_steps if s.get("size_tier") == "small"],
            "full": [s for s in cls_steps if s.get("size_tier") == "full"],
        })
    return blocks


def build_rc_size_schedule(
    names: list[str],
    train_groups: dict[str, list[int]],
    *,
    cycles_small: int = 3,
    cycles_full: int = 5,
    shuffle_within_tier: bool = False,
    rng: random.Random | None = None,
) -> list[dict]:
    """RC-only curriculum: repeat all small instances, then all full instances."""
    return build_progressive_schedule(
        names,
        train_groups,
        class_order=["RC"],
        cycles_small=cycles_small,
        cycles_full=cycles_full,
        shuffle_within_tier=shuffle_within_tier,
        rng=rng,
    )


def progressive_total_steps(
    names: list[str],
    train_groups: dict[str, list[int]],
    *,
    class_order: list[str] | None = None,
    cycles_small: int = 3,
    cycles_full: int = 3,
    cycles_by_class: dict | None = None,
) -> int:
    """Expected iteration count for a progressive schedule (without shuffling)."""
    from src.data.instance_class import split_size_tiers

    order = [c.upper() for c in (class_order or ["C", "RC", "R"])]
    total = 0
    for cls in order:
        indices = train_groups.get(cls, [])
        if not indices:
            continue
        c_small, c_full = _cycles_for_class(
            cls,
            cycles_small=cycles_small,
            cycles_full=cycles_full,
            cycles_by_class=cycles_by_class,
        )
        small, full = split_size_tiers(names, indices)
        total += len(small) * c_small + len(full) * c_full
    return total


def active_train_classes(cfg: dict) -> list[str]:
    """Classes used for training / validation (defaults to all three)."""
    ccfg = cfg.get("curriculum") or {}
    classes = ccfg.get("train_classes") or ccfg.get("class_order") or list(CLASSES)
    from src.data.instance_class import class_filter_for

    filtered = sorted(class_filter_for(classes))
    return filtered or list(CLASSES)
