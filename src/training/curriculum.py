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

from src.data.instance_class import instance_class
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
