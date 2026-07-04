"""Flat training schedules."""
from __future__ import annotations

import random


def sample_fleet_size(instance, extra_margin: int, rng: random.Random) -> int:
    """Sample fleet size m in [m_min, m_min + extra_margin] (capped by #customers)."""
    m_min = instance.m_min
    m_max = min(instance.num_customers, m_min + extra_margin)
    return rng.randint(m_min, max(m_min, m_max))


def build_sequential_schedule(
    num_instances: int,
    passes: int,
    *,
    shuffle_within_pass: bool = False,
    rng: random.Random | None = None,
) -> list[int]:
    """Repeat all instance indices ``passes`` times (one index per training step)."""
    if num_instances < 1:
        raise ValueError("build_sequential_schedule requires at least one training instance")
    if passes < 1:
        raise ValueError("passes must be >= 1")

    rng = rng or random.Random(0)
    base = list(range(num_instances))
    schedule: list[int] = []
    for _ in range(int(passes)):
        order = list(base)
        if shuffle_within_pass:
            rng.shuffle(order)
        schedule.extend(order)
    return schedule
