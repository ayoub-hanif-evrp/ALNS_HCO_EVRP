"""Solomon/Schneider and Hanif instance-class detection (C / R / RC).

Supported naming conventions:

* Schneider-style: ``c101_21``, ``c101C5``, ``r201C10``, ``rc108C15``
* Hanif-style: ``C1_10_1``, ``RC2_10_5`` (Homberger 1000-customer, series 1/2)
"""
from __future__ import annotations

import re

CLASSES = ("C", "R", "RC")

# Schneider reduced instances (e.g. rc108C5, rc201C10).
SMALL_SIZE_MARKERS = ("C5", "C10", "C15")

# Hanif Homberger series 1 (C1/R1/RC1) — curriculum "small" tier (relaxed TW variant B).
_HANIF_SERIES1 = re.compile(r"^(RC1|C1|R1)_", re.IGNORECASE)


def is_small_instance(name: str) -> bool:
    """True for Schneider reduced sizes or Hanif series-1 instances (C1/R1/RC1)."""
    key = name.upper()
    if _HANIF_SERIES1.match(key):
        return True
    return any(key.endswith(marker) for marker in SMALL_SIZE_MARKERS)


def instance_class(name: str) -> str:
    """Return the customer-distribution class ('C', 'R' or 'RC') for a name."""
    key = name.strip().lower()
    if key.startswith("rc"):
        return "RC"
    if key.startswith("r"):
        return "R"
    if key.startswith("c"):
        return "C"
    return "R"


def group_indices_by_class(names: list[str]) -> dict[str, list[int]]:
    """Map each class to the list of dataset indices whose name belongs to it."""
    groups: dict[str, list[int]] = {c: [] for c in CLASSES}
    for idx, name in enumerate(names):
        groups[instance_class(name)].append(idx)
    return groups


def class_filter_for(classes: list[str]) -> set[str]:
    """Normalize a list of requested class labels to a set of valid classes."""
    requested = {c.strip().upper() for c in classes}
    return {c for c in requested if c in CLASSES}


def filter_groups_by_classes(
    groups: dict[str, list[int]], classes: list[str]
) -> dict[str, list[int]]:
    """Keep only the requested classes (empty lists if a class has no instances)."""
    allowed = class_filter_for(classes)
    return {c: list(groups.get(c, [])) for c in allowed if c in CLASSES}


def split_size_tiers(names: list[str], indices: list[int]) -> tuple[list[int], list[int]]:
    """Split indices into small vs full tiers (Schneider C5/C15 or Hanif series 1/2)."""
    small, full = [], []
    for idx in indices:
        (small if is_small_instance(names[idx]) else full).append(idx)
    return sorted(small), sorted(full)


def rc_size_groups(names: list[str], rc_indices: list[int]) -> tuple[list[int], list[int]]:
    """Split RC train/eval indices into small (C5/C10/C15) and full (_21) tiers."""
    return split_size_tiers(names, rc_indices)
