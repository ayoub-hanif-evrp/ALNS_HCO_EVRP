"""Solomon/Schneider instance-class detection (C / R / RC).

Instance names follow the Solomon convention used by the converted dataset, e.g.
``c101_21``, ``c101C5``, ``r201C10``, ``rc108C15``. The leading letters give the
customer-distribution class:

* ``C``  — clustered customers       (names starting with ``c``)
* ``R``  — random / dispersed         (names starting with ``r`` but not ``rc``)
* ``RC`` — mixed clustered + random   (names starting with ``rc``)
"""
from __future__ import annotations

CLASSES = ("C", "R", "RC")


def instance_class(name: str) -> str:
    """Return the customer-distribution class ('C', 'R' or 'RC') for a name."""
    key = name.strip().lower()
    if key.startswith("rc"):
        return "RC"
    if key.startswith("r"):
        return "R"
    if key.startswith("c"):
        return "C"
    # Unknown prefix: default to R so it is never silently dropped.
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
