"""Scoped memoization for exact mask computation — semantics unchanged."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .ev_state import EVState

_current_scratch: ContextVar["FeasibilityScratch | None"] = ContextVar(
    "_feasibility_scratch", default=None,
)


def ev_state_key(ev: EVState) -> tuple:
    """Quantized state key — int scaling avoids millions of ``round()`` calls."""
    return (
        int(ev.ev_id),
        int(ev.current_node),
        int(float(ev.current_time) * 1000),
        int(float(ev.remaining_battery) * 10000),
        int(float(ev.remaining_capacity) * 10000),
    )


@dataclass
class FeasibilityScratch:
    """Per-decision memo tables cleared after each mask computation."""

    customer: dict[tuple, tuple[bool, str]] = field(default_factory=dict)
    feasible_indices: dict[tuple, list[int]] = field(default_factory=dict)
    feasible_sets: dict[tuple, frozenset[int]] = field(default_factory=dict)
    has_feasible: dict[tuple, bool] = field(default_factory=dict)
    charge_portions: dict[tuple, tuple] = field(default_factory=dict)
    charge_analysis: dict[tuple, object] = field(default_factory=dict)


def get_scratch() -> FeasibilityScratch | None:
    return _current_scratch.get()


@contextmanager
def feasibility_scratch_scope():
    scratch = FeasibilityScratch()
    token = _current_scratch.set(scratch)
    try:
        yield scratch
    finally:
        _current_scratch.reset(token)
