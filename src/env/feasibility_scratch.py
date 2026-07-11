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
    return (
        int(ev.ev_id),
        int(ev.current_node),
        round(float(ev.current_time), 6),
        round(float(ev.remaining_battery), 4),
        round(float(ev.remaining_capacity), 4),
    )


@dataclass
class FeasibilityScratch:
    """Per-decision memo tables cleared after each mask computation."""

    customer: dict[tuple, tuple[bool, str]] = field(default_factory=dict)
    feasible_indices: dict[tuple, list[int]] = field(default_factory=dict)
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
