"""EV selection for the asynchronous event-driven decoder."""
from __future__ import annotations

from .ev_state import EVState


def select_next_ev(evs: list[EVState]) -> EVState | None:
    """Return the unfinished EV that becomes available earliest.

    Tie-breakers, in order:
        1. smaller current_time
        2. fewer route nodes (prefer EVs that have done less)
        3. lower ev_id
    Returns ``None`` if every EV is finished.
    """
    candidates = [ev for ev in evs if not ev.finished]
    if not candidates:
        return None
    return min(candidates, key=lambda ev: (ev.current_time, len(ev.route_nodes), ev.ev_id))
