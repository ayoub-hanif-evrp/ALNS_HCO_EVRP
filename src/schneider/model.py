"""Schneider evaluation rules: homogeneous stations, lex min K then D, no wait/price."""
from __future__ import annotations

from src.schneider.evaluation.schneider_bks import BKS_CITATION, BKS_PR, compare_row, summarize

# Discrete PR portions used by the current Schneider-compatible solver.
PORTIONS_PR = (0.10, 0.20, 0.35, 0.50, 0.75, 1.0)
PORTIONS_FR = (1.0,)


def lex_key(feasible: bool, used_k: int, distance: float) -> tuple:
    return (0 if feasible else 1, int(used_k), float(distance))


__all__ = ["BKS_CITATION", "BKS_PR", "compare_row", "summarize", "PORTIONS_PR", "PORTIONS_FR", "lex_key"]
