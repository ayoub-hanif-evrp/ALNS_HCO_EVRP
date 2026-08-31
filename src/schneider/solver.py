"""Schneider-compatible CARE-FI wrapper. Does not use Hanif station economics."""
from __future__ import annotations

from src.schneider.search.schneider_solve import SchneiderSolveResult, solve_schneider

__all__ = ["solve_schneider", "SchneiderSolveResult"]
