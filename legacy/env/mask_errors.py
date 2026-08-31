"""Mask and action validation errors."""
from __future__ import annotations


class InvalidActionMaskError(RuntimeError):
    """Raised when the policy selects a node/charge/depot mode outside the canonical mask."""


class StaleMaskError(RuntimeError):
    """Raised when a mask result was computed before the latest environment mutation."""


class MaskParityError(RuntimeError):
    """Raised when reference and optimized exact masks disagree."""
