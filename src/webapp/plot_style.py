"""Shared colors and class/stage grouping for the training dashboard."""

from __future__ import annotations

# Fixed colors: one color per instance class everywhere in the dashboard.
CLASS_COLORS: dict[str, str] = {
    "C": "#2563eb",   # blue
    "R": "#dc2626",   # red
    "RC": "#16a34a",  # green
}

CLASS_ORDER: list[str] = ["C", "R", "RC"]

CLASS_STAGE_ID: dict[str, int] = {"C": 1, "R": 2, "RC": 3}

STAGE_COLORS: dict[int, str] = {
    0: "#64748b",
    1: CLASS_COLORS["C"],
    2: CLASS_COLORS["R"],
    3: CLASS_COLORS["RC"],
    4: "#9333ea",
    5: "#d97706",
}

# Named curriculum stages (optional labels from config).
CURRICULUM_STAGE_COLORS: dict[str, str] = {
    "C_R": "#7c3aed",
    "C_R_RC": "#0891b2",
    "hard_finetune": "#d97706",
}


def curriculum_stage_color(name: str, index: int = 0) -> str:
    """Color for a curriculum stage label; falls back to SECTION palette."""
    key = str(name or "").strip()
    if key in CURRICULUM_STAGE_COLORS:
        return CURRICULUM_STAGE_COLORS[key]
    if index > 0 and index in STAGE_COLORS:
        return STAGE_COLORS[index]
    return SECTION_ACCENTS[index % len(SECTION_ACCENTS)]

OVERALL_LINE_COLOR = "#94a3b8"
ROLLING_OVERALL_COLOR = "#0f172a"

COLORS = {
    "blue": CLASS_COLORS["C"],
    "green": CLASS_COLORS["RC"],
    "teal": "#0f766e",
    "orange": "#d97706",
    "red": CLASS_COLORS["R"],
    "purple": "#7c3aed",
    "slate": "#64748b",
    "muted": "#94a3b8",
    "gold": "#ca8a04",
    "rose": "#e11d48",
    "navy": "#1e293b",
}

SECTION_ACCENTS = [
    "#2563eb",
    "#7c3aed",
    "#059669",
    "#d97706",
    "#dc2626",
    "#0891b2",
    "#4f46e5",
]

# Distinct palette for cost-component lines (not class-based).
COMPONENT_COLORS = [
    "#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed",
    "#0f766e", "#ca8a04", "#64748b", "#e11d48", "#0891b2",
]
