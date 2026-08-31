"""Canonical locations of the frozen paper results.

All final artifacts live directly under ``outputs/``:

classical, proposed, references, constructor, alns, vns_ts, figures, tables.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

CLASSICAL = OUTPUTS / "classical"
PROPOSED = OUTPUTS / "proposed"
REFERENCES = OUTPUTS / "references"
CONSTRUCTOR = OUTPUTS / "constructor"
ALNS = OUTPUTS / "alns"
VNS_TS = OUTPUTS / "vns_ts"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
HANIF_OUT = OUTPUTS / "hanif"
SCHNEIDER_OUT = OUTPUTS / "schneider"
SCHNEIDER_FROZEN_PR = OUTPUTS / "schneider" / "pr"

FROZEN_RESULT_DIRS = (
    CLASSICAL,
    PROPOSED,
    REFERENCES,
    CONSTRUCTOR,
    ALNS,
    VNS_TS,
    FIGURES,
    TABLES,
)
