"""Canonical locations of the frozen paper results."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUTPUTS = ROOT / "outputs"
HANIF_OUT = OUTPUTS / "hanif"
SCHNEIDER_OUT = OUTPUTS / "schneider"

CLASSICAL = HANIF_OUT / "classical"
PROPOSED = HANIF_OUT / "proposed"
REFERENCES = HANIF_OUT / "references"
CONSTRUCTOR = HANIF_OUT / "baselines" / "constructor"
ALNS = HANIF_OUT / "baselines" / "alns"
VNS_TS = HANIF_OUT / "baselines" / "vns_ts"
FIGURES = HANIF_OUT / "figures"
TABLES = HANIF_OUT / "tables"
SCHNEIDER_FROZEN_PR = SCHNEIDER_OUT / "pr"
SCHNEIDER_FROZEN_FR = SCHNEIDER_OUT / "fr"

FROZEN_RESULT_DIRS = (
    CLASSICAL,
    PROPOSED,
    REFERENCES,
    CONSTRUCTOR,
    ALNS,
    VNS_TS,
    FIGURES,
    TABLES,
    SCHNEIDER_FROZEN_PR,
    SCHNEIDER_FROZEN_FR,
)
