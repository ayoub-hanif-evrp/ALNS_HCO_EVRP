"""Output roots. Frozen archives are never overwritten. Scratch runs go to outputs/_tmp."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

HANIF_OUT = OUTPUTS / "care-fi"
SCHNEIDER_OUT = OUTPUTS / "schneider"
TMP_OUT = OUTPUTS / "_tmp"

FROZEN_HANIF_CLASSICAL = HANIF_OUT / "classical"
FROZEN_HANIF_PROPOSED = HANIF_OUT / "proposed"
FROZEN_HANIF_TABLES = HANIF_OUT / "tables"
FROZEN_HANIF_FIGURES = HANIF_OUT / "figures"
FROZEN_HANIF_BASELINES = HANIF_OUT / "baselines"
FROZEN_SCHNEIDER_PR = SCHNEIDER_OUT / "pr"
FROZEN_SCHNEIDER_FR = SCHNEIDER_OUT / "fr"
FROZEN_SCHNEIDER_FIRST = FROZEN_SCHNEIDER_PR
