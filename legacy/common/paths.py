"""Output roots. Frozen archives stay where they are and are never overwritten."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

# Frozen paper archives (do not write here).
FROZEN_HANIF_CLASSICAL = OUTPUTS / "classical"
FROZEN_HANIF_PROPOSED = OUTPUTS / "proposed"
FROZEN_HANIF_TABLES = OUTPUTS / "tables"
FROZEN_HANIF_FIGURES = OUTPUTS / "figures"
FROZEN_SCHNEIDER_FIRST = OUTPUTS / "schneider"

# Active write roots for the refactored runners.
HANIF_OUT = OUTPUTS / "hanif"
SCHNEIDER_OUT = OUTPUTS / "schneider"
