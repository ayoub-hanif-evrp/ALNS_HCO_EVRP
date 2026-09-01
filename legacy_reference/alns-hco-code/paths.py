"""ALNS-HCO output root. Scratch runs go to outputs/_tmp."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
ALNS_HCO_OUT = OUTPUTS / "alns-hco"
TMP_OUT = OUTPUTS / "_tmp"
