"""Hanif vs Schneider output roots. Never mix the two."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
HANIF_OUT = OUTPUTS / "hanif"
SCHNEIDER_OUT = OUTPUTS / "schneider"
LEGACY_ALNS_HCO = OUTPUTS / "alns-hco"
LEGACY_REFERENCE = ROOT / "legacy_reference"


def hanif_dirs() -> dict[str, Path]:
    paper = HANIF_OUT / "paper"
    paths = {
        "root": HANIF_OUT,
        "raw": HANIF_OUT / "raw",
        "runs": HANIF_OUT / "runs",
        "summaries": HANIF_OUT / "summaries",
        "tables": HANIF_OUT / "tables",
        "figures": HANIF_OUT / "figures",
        "diagnostics": HANIF_OUT / "diagnostics",
        "paper": paper,
        "paper_tables": paper / "tables",
        "paper_figures": paper / "figures",
        "paper_supplement": paper / "supplement",
        "paper_data": paper / "data",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def schneider_dirs() -> dict[str, Path]:
    paths = {
        "root": SCHNEIDER_OUT,
        "raw": SCHNEIDER_OUT / "raw",
        "runs": SCHNEIDER_OUT / "runs",
        "summaries": SCHNEIDER_OUT / "summaries",
        "tables": SCHNEIDER_OUT / "tables",
        "figures": SCHNEIDER_OUT / "figures",
        "diagnostics": SCHNEIDER_OUT / "diagnostics",
        "paper": SCHNEIDER_OUT / "paper",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths
