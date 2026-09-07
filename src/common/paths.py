"""Hanif paper result roots for the clean revision study."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HANIF_OUT = ROOT / "outputs"
REVIEW = ROOT / "review_before_delete"


def hanif_dirs() -> dict[str, Path]:
    paths = {
        "root": HANIF_OUT,
        "raw": HANIF_OUT / "raw",
        "processed": HANIF_OUT / "processed",
        "ablation": HANIF_OUT / "ablation",
        "statistics": HANIF_OUT / "statistics",
        "tables": HANIF_OUT / "tables",
        "figures": HANIF_OUT / "figures",
        "supplement": HANIF_OUT / "figures" / "supplement",
        "diagnostics": HANIF_OUT / "diagnostics",
        "validation": HANIF_OUT / "validation",
        "analysis": HANIF_OUT / "analysis",
        "config": HANIF_OUT / "config",
        "logs": HANIF_OUT / "logs",
        "shared_phase1": HANIF_OUT / "shared_phase1",
        "paper": HANIF_OUT / "processed",
        "paper_tables": HANIF_OUT / "tables",
        "paper_figures": HANIF_OUT / "figures",
        "paper_supplement": HANIF_OUT / "figures" / "supplement",
        "paper_data": HANIF_OUT / "processed",
        "runs": HANIF_OUT / "raw",
        "summaries": HANIF_OUT / "processed",
        "legacy": HANIF_OUT / "legacy_pre_revision",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def schneider_dirs() -> dict[str, Path]:
    """Unused by the Hanif paper. Kept so old imports do not crash."""
    root = REVIEW / "_unused_schneider_outputs"
    paths = {
        "root": root,
        "raw": root / "raw",
        "runs": root / "runs",
        "summaries": root / "summaries",
        "tables": root / "tables",
        "figures": root / "figures",
        "diagnostics": root / "diagnostics",
        "paper": root / "paper",
    }
    return paths
