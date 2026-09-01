"""Load converted Hanif 1000-customer instances."""
from __future__ import annotations

from pathlib import Path

from src.hanif.instance import load_instance
from src.hanif.paths import ROOT
from src.hanif.data.hanif_convert import DEFAULT_OUTPUT_A, DEFAULT_OUTPUT_B

TW_PATHS = {
    "original": ROOT / DEFAULT_OUTPUT_A,
    "relaxed": ROOT / DEFAULT_OUTPUT_B,
}

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
REPRESENTATIVE = tuple(f"{fam}_10_1" for fam in FAMILIES)


def instance_dir(name: str, tw: str) -> Path:
    key = "relaxed" if str(tw).lower().startswith("relax") else "original"
    return TW_PATHS[key] / name


def load_hanif(name: str, tw: str):
    path = instance_dir(name, tw)
    if not path.is_dir():
        raise FileNotFoundError(f"Hanif instance not converted: {path}")
    return load_instance(path)


def list_instances(root: Path | None = None) -> list[str]:
    folder = root or TW_PATHS["original"]
    if not folder.is_dir():
        return []
    names = [p.name for p in folder.iterdir() if (p / "evrp_instance.json").is_file()]
    return sorted(names)
