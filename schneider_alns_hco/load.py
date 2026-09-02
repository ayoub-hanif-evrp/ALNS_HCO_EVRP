"""Load converted Schneider EVRPTW folders with the ALNS-HCO instance type."""
from __future__ import annotations

from pathlib import Path

from config import INSTANCE_DIR, LARGE_56, ROOT


def family_of(name: str) -> str:
    stem = name.lower().replace("_21", "")
    if stem.startswith("rc"):
        return f"RC{stem[2]}"
    if stem.startswith("r"):
        return f"R{stem[1]}"
    if stem.startswith("c"):
        return f"C{stem[1]}"
    return "UNK"


def instance_path(name: str) -> Path:
    return INSTANCE_DIR / name


def ensure_converted(names: list[str] | None = None) -> None:
    missing = [n for n in (names or list(LARGE_56)) if not (instance_path(n) / "evrp_instance.json").is_file()]
    if not missing:
        return
    import sys

    sys.path.insert(0, str(ROOT))
    from src.schneider.data.schneider_txt_convert import ensure_schneider_converted

    ensure_schneider_converted(names=missing)


def load_schneider(name: str):
    """Load one Schneider instance so ALNS-HCO replay/HCO can run on it.

    Uses the Hanif/CARE-FI instance loader (same converted-folder format).
    Schneider folders have no ``stations.csv``, so wait/price stay off.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    ensure_converted([name])
    path = instance_path(name)
    if not (path / "evrp_instance.json").is_file():
        raise FileNotFoundError(f"Schneider instance not converted: {path}")
    from src.hanif.instance import load_instance

    return load_instance(path)


def list_instances() -> list[str]:
    if not INSTANCE_DIR.is_dir():
        return list(LARGE_56)
    found = [p.name for p in INSTANCE_DIR.iterdir() if (p / "evrp_instance.json").is_file()]
    return [n for n in LARGE_56 if n in found] or list(LARGE_56)
