"""Load Hanif instances from raw ``.txt`` (slow) or use ``dataset/converted_hanif_data`` (fast)."""
from __future__ import annotations

from pathlib import Path

from src.data.hanif_converter import build_hanif_instance
from src.data.hanif_parser import parse_hanif_instance
from src.data.instance_loader import EVRPTWInstance, apply_load_capacity


def is_hanif_dataset_root(path: str | Path) -> bool:
    """True when ``path`` contains Hanif ``.txt`` files at the top level."""
    path = Path(path)
    if not path.is_dir():
        return False
    return any(path.glob("*.txt")) and not any(path.glob("*/evrp_instance.json"))


def resolve_ev_specs_path(dataset_path: str | Path, configured: str | Path | None = None) -> Path | None:
    if configured:
        p = Path(configured)
        return p if p.exists() else None
    path = Path(dataset_path)
    for candidate in (path.parent / "ev_specs.json", path.parent.parent / "ev_specs.json"):
        if candidate.exists():
            return candidate
    return None


def list_hanif_txt_files(dataset_path: str | Path) -> list[Path]:
    files = sorted(
        p for p in Path(dataset_path).glob("*.txt")
        if not p.name.startswith(".") and "_stations" not in p.stem
    )
    if not files:
        raise FileNotFoundError(f"No Hanif .txt instances in {dataset_path}")
    return files


def load_hanif_instance(
    txt_path: str | Path,
    *,
    ev_specs_path: str | Path | None = None,
    load_capacity: float | None = None,
    variant: str = "variant_B_relaxed_TW",
) -> EVRPTWInstance:
    parsed = parse_hanif_instance(txt_path)
    _, _, _, _, _, instance, _ = build_hanif_instance(
        parsed, ev_specs_path=ev_specs_path, variant=variant, load_capacity_override=load_capacity,
    )
    if load_capacity is not None:
        apply_load_capacity(instance, load_capacity)
    return instance
