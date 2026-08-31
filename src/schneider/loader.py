"""Load official Schneider large EVRPTW instances (100 customers, 21 stations)."""
from __future__ import annotations

from src.schneider.instance import load_instance
from src.schneider.paths import ROOT
from src.schneider.data.schneider_txt_convert import DEFAULT_CONVERTED, LARGE_56, ensure_schneider_converted


def instance_dir(name: str):
    return ROOT / DEFAULT_CONVERTED / name


def load_schneider(name: str):
    ensure_schneider_converted(names=[name])
    return load_instance(instance_dir(name))


def all_names() -> list[str]:
    return list(LARGE_56)
