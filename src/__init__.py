"""EVRPTW solvers. Hyphenated folders are registered as importable packages."""
from pathlib import Path
from types import ModuleType
import sys

_ROOT = Path(__file__).resolve().parent
_ALIASES = {
    "src.hanif": _ROOT / "care-fi",
    "src.care_fi": _ROOT / "care-fi",
    "src.alns_hco": _ROOT / "alns-hco",
}


def _register_aliases() -> None:
    for name, path in _ALIASES.items():
        if name in sys.modules or not path.is_dir():
            continue
        pkg = ModuleType(name)
        pkg.__path__ = [str(path)]
        pkg.__file__ = str(path / "__init__.py")
        pkg.__package__ = name
        sys.modules[name] = pkg


_register_aliases()
