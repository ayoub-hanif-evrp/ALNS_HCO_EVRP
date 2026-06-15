"""Compatibility shims so NiceGUI's dependencies work on Python 3.12+/3.14.

Python 3.14 removed the long-deprecated ``pkgutil.find_loader`` and
``pkgutil.get_loader`` helpers. NiceGUI's transitive dependency ``vbuild`` still
calls ``pkgutil.find_loader`` at import time, which raises AttributeError.

Importing this module *before* NiceGUI restores those helpers as thin wrappers
around ``importlib.util.find_spec`` without modifying any installed package.
"""
from __future__ import annotations

import importlib.util
import pkgutil


def _install_pkgutil_shims() -> None:
    if not hasattr(pkgutil, "find_loader"):
        def find_loader(name: str):  # type: ignore[no-redef]
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ValueError, ModuleNotFoundError):
                return None
            return spec.loader if spec else None

        pkgutil.find_loader = find_loader  # type: ignore[attr-defined]

    if not hasattr(pkgutil, "get_loader"):
        def get_loader(module):  # type: ignore[no-redef]
            name = module if isinstance(module, str) else getattr(module, "__name__", None)
            if name is None:
                return None
            return pkgutil.find_loader(name)  # type: ignore[attr-defined]

        pkgutil.get_loader = get_loader  # type: ignore[attr-defined]


_install_pkgutil_shims()
