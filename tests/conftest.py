"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable as `src.*`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATASET_PATH = PROJECT_ROOT / "dataset" / "evrptw_instances"
SMALL_INSTANCE = "c101C5"  # 5 customers, fast for tests


@pytest.fixture(scope="session")
def dataset_path() -> Path:
    return DATASET_PATH


@pytest.fixture(scope="session")
def small_instance():
    from src.data.instance_loader import load_instance

    return load_instance(DATASET_PATH / SMALL_INSTANCE)
