"""Clear live training outputs before a new run starts."""
from __future__ import annotations

import os
import stat
import time
from pathlib import Path

from src.utils.logging_utils import get_logger

LOGGER = get_logger()


def clear_training_outputs(out_cfg: dict) -> None:
    """Remove previous run artifacts so a new training session starts fresh.

    Clears logs, result CSVs/JSON, training figures and checkpoints. Does not
    delete ``outputs/runs/`` archives (historical copies) or evaluation
    ``outputs/results/archive/``.

    On Windows, locked files (OneDrive, open viewers) are skipped with a warning
    rather than aborting training.
    """
    root = Path(out_cfg.get("root", "outputs"))

    logs_dir = Path(out_cfg.get("logs", root / "logs"))
    _clear_files(logs_dir, ("*.jsonl",))

    results_dir = Path(out_cfg.get("results", root / "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("*.csv", "*.json", "*.txt"):
        _clear_files(results_dir, (pattern,))

    figures_training = Path(out_cfg.get("figures", root / "figures")) / "training"
    _clear_tree(figures_training)
    figures_training.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(out_cfg.get("checkpoints", root / "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _clear_files(ckpt_dir, ("*.pt",))

    LOGGER.info("Cleared previous training outputs (logs, results, figures/training, checkpoints).")


def _clear_files(directory: Path, patterns: tuple[str, ...]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for f in directory.glob(pattern):
            _unlink_safe(f)


def _clear_tree(directory: Path) -> None:
    """Delete directory contents file-by-file (more reliable than rmtree on Windows)."""
    if not directory.exists():
        return
    failed: list[Path] = []
    for path in sorted(directory.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_file() or path.is_symlink():
            if not _unlink_safe(path):
                failed.append(path)
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                failed.append(path)
    if directory.exists():
        try:
            directory.rmdir()
        except OSError:
            if any(directory.iterdir()):
                failed.append(directory)
    if failed:
        LOGGER.warning(
            f"Could not remove {len(failed)} locked path(s) under {directory} "
            "(close viewers / wait for OneDrive sync). Training will continue."
        )


def _unlink_safe(path: Path, *, retries: int = 3) -> bool:
    for attempt in range(retries):
        try:
            path.unlink(missing_ok=True)
            return True
        except PermissionError:
            _make_writable(path)
            try:
                path.unlink(missing_ok=True)
                return True
            except OSError:
                pass
        except OSError:
            pass
        if attempt < retries - 1:
            time.sleep(0.15 * (attempt + 1))
    return not path.exists()


def _make_writable(path: Path) -> None:
    try:
        os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
