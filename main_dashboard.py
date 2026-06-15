"""Launch the live training dashboard (NiceGUI).

Run this in a separate terminal while training is in progress:

    python main_dashboard.py --config configs/default.yaml

Then open http://localhost:8080 in a browser. The page auto-refreshes every
few seconds and reads the training log the trainer writes to
``outputs/logs/train_log.jsonl``.
"""
from __future__ import annotations

# Restore pkgutil.find_loader for NiceGUI's deps on Python 3.14 before any import.
import src.webapp._compat  # noqa: F401

import argparse
from pathlib import Path

from nicegui import ui

from src.utils.config import load_config
from src.webapp.dashboard import create_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live training dashboard for the E-VRPTW solver.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--log", type=str, default=None, help="Override path to the train_log.jsonl file.")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve the dashboard on.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host/interface to bind.")
    return parser.parse_args()


args = parse_args()

if args.log is not None:
    log_path = Path(args.log)
    config = load_config(args.config)
else:
    config = load_config(args.config)
    log_path = Path(config["outputs"]["logs"]) / "train_log.jsonl"

create_dashboard(
    log_path,
    config_path=Path(args.config),
    outputs_root=Path(config["outputs"]["root"]),
)

ui.run(
    title="E-VRPTW Live Training",
    host=args.host,
    port=args.port,
    reload=False,
    show=False,
)
