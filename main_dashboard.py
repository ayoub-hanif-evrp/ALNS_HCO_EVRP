"""Launch the live training dashboard (NiceGUI).

This is the primary training monitor. Run in a second terminal while training:

    python main_dashboard.py

Open http://localhost:8080 — scalar training metrics auto-refresh every 3s.

For route maps and solution-behavior diagnostics (manual instance/epoch selection):

    python main_maps_dashboard.py

Open http://localhost:8081
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
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config.")
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
    title="MARL EVRP Live Training",
    host=args.host,
    port=args.port,
    reload=False,
    show=False,
)
