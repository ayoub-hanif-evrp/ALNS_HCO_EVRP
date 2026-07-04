"""Launch the solution-behavior maps explorer (NiceGUI).

Run alongside training — manual instance/epoch selection, no auto-refresh:

    python main_maps_dashboard.py

Open http://localhost:8081 — pick a training instance, scrub epochs, click
**Load snapshot** to generate route maps and diagnostic plots for that point.
"""
from __future__ import annotations

import src.webapp._compat  # noqa: F401

import argparse
from pathlib import Path

from nicegui import ui

from src.utils.config import load_config
from src.webapp.maps_dashboard import create_maps_dashboard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Solution-behavior maps explorer for E-VRPTW training.")
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    parser.add_argument("--log", type=str, default=None)
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument(
        "--main-dashboard-port", type=int, default=8080,
        help="Port of main_dashboard.py for the back-link.",
    )
    return parser.parse_args()


args = parse_args()
config = load_config(args.config)
log_path = Path(args.log) if args.log else Path(config["outputs"]["logs"]) / "train_log.jsonl"

create_maps_dashboard(
    log_path,
    config_path=Path(args.config),
    outputs_root=Path(config["outputs"]["root"]),
    main_dashboard_port=args.main_dashboard_port,
)

ui.run(
    title="MARL EVRP Solution Maps",
    host=args.host,
    port=args.port,
    reload=False,
    show=False,
)
