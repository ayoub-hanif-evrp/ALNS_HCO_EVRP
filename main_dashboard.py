"""Launch the live training dashboard (NiceGUI).

Automatically loads the latest run under outputs/runs/.

    python main_dashboard.py

Open http://localhost:8080
"""
from __future__ import annotations

import src.webapp._compat  # noqa: F401

import argparse
from pathlib import Path

from nicegui import ui

from src.utils.config import load_config
from src.webapp.dashboard import create_dashboard
from src.webapp.dashboard_data import _latest_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training / warm-start pipeline dashboard.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Fallback config path.")
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory to load (e.g. outputs/runs/run_20260729_140606). Default: latest by mtime.",
    )
    parser.add_argument("--port", type=int, default=8080, help="Port to serve the dashboard on.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host/interface to bind.")
    return parser.parse_args()


args = parse_args()
config = load_config(args.config)
outputs_root = Path(config["outputs"]["root"])

run_dir = Path(args.run) if args.run else _latest_run_dir(outputs_root)
if run_dir is not None:
    run_dir = Path(run_dir)
    # Prefer the run's own config when present.
    run_cfg = run_dir / "config.yaml"
    if run_cfg.is_file():
        try:
            config = load_config(str(run_cfg))
        except Exception:
            pass
    # Live RL log inside the run, then legacy global log.
    candidates = [
        run_dir / "logs" / "train_log.jsonl",
        run_dir / "rl" / "train_log.jsonl",
        outputs_root / "logs" / "train_log.jsonl",
    ]
    log_path = next((p for p in candidates if p.is_file() and p.stat().st_size > 0), None)
    if log_path is None:
        log_path = next((p for p in candidates if p.is_file()), candidates[0])
    print(f"Dashboard loading run: {run_dir}")
    print(f"Metrics log: {log_path}")
else:
    log_path = outputs_root / "logs" / "train_log.jsonl"
    print("No run directory found under outputs/runs/ — waiting for training metrics.")

create_dashboard(
    log_path,
    config_path=Path(args.config),
    outputs_root=outputs_root,
)

ui.run(
    title="Transformer MARL · Schneider EVRPTW Dashboard",
    host=args.host,
    port=args.port,
    reload=False,
    show=True,
)
