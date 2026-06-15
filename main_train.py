"""Entry point for training the Multi-Agent Transformer E-VRPTW solver."""
from __future__ import annotations

import argparse

from src.training.train_rl import train
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the E-VRPTW Transformer solver.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    train(config)


if __name__ == "__main__":
    main()
