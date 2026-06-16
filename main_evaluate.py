"""Entry point for evaluating the Multi-Agent Transformer E-VRPTW solver."""
from __future__ import annotations

import argparse

from src.evaluation.evaluate import evaluate
from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the E-VRPTW Transformer solver.")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Path to YAML config.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a model checkpoint.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    evaluate(config, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
