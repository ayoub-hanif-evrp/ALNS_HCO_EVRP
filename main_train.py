"""Entry point for training the Multi-Agent Transformer E-VRPTW solver.

Two training modes:

* Flat REINFORCE (default):
      python main_train.py --config configs/default.yaml

* Curriculum (Phase 0 checks + phased C -> C+R -> C+R+RC -> hard cases):
      python main_train.py --config configs/curriculum.yaml --curriculum

* Phase 0 environment/mask checks only (no training):
      python main_train.py --config configs/curriculum.yaml --phase0-only
"""
from __future__ import annotations

import argparse

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the E-VRPTW Transformer solver.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs (flat mode).")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--curriculum", action="store_true", help="Run phased curriculum training.")
    parser.add_argument("--phase0-only", action="store_true", help="Run only the Phase 0 checks and exit.")
    parser.add_argument("--force", action="store_true", help="Train even if Phase 0 checks fail.")
    return parser.parse_args()


def _run_phase0_only(config: dict) -> None:
    from src.training.phase0_checks import (
        format_report,
        phase0_passed,
        pick_check_instance,
        run_phase0_checks,
    )

    instance = pick_check_instance(config["dataset"]["path"])
    results = run_phase0_checks(instance)
    print(format_report(results))
    raise SystemExit(0 if phase0_passed(results) else 1)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed

    if args.phase0_only:
        _run_phase0_only(config)
        return

    if args.curriculum or config.get("curriculum"):
        from src.training.curriculum_train import train_curriculum

        train_curriculum(config, run_phase0=True, force=args.force)
        return

    from src.training.train_rl import train

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    train(config)


if __name__ == "__main__":
    main()
