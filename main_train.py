"""Entry point for the full Expert -> BC -> RL training pipeline.

    python main_train.py --config configs/train.yaml --device cuda
"""
from __future__ import annotations

import argparse
import sys

from src.utils.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Full warm-start pipeline: expert -> BC -> RL fine-tune -> evaluation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config (default: configs/train.yaml).",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Optional override for training.cycles (RL fine-tune).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume RL from a checkpoint (advanced).",
    )
    # Kept for developer debugging only — not part of the normal two-command workflow.
    parser.add_argument(
        "--rl-only",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    tcfg = config.setdefault("training", {})
    if args.cycles is not None:
        tcfg["cycles"] = int(args.cycles)
    if args.resume is not None:
        tcfg["resume_from"] = args.resume

    # Developer escape hatch: skip pipeline orchestration.
    if args.rl_only:
        from src.training.train_rl import train

        train(config)
        return

    from src.training.pipeline import PipelineError, run_training_pipeline

    print("=" * 60)
    print(f"Full pipeline | config: {args.config}")
    print("=" * 60)
    try:
        run_dir = run_training_pipeline(config, config_path=args.config)
    except PipelineError as exc:
        print(f"PIPELINE STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Done. Run directory: {run_dir}")
    print("Dashboard: python main_dashboard.py")


if __name__ == "__main__":
    main()
