"""Entry point for exact-mask semi-MDP training.

Quick test:

    python main_train.py --phase0-only

Full training:

    python main_train.py --config configs/train.yaml --device cuda
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.utils.config import load_config

_QUICK_TEST_TARGETS = [
    "tests/test_scientific_semantics.py::test_critic_remaining_cost_targets",
    "tests/test_scientific_semantics.py::test_normalizer_checkpoint_roundtrip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-mask semi-MDP training (C/R/RC interleaving via configs/train.yaml).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to YAML config (default: configs/train.yaml).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Optional override for training.cycles.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Optional override for training.updates (single-instance debug configs).",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from a checkpoint (actor, critic, optimizers, normalizers, counters).",
    )
    parser.add_argument(
        "--phase0-only",
        action="store_true",
        help="Run only environment/mask sanity checks and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if optional gates fail.",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip sanity checks when using --run-gates.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip quick unit tests when using --run-gates.",
    )
    parser.add_argument(
        "--run-gates",
        action="store_true",
        help="Also run sanity checks + quick tests before training (optional).",
    )
    return parser.parse_args()


def _run_phase0(config: dict) -> bool:
    from src.training.phase0_checks import (
        format_report,
        phase0_passed,
        pick_check_instance,
        run_phase0_checks,
    )

    instance = pick_check_instance(config["dataset"]["path"])
    results = run_phase0_checks(instance)
    print(format_report(results))
    return phase0_passed(results)


def _run_quick_tests() -> bool:
    cmd = [sys.executable, "-m", "pytest", *_QUICK_TEST_TARGETS, "-q", "--tb=short"]
    print("Running quick unit tests:", " ".join(_QUICK_TEST_TARGETS))
    proc = subprocess.run(cmd, check=False)
    return proc.returncode == 0


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed

    if args.phase0_only:
        ok = _run_phase0(config)
        raise SystemExit(0 if ok else 1)

    run_gates = bool(args.run_gates) and not args.skip_phase0
    if run_gates:
        print("=" * 60)
        print("Optional gates — environment/mask sanity checks")
        print("=" * 60)
        if not _run_phase0(config):
            if not args.force:
                print("Sanity checks failed. Fix issues or pass --force to continue.")
                raise SystemExit(1)
            print("WARNING: sanity checks failed but --force set; continuing.")

    if run_gates and not args.skip_tests:
        print("=" * 60)
        print("Optional gates — quick unit tests")
        print("=" * 60)
        if not _run_quick_tests():
            if not args.force:
                print("Quick tests failed. Fix issues or pass --force to continue.")
                raise SystemExit(1)
            print("WARNING: quick tests failed but --force set; continuing.")

    tcfg = config.setdefault("training", {})
    if args.cycles is not None:
        tcfg["cycles"] = int(args.cycles)
    if args.updates is not None:
        tcfg["updates"] = int(args.updates)
    if args.epochs is not None:
        tcfg["epochs"] = args.epochs
    if args.resume is not None:
        tcfg["resume_from"] = args.resume

    from src.training.train_rl import train

    print("=" * 60)
    print(f"Full training | config: {args.config}")
    print("=" * 60)
    train(config)


if __name__ == "__main__":
    main()
