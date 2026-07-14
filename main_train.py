"""Entry point for the exact-mask semi-MDP training pipeline.

Three training phases (choose via config):

  Phase 1 — Single RC:
    python main_train.py --config configs/train_phase1_single_RC21.yaml --device cuda

  Phase 2 — Full RC:
    python main_train.py --config configs/train_phase2_full_RC21.yaml --device cuda

  Phase 3 — All classes:
    python main_train.py --config configs/train_phase3_all_classes.yaml --device cuda

Sanity only (not a training phase):

    python main_train.py --config configs/train_phase1_single_RC21.yaml --phase0-only
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from src.utils.config import load_config

# Pytest cases that do not load the pandas/pyarrow instance fixture (Windows-safe).
_QUICK_TEST_TARGETS = [
    "tests/test_scientific_semantics.py::test_critic_remaining_cost_targets",
    "tests/test_scientific_semantics.py::test_normalizer_checkpoint_roundtrip",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exact-mask semi-MDP training (Phase 1 / 2 / 3 via config).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_phase1_single_RC21.yaml",
        help="Path to YAML config (default: Phase 1).",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Optional override for training.cycles (Phase 2/3).",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="Optional override for training.updates (Phase 1).",
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
        help="Run only Phase 0 environment/mask sanity checks and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if Phase 0 or quick tests fail.",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip Phase 0 sanity checks before training.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip fast deterministic unit tests before training.",
    )
    parser.add_argument(
        "--run-gates",
        action="store_true",
        help="Also run Phase 0 + quick tests before this training phase (optional).",
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

    # Default: training commands do not re-run gates (see COMMANDS.md §0).
    run_gates = bool(args.run_gates) and not args.skip_phase0
    if run_gates:
        print("=" * 60)
        print("Optional gates — Phase 0 sanity checks")
        print("=" * 60)
        if not _run_phase0(config):
            if not args.force:
                print("Phase 0 failed. Fix issues or pass --force to continue.")
                raise SystemExit(1)
            print("WARNING: Phase 0 failed but --force set; continuing.")

    if run_gates and not args.skip_tests:
        print("=" * 60)
        print("Optional gates — quick deterministic unit tests")
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

    from src.data.training_phases import normalize_training_phase
    from src.training.train_rl import train

    phase = normalize_training_phase(config)
    print("=" * 60)
    print(f"Training — phase: {phase}")
    print(f"Config: {args.config}")
    print("=" * 60)
    train(config)


if __name__ == "__main__":
    main()
