"""Entry point for the exact-mask semi-MDP training pipeline.

Unified command (recommended):

    python main_train.py --cycles 5 --device cuda

Pipeline order:
  1. Phase 0 environment/mask sanity checks
  2. Fast deterministic unit tests (no dataset fixture)
  3. One-instance overfit gate on rc102_21 (unless skipped)
  4. Full RC _21 cyclic training

Cycle modes:
  --cycles 0   overfit gate only (Phase 0 + tests + 500 updates on one instance)
  --cycles N   gates then train all 13 RC _21 instances for N cycles (default from config: 100)

Examples:
    python main_train.py --cycles 0          # overfit gate only
    python main_train.py --cycles 5          # gates + 5 training cycles
    python main_train.py --phase0-only       # sanity checks only
    python main_train.py --cycles 5 --skip-overfit --skip-tests
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
        description="Exact-mask semi-MDP training pipeline (Phase 0 -> tests -> overfit -> train).",
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
        help=(
            "Training cycles: 0 = overfit gate only; N>0 = full cyclic training "
            "after gates (default: training.cycles from config)."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--device", type=str, default=None, help="Override device (cpu/cuda/auto).")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument(
        "--overfit",
        action="store_true",
        help="Alias for --cycles 0 (overfit gate only).",
    )
    parser.add_argument(
        "--phase0-only",
        action="store_true",
        help="Run only Phase 0 checks and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Continue even if Phase 0 or quick tests fail.",
    )
    parser.add_argument(
        "--skip-phase0",
        action="store_true",
        help="Skip Phase 0 sanity checks.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip fast deterministic unit tests.",
    )
    parser.add_argument(
        "--skip-overfit",
        action="store_true",
        help="Skip one-instance overfit gate before cyclic training.",
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


def _resolve_cycles(args: argparse.Namespace, config: dict) -> int:
    if args.overfit:
        return 0
    if args.cycles is not None:
        return int(args.cycles)
    return int(config.get("training", {}).get("cycles", 100))


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

    if not args.skip_phase0:
        print("=" * 60)
        print("Stage 1a — Phase 0 sanity checks")
        print("=" * 60)
        if not _run_phase0(config):
            if not args.force:
                print("Phase 0 failed. Fix issues or pass --force to continue.")
                raise SystemExit(1)
            print("WARNING: Phase 0 failed but --force set; continuing.")

    if not args.skip_tests:
        print("=" * 60)
        print("Stage 1b — quick deterministic unit tests")
        print("=" * 60)
        if not _run_quick_tests():
            if not args.force:
                print("Quick tests failed. Fix issues or pass --force to continue.")
                raise SystemExit(1)
            print("WARNING: quick tests failed but --force set; continuing.")

    cycles = _resolve_cycles(args, config)
    tcfg = config.setdefault("training", {})

    if args.epochs is not None:
        tcfg["epochs"] = args.epochs

    from src.training.train_rl import train

    run_overfit = cycles == 0 or not args.skip_overfit

    if run_overfit:
        print("=" * 60)
        print("Stage 3 — one-instance overfit gate (rc102_21)")
        print("=" * 60)
        overfit_cfg = dict(config)
        overfit_cfg["training"] = dict(tcfg)
        overfit_cfg["training"]["debug_overfit_one_instance"] = True
        train(overfit_cfg)
        if cycles == 0:
            print("Overfit-only run complete (--cycles 0).")
            return

    print("=" * 60)
    print(f"Stage 4+ — cyclic training ({cycles} cycles on 13 RC _21 instances)")
    print("=" * 60)
    train_cfg = dict(config)
    train_cfg["training"] = dict(tcfg)
    train_cfg["training"]["debug_overfit_one_instance"] = False
    train_cfg["training"]["cycles"] = cycles
    train(train_cfg)


if __name__ == "__main__":
    main()
