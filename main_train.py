"""Entry point for the Expert -> BC -> RL training pipeline.

Full pipeline:
    python main_train.py --config configs/train.yaml --device cuda

Single stage (later stages reuse the latest run folder automatically):
    python main_train.py --config configs/train.yaml --device cuda --stage expert
    python main_train.py --config configs/train.yaml --device cuda --stage demos
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.utils.config import load_config

STAGE_CHOICES = ("all", "checks", "expert", "demos", "bc", "rl")
STAGES_CONTINUE_LATEST = frozenset({"demos", "bc", "rl"})


def _apply_stage_flags(config: dict, stage: str) -> None:
    """Enable only the requested pipeline stage (or all stages)."""
    pipe = config.setdefault("pipeline", {})
    if stage == "all":
        return
    # Force the selected stage to run (do not skip via cache reuse).
    pipe["reuse_cached_stages"] = False
    pipe["run_internal_checks"] = stage == "checks"
    pipe["run_expert"] = stage == "expert"
    pipe["generate_demonstrations"] = stage == "demos"
    pipe["run_behavior_cloning"] = stage == "bc"
    pipe["run_rl_finetuning"] = stage == "rl"
    pipe["run_final_evaluation"] = False


def _latest_run_dir(config: dict) -> Path | None:
    root = Path((config.get("outputs") or {}).get("root", "outputs")) / "runs"
    if not root.is_dir():
        return None
    runs = sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return runs[0] if runs else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Warm-start pipeline: expert -> BC -> RL fine-tune.",
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
        "--stage",
        type=str,
        default="all",
        choices=STAGE_CHOICES,
        help="Run one pipeline stage, or 'all' for the full pipeline (default).",
    )
    # Advanced / internal only.
    parser.add_argument("--resume", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--resume-run", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--rl-only", action="store_true", help=argparse.SUPPRESS)
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

    _apply_stage_flags(config, args.stage)

    from src.utils.gpu_runtime import configure_cuda_runtime

    gpu_cfg = config.get("gpu") or {}
    configure_cuda_runtime(
        cudnn_benchmark=bool(gpu_cfg.get("cudnn_benchmark", True)),
        allow_tf32=bool(gpu_cfg.get("allow_tf32", True)),
    )

    # Later stages continue in the latest run folder automatically.
    resume_run = args.resume_run
    if resume_run is None and args.stage in STAGES_CONTINUE_LATEST:
        latest = _latest_run_dir(config)
        if latest is None:
            print(
                f"No run folder found under outputs/runs/. "
                f"Run --stage expert first.",
                file=sys.stderr,
            )
            raise SystemExit(1)
        resume_run = str(latest)

    if args.rl_only:
        from src.training.train_rl import train

        train(config)
        return

    from src.training.pipeline import PipelineError, run_training_pipeline

    print("=" * 60)
    label = "Full pipeline" if args.stage == "all" else f"Stage: {args.stage}"
    print(f"{label} | config: {args.config}")
    if resume_run:
        print(f"Run directory: {resume_run}")
    print("=" * 60)
    try:
        run_dir = run_training_pipeline(
            config,
            config_path=args.config,
            resume_run=resume_run,
        )
    except PipelineError as exc:
        print(f"PIPELINE STOPPED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Done. Run directory: {run_dir}")
    print("Dashboard: python main_dashboard.py")


if __name__ == "__main__":
    main()
