#!/usr/bin/env python3
"""Minimal training-step speed benchmark (no full experiment).

Runs a few policy decisions / short episodes on 100-customer C/R/RC instances
and reports decision timing + duplicate-call counters.

Usage:
  python scripts/benchmark_training_step.py
  python scripts/benchmark_training_step.py --device cuda --max-decisions 30
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data.instance_loader import load_instance
from src.env.analysis_counters import (
    expected_per_decision_ok,
    reset_all_counters,
    snapshot_for_log,
    get_cycle_counters,
)
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.models.policy import TransformerEVRPTWPolicy
from src.training.actor_critic import RLConfig
from src.training.rollout_ac import run_episode_trajectory


DATASET = ROOT / "dataset" / "Schneider" / "evrptw_instances"


def _find(name: str) -> Path | None:
    if not DATASET.exists():
        return None
    hits = list(DATASET.rglob(f"{name}.txt")) + list(DATASET.rglob(name))
    return hits[0] if hits else None


def _load_cfg() -> dict:
    path = ROOT / "configs" / "train.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-decisions", type=int, default=40)
    ap.add_argument("--instances", nargs="*", default=["c101_21", "r101_21", "rc101_21"])
    args = ap.parse_args()

    cfg = _load_cfg()
    device = torch.device(args.device)
    portions = list((cfg.get("problem") or {}).get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    # Tiny model for a fast representative slice (same mask path as training).
    model_cfg = dict(cfg.get("model") or {})
    model_cfg.update({
        "embedding_dim": 64,
        "encoder_layers": 1,
        "heads": 4,
        "feedforward_dim": 128,
        "dropout": 0.0,
        "torch_compile": False,
    })
    policy = TransformerEVRPTWPolicy(model_cfg, charging_portions=portions).to(device)
    policy.eval()
    weights = CostWeights.from_config(cfg.get("cost") or {})
    rl_cfg = RLConfig.from_config(cfg)
    env_cfg = cfg.get("environment") or {}

    decision_secs: list[float] = []
    episode_secs: list[float] = []
    analysis_fracs: list[float] = []
    cache_hits = 0
    cache_misses = 0

    reset_all_counters()
    for name in args.instances:
        path = _find(name)
        if path is None:
            print(f"SKIP missing instance {name}")
            continue
        inst = load_instance(path)
        t0 = time.perf_counter()
        result, traj = run_episode_trajectory(
            inst,
            policy,
            m=min(8, max(3, inst.num_customers // 20)),
            weights=weights,
            mode="greedy",
            device=device,
            charging_portions=portions,
            use_horizon_check=True,
            use_depot_charging=True,
            max_steps_factor=2,
            rl_cfg=rl_cfg,
            collect_trajectory=True,
            training_mode=True,
            environment_config=env_cfg,
            use_amp=False,
        )
        # Cap reported decisions for partial-episode style stats.
        n_dec = min(int(result.num_decisions), int(args.max_decisions))
        ep_s = time.perf_counter() - t0
        episode_secs.append(ep_s)
        ts = getattr(traj, "training_stats", None) or {}
        analysis_ms = float(ts.get("decision_analysis_ms", 0.0))
        total_ms = max(ep_s * 1000.0, 1e-6)
        analysis_fracs.append(100.0 * analysis_ms / total_ms)
        if n_dec > 0:
            decision_secs.append(ep_s / max(int(result.num_decisions), 1))
        metrics = getattr(result.env, "_cache_metrics", None) or {}
        cache_hits += int(metrics.get("hits", 0))
        cache_misses += int(metrics.get("misses", 0))
        print(
            f"{name}: decisions={result.num_decisions} "
            f"episode_s={ep_s:.3f} analysis_ms={analysis_ms:.1f} "
            f"done={result.done_reason}"
        )

    counters = snapshot_for_log(get_cycle_counters())
    warns = expected_per_decision_ok(counters)
    hit_rate = cache_hits / max(cache_hits + cache_misses, 1)

    print("\n=== Training-step speed benchmark ===")
    if decision_secs:
        print(f"avg seconds / policy decision: {statistics.mean(decision_secs):.4f}")
    if episode_secs:
        print(f"avg seconds / episode:         {statistics.mean(episode_secs):.3f}")
    if analysis_fracs:
        print(f"decision analysis % of time:   {statistics.mean(analysis_fracs):.1f}%")
    print(f"cache hit rate:                {hit_rate:.3f} ({cache_hits}/{cache_hits + cache_misses})")
    print("duplicate-call counters:")
    for k, v in counters.items():
        print(f"  {k}: {v}")
    if warns:
        print("WARNINGS:", ", ".join(warns))
        return 1
    print("OK: no hot-path duplicate-call warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
