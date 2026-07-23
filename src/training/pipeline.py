"""Single-command training pipeline: Expert -> BC -> RL fine-tune -> Eval.

Called from ``main_train.py``. Stages write under one run directory:

    outputs/runs/run_<timestamp>/
        config.yaml
        expert/
        demonstrations/
        bc/
        rl/
        evaluation/
        checkpoints/
        logs/
        metrics/
        dashboard/
"""
from __future__ import annotations

import compileall
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import yaml

from src.utils.logging_utils import get_logger

LOGGER = get_logger()

STAGE_DIRS = (
    "dataset",
    "expert",
    "demonstrations",
    "bc",
    "rl",
    "evaluation",
    "checkpoints",
    "logs",
    "metrics",
    "dashboard",
)


class PipelineError(RuntimeError):
    """Fatal stage gate failure - training must stop."""


def _pipeline_flags(config: dict) -> dict:
    raw = dict(config.get("pipeline") or {})
    defaults = {
        "run_internal_checks": True,
        "run_internal_sanity_checks": True,
        "run_expert": True,
        "generate_demonstrations": True,
        "run_behavior_cloning": True,
        "run_rl_finetuning": True,
        "run_final_evaluation": True,
        "reuse_cached_stages": True,
    }
    out = {**defaults, **raw}
    if out.get("run_internal_sanity_checks") and not raw.get("run_internal_checks"):
        out["run_internal_checks"] = True
    if config.get("expert") and config["expert"].get("enabled") is False:
        out["run_expert"] = False
        out["generate_demonstrations"] = False
    if config.get("behavior_cloning") and config["behavior_cloning"].get("enabled") is False:
        out["run_behavior_cloning"] = False
    rl = config.get("rl") or {}
    if rl.get("enabled") is False:
        out["run_rl_finetuning"] = False
    return out


def _config_fingerprint(config: dict) -> str:
    payload = {
        "seed": config.get("seed"),
        "dataset": config.get("dataset"),
        "environment": config.get("environment"),
        "constraints": config.get("constraints"),
        "expert": config.get("expert"),
        "behavior_cloning": {
            k: v for k, v in (config.get("behavior_cloning") or {}).items()
            if k != "output_dir"
        },
        "model": config.get("model"),
        "problem": config.get("problem"),
        "cost": config.get("cost"),
        "rl_bc_reg": (config.get("rl") or {}).get("bc_regularization"),
        "training_cycles": (config.get("training") or {}).get("cycles"),
        "optimizer": config.get("optimizer"),
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def create_run_directory(config: dict) -> Path:
    root = Path((config.get("outputs") or {}).get("root", "outputs"))
    run_id = time.strftime("run_%Y%m%d_%H%M%S")
    run_dir = root / "runs" / run_id
    for name in STAGE_DIRS:
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8",
    )
    (run_dir / "pipeline_fingerprint.txt").write_text(
        _config_fingerprint(config), encoding="utf-8",
    )
    LOGGER.info("Pipeline run directory: %s", run_dir)
    return run_dir


def _wire_run_outputs(config: dict, run_dir: Path) -> dict:
    cfg = json.loads(json.dumps(config, default=str))
    cfg["pipeline_run_dir"] = str(run_dir)
    cfg["outputs"] = {
        **(cfg.get("outputs") or {}),
        "root": str(Path(run_dir).parents[1]),
        "checkpoints": str(run_dir / "checkpoints"),
        "logs": str(run_dir / "logs"),
        "results": str(run_dir / "metrics"),
        "figures": str(run_dir / "dashboard" / "figures"),
        "routes": str(run_dir / "evaluation" / "routes"),
        "traces": str(run_dir / "evaluation" / "traces"),
        "run_dir": str(run_dir),
        "rl_dir": str(run_dir / "rl"),
    }
    cfg.setdefault("demonstrations", {})
    cfg["demonstrations"]["output"] = str(run_dir / "demonstrations" / "hanif_expert_train.pt")
    cfg["demonstrations"]["episodes_output"] = str(run_dir / "demonstrations" / "expert_episodes.pt")
    cfg["demonstrations"]["eval_report"] = str(run_dir / "expert" / "expert_metrics.json")
    cfg["demonstrations"]["input"] = str(run_dir / "demonstrations" / "hanif_expert_train.pt")
    cfg.setdefault("behavior_cloning", {})
    cfg["behavior_cloning"]["output_dir"] = str(run_dir / "bc")
    cfg["behavior_cloning"]["early_stopping"] = False
    cfg.setdefault("initialization", {})
    cfg["initialization"]["actor_checkpoint"] = str(run_dir / "bc" / "bc_best_policy.pt")
    cfg["initialization"]["load_actor"] = True
    cfg["initialization"]["load_critic"] = False
    cfg["initialization"]["reset_optimizer"] = True
    cfg.setdefault("evaluation", {})
    cfg["evaluation"]["run_after_training"] = False
    return cfg


def stage_hanif_dataset(config: dict, run_dir: Path) -> tuple[dict, dict]:
    """Parse/convert Hanif, write compatibility report + deterministic splits."""
    LOGGER.info("=" * 60)
    LOGGER.info("Stage — Hanif dataset report + deterministic split")
    LOGGER.info("=" * 60)
    from src.data.hanif_dataset import (
        HanifDatasetError,
        build_hanif_dataset_report,
        resolve_hanif_split,
        save_split_into_config,
    )

    try:
        train, validation, test, meta = resolve_hanif_split(config)
        save_split_into_config(config, train, validation, test, meta)
        report = build_hanif_dataset_report(config, run_dir=run_dir, split_meta=meta)
    except HanifDatasetError as exc:
        raise PipelineError(str(exc)) from exc
    LOGGER.info(
        "Hanif split | train=%d val=%d test=%d | pilot=%s",
        len(train), len(validation), len(test), meta.get("pilot_enabled"),
    )
    return report, meta


def run_internal_checks(config: dict) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage — internal sanity checks")
    LOGGER.info("=" * 60)

    root = Path(__file__).resolve().parents[2]
    ok = compileall.compile_dir(str(root / "src" / "expert"), quiet=1)
    ok = compileall.compile_dir(str(root / "src" / "training"), quiet=1) and ok
    if not ok:
        raise PipelineError("compileall failed on src/expert or src/training")

    from src.training.config_validation import validate_training_config

    try:
        validate_training_config(config)
    except Exception as exc:
        raise PipelineError(f"Config validation failed: {exc}") from exc

    from src.data.training_phases import resolve_training_phase_split

    try:
        train_names, eval_names, meta = resolve_training_phase_split(config)
    except Exception as exc:
        raise PipelineError(f"Dataset split validation failed: {exc}") from exc
    if not train_names:
        raise PipelineError("Training split is empty")
    LOGGER.info(
        "Dataset OK | train=%d eval=%d phase=%s",
        len(train_names), len(eval_names), meta.get("training_phase"),
    )

    from src.data.instance_loader import load_instance
    from src.data.hanif_dataset import is_hanif_config

    ds_path = Path(config["dataset"]["path"])
    train_names = list((config.get("dataset") or {}).get("train_instances") or [])
    small = ds_path / (train_names[0] if train_names else "C1_10_1")
    if not small.exists():
        from src.data.instance_loader import list_instance_dirs
        dirs = list_instance_dirs(ds_path)
        if not dirs:
            raise PipelineError(f"No instances under {ds_path}")
        small = dirs[0]
    inst = load_instance(small)

    if is_hanif_config(config) and inst.num_customers >= 500:
        # Full exact-mask analysis on 1000-customer Hanif is too heavy for sanity checks.
        # Validate loader + station attrs + policy import only; expert legality runs in demos.
        if inst.station_wait_mu is None or inst.station_density_score is None:
            raise PipelineError(
                f"Hanif smoke: station wait/density missing on {inst.instance_name}"
            )
        from src.training.train_rl import build_policy

        _ = build_policy(config, list(config["problem"]["charging_portions"]))
        LOGGER.info(
            "Hanif smoke OK (load+attrs+policy) | %s N=%d customers=%d "
            "(skipped full exact-mask analyze in checks)",
            inst.instance_name, inst.num_nodes, inst.num_customers,
        )
        return

    from src.env.evrptw_env import EVRPTWEnv
    from src.env.policy_action import apply_policy_action
    from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert

    env = EVRPTWEnv(
        inst, m=2,
        charging_portions=list(config["problem"]["charging_portions"]),
        use_horizon_check=True,
        use_depot_charging=True,
        hard_time_windows=True,
    )
    env.environment_config = config.get("environment") or {}
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(config))
    ev = env.select_acting_ev()
    if ev is None:
        raise PipelineError("Expert smoke: no acting EV")
    analysis = env.analyze_decision(ev)
    if env.try_controlled_closure_from_analysis(ev, analysis) is None:
        if env.try_auto_close_from_analysis(ev, analysis) is None:
            action = expert.select_action(env, ev, analysis)
            if not bool(analysis.final_node_mask[action.node]):
                raise PipelineError("Expert smoke: selected masked node")
            apply_policy_action(env, ev, action, mask_result=analysis.mask_result)
    LOGGER.info("Exact-mask + expert legality smoke OK (%s)", inst.instance_name)


def _write_expert_artifacts(run_dir: Path, report: dict) -> Path:
    metrics_path = run_dir / "expert" / "expert_metrics.json"
    traj_path = run_dir / "expert" / "expert_trajectories_summary.csv"
    metrics_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    rows = report.get("per_instance") or []
    if rows:
        keys = sorted({k for r in rows for k in r.keys() if k != "breakdown"})
        with traj_path.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in keys})
    else:
        traj_path.write_text("instance_name\n", encoding="utf-8")
    return metrics_path


def stage_expert(config: dict, run_dir: Path, *, reuse: bool) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 1 — expert evaluation")
    LOGGER.info("=" * 60)
    metrics_path = run_dir / "expert" / "expert_metrics.json"
    traj_path = run_dir / "expert" / "expert_trajectories_summary.csv"
    if reuse and metrics_path.is_file() and traj_path.is_file():
        LOGGER.info("Reusing cached expert metrics: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    from src.expert.generate_demonstrations import evaluate_expert

    report = evaluate_expert(config, on_holdout=False)
    if not report.get("hard_violation_free", False):
        raise PipelineError(
            "Expert produced hard EVRPTW violations "
            f"(battery={report.get('mean_battery_violations')}, "
            f"tw={report.get('mean_time_window_violations')}, "
            f"cap={report.get('mean_capacity_violations')})"
        )
    metrics_path = _write_expert_artifacts(run_dir, report)
    LOGGER.info(
        "Expert done | served=%.3f feas=%.3f -> %s",
        report["mean_served_ratio"], report["feasibility_rate"], metrics_path,
    )
    return report


def stage_expert_from_demonstrations(
    config: dict,
    run_dir: Path,
    demo_path: Path,
    *,
    reuse: bool,
) -> dict:
    """Write expert metrics from demo episodes (same rollouts — no second expert pass)."""
    LOGGER.info("=" * 60)
    LOGGER.info("Writing expert metrics from demos (no second rollout)")
    LOGGER.info("=" * 60)
    metrics_path = run_dir / "expert" / "expert_metrics.json"
    traj_path = run_dir / "expert" / "expert_trajectories_summary.csv"
    if reuse and metrics_path.is_file() and traj_path.is_file():
        LOGGER.info("Reusing cached expert metrics: %s", metrics_path)
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    import torch

    from src.expert.generate_demonstrations import (
        expert_report_from_demo_episodes,
        log_expert_report,
    )

    payload = torch.load(demo_path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes") or []
    if not episodes and payload.get("decisions") is not None:
        # Decision-only cache: rebuild lightweight metrics from embedded episode copies if present.
        raise PipelineError(
            "Expert metrics require episode rollouts in the demonstration file; "
            "episodes missing from decision-level cache."
        )
    meta = dict(payload.get("meta") or {})
    report = expert_report_from_demo_episodes(episodes, split="train", meta=meta)
    log_expert_report(report)
    if not report.get("hard_violation_free", False):
        raise PipelineError(
            "Expert/demos produced hard EVRPTW violations "
            f"(battery={report.get('mean_battery_violations')}, "
            f"tw={report.get('mean_time_window_violations')}, "
            f"cap={report.get('mean_capacity_violations')})"
        )
    metrics_path = _write_expert_artifacts(run_dir, report)
    LOGGER.info(
        "Expert metrics from demos | served=%.3f feas=%.3f -> %s",
        report["mean_served_ratio"], report["feasibility_rate"], metrics_path,
    )
    return report


def stage_demonstrations(config: dict, run_dir: Path, *, reuse: bool) -> Path:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 1 — expert demonstrations (decision-level cache)")
    LOGGER.info("=" * 60)
    out_path = Path(config["demonstrations"]["output"])
    episodes_path = Path(
        (config.get("demonstrations") or {}).get(
            "episodes_output", run_dir / "demonstrations" / "expert_episodes.pt",
        )
    )
    summary_path = run_dir / "demonstrations" / "demonstration_summary.json"
    if reuse and out_path.is_file() and summary_path.is_file():
        LOGGER.info("Reusing cached decision-level demonstrations: %s", out_path)
        return out_path

    from src.data.evrptw_dataset import split_dataset
    from src.data.instance_loader import load_capacity_from_config
    from src.data.training_phases import resolve_training_phase_split
    from src.expert.generate_demonstrations import generate_demonstrations
    from src.training.behavior_cloning import build_bc_decision_cache
    import torch

    # Generate episode rollouts first (internal artifact).
    cfg = json.loads(json.dumps(config, default=str))
    cfg.setdefault("demonstrations", {})
    cfg["demonstrations"]["output"] = str(episodes_path)
    t0 = time.time()
    path = generate_demonstrations(cfg)
    expert_generation_time = time.time() - t0

    payload = torch.load(path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes") or []
    if not episodes:
        raise PipelineError("Demonstration file is empty - cannot continue to BC")
    n_actions = sum(len(ep.get("actions") or []) for ep in episodes)
    if n_actions == 0:
        raise PipelineError("Demonstrations contain zero actions - cannot continue to BC")

    train_names, eval_names, _ = resolve_training_phase_split(config)
    ds_cfg = dict(config["dataset"])
    train_ds, _ = split_dataset(
        ds_cfg["path"],
        train_ratio=0.8,
        seed=int(config.get("seed", 42)),
        train_filter=ds_cfg.get("train_size_filter") if ds_cfg.get("train_size_filter") is not None else [],
        eval_filter=ds_cfg.get("eval_size_filter") if ds_cfg.get("eval_size_filter") is not None else [],
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_capacity_from_config(config),
    )
    instance_by_name = {n: train_ds[train_ds.names.index(n)] for n in train_ds.names}
    portions = list(config["problem"]["charging_portions"])
    demo_workers = int((config.get("demonstrations") or {}).get("num_workers", 6) or 6)
    decisions = build_bc_decision_cache(
        episodes,
        instance_by_name,
        charging_portions=portions,
        environment_config=config.get("environment") or {},
        num_workers=demo_workers,
    )
    if not decisions:
        raise PipelineError("Decision-level demonstration cache is empty")

    decision_payload = {
        "decisions": decisions,
        "episodes": episodes,
        "meta": {
            **(payload.get("meta") or {}),
            "format": "hanif_decision_level_v1",
            "n_decisions": len(decisions),
            "n_episodes": len(episodes),
            "expert_generation_time": expert_generation_time,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(decision_payload, out_path)

    summary = {
        "path": str(out_path),
        "episodes_path": str(path),
        "n_episodes": len(episodes),
        "n_actions": n_actions,
        "n_decisions": len(decisions),
        "expert_generation_time": expert_generation_time,
        "instances": sorted({ep.get("instance_name") for ep in episodes}),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    LOGGER.info(
        "Demonstrations OK | episodes=%d decisions=%d -> %s",
        len(episodes), len(decisions), out_path,
    )
    return out_path


def stage_behavior_cloning(config: dict, run_dir: Path, *, reuse: bool) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 2 — behavior cloning")
    LOGGER.info("=" * 60)
    bcfg = config.setdefault("behavior_cloning", {})
    bcfg["early_stopping"] = False
    out_dir = Path(bcfg.get("output_dir", run_dir / "bc"))
    best = out_dir / "bc_best_policy.pt"
    last = out_dir / "bc_last_policy.pt"
    eval_json = out_dir / "bc_eval_metrics.json"
    if reuse and best.is_file() and last.is_file() and eval_json.is_file():
        LOGGER.info("Reusing cached BC checkpoints: %s", best)
        return json.loads(eval_json.read_text(encoding="utf-8"))

    from src.training.behavior_cloning import evaluate_bc_policy, train_behavior_cloning
    from src.training.train_rl import load_policy
    from src.utils.seed import resolve_device

    # Resume path: checkpoints exist but greedy eval crashed / was never written.
    if reuse and best.is_file() and last.is_file() and not eval_json.is_file():
        if not bool(bcfg.get("run_greedy_eval", False)):
            LOGGER.info("Reusing BC checkpoints (greedy eval skipped): %s", best)
            stub = {
                "skipped": True,
                "best_checkpoint": str(best),
                "last_checkpoint": str(last),
            }
            eval_json.write_text(json.dumps(stub, indent=2), encoding="utf-8")
            return stub
        LOGGER.info(
            "Reusing BC checkpoints; running greedy eval only (skipping BC train): %s",
            best,
        )
        device = resolve_device(config.get("device", "auto"))
        policy = load_policy(best, device=device)
        eval_report = evaluate_bc_policy(config, policy, device)
        eval_json.write_text(json.dumps(eval_report, indent=2, default=str), encoding="utf-8")
        summary = {
            "best_checkpoint": str(best),
            "last_checkpoint": str(last),
            "epochs_completed": "reused",
            "early_stopping": False,
            "greedy_eval": eval_report,
            "resumed_eval_only": True,
        }
        (out_dir / "bc_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8",
        )
        LOGGER.info("BC eval-only resume done | best=%s", best)
        return summary

    bcfg["best_checkpoint_name"] = "bc_best_policy.pt"
    bcfg["last_checkpoint_name"] = "bc_last_policy.pt"
    bcfg["metrics_csv_name"] = "bc_metrics.csv"
    bcfg["eval_json_name"] = "bc_eval_metrics.json"
    summary = train_behavior_cloning(config)
    if not best.is_file():
        raise PipelineError(f"BC failed to write actor checkpoint: {best}")
    LOGGER.info("BC done | best=%s last=%s", best, last)
    return summary


def stage_rl_finetune(config: dict, run_dir: Path) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 3 — RL fine-tuning from BC")
    LOGGER.info("=" * 60)
    bc_ckpt = Path(config["initialization"]["actor_checkpoint"])
    if not bc_ckpt.is_file():
        raise PipelineError(f"BC actor checkpoint missing for RL fine-tune: {bc_ckpt}")

    from src.training.train_rl import load_policy, train

    try:
        _ = load_policy(bc_ckpt, device="cpu")
    except Exception as exc:
        raise PipelineError(f"BC checkpoint incompatible with RL actor: {exc}") from exc

    summary = train(config)

    ckpt_dir = Path(config["outputs"]["checkpoints"])
    last_src = ckpt_dir / "last.pt"
    best_src = ckpt_dir / "best.pt"
    best_feas = ckpt_dir / "best_cycle_feasibility.pt"
    best_obj = ckpt_dir / "best_cycle_objective.pt"
    rl_dir = run_dir / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)
    if last_src.is_file():
        shutil.copy2(last_src, rl_dir / "rl_last.pt")
        shutil.copy2(last_src, run_dir / "checkpoints" / "rl_last.pt")
    if best_src.is_file():
        shutil.copy2(best_src, rl_dir / "rl_best_served.pt")
        shutil.copy2(best_src, rl_dir / "rl_best.pt")
        shutil.copy2(best_src, run_dir / "checkpoints" / "rl_best_served.pt")
    elif last_src.is_file():
        shutil.copy2(last_src, rl_dir / "rl_best_served.pt")
        shutil.copy2(last_src, rl_dir / "rl_best.pt")
    if best_feas.is_file():
        shutil.copy2(best_feas, rl_dir / "rl_best_feasibility.pt")
        shutil.copy2(best_feas, run_dir / "checkpoints" / "rl_best_feasibility.pt")
    if best_obj.is_file():
        shutil.copy2(best_obj, rl_dir / "rl_best_objective.pt")
        shutil.copy2(best_obj, run_dir / "checkpoints" / "rl_best_objective.pt")

    metrics_src = Path(config["outputs"]["results"]) / "training_metrics.csv"
    if metrics_src.is_file():
        shutil.copy2(metrics_src, rl_dir / "rl_metrics.csv")
        shutil.copy2(metrics_src, run_dir / "metrics" / "rl_metrics.csv")
    log_src = Path(config["outputs"]["logs"]) / "train_log.jsonl"
    if log_src.is_file():
        shutil.copy2(log_src, rl_dir / "train_log.jsonl")

    (rl_dir / "rl_summary.json").write_text(
        json.dumps(summary if isinstance(summary, dict) else {"ok": True}, indent=2, default=str),
        encoding="utf-8",
    )
    LOGGER.info("RL fine-tune done -> %s", rl_dir)
    return summary if isinstance(summary, dict) else {}


def stage_final_evaluation(config: dict, run_dir: Path) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage — multi-checkpoint final evaluation (test)")
    LOGGER.info("=" * 60)
    from src.training.final_report import (
        evaluate_checkpoint_on_split,
        select_final_checkpoint,
    )

    # Validation-first selection uses validation_instances when present.
    ds = config.setdefault("dataset", {})
    val_names = list(ds.get("validation_instances") or [])
    test_names = list(ds.get("test_instances") or ds.get("eval_instances") or [])
    if not test_names:
        raise PipelineError("No test instances for final evaluation")

    candidates: dict[str, Path] = {}
    bc_best = run_dir / "bc" / "bc_best_policy.pt"
    bc_last = run_dir / "bc" / "bc_last_policy.pt"
    if bc_best.is_file():
        candidates["bc_best"] = bc_best
    if bc_last.is_file():
        candidates["bc_last"] = bc_last
    for label, fname in (
        ("rl_best_served", "rl_best_served.pt"),
        ("rl_best_feasibility", "rl_best_feasibility.pt"),
        ("rl_best_objective", "rl_best_objective.pt"),
        ("rl_last", "rl_last.pt"),
    ):
        p = run_dir / "rl" / fname
        if p.is_file():
            candidates[label] = p
    if not candidates:
        raise PipelineError("No BC/RL checkpoints found for final evaluation")

    # Select on validation (prefer), then evaluate selected + all on test for comparison.
    selection_names = val_names if val_names else test_names
    sel_cfg = json.loads(json.dumps(config, default=str))
    sel_cfg["dataset"]["eval_instances"] = list(selection_names)
    sel_cfg["dataset"]["test_instances"] = list(selection_names)

    val_results: dict[str, dict] = {}
    for label, ckpt in candidates.items():
        LOGGER.info("Validation eval | %s | %s", label, ckpt.name)
        val_results[label] = evaluate_checkpoint_on_split(
            sel_cfg, ckpt,
            out_dir=run_dir / "evaluation" / "validation" / label,
            label=label,
        )

    selected, _ = select_final_checkpoint(val_results)
    LOGGER.info("Selected checkpoint on validation protocol: %s", selected)

    test_cfg = json.loads(json.dumps(config, default=str))
    test_cfg["dataset"]["eval_instances"] = list(test_names)
    test_cfg["dataset"]["test_instances"] = list(test_names)

    test_results: dict[str, dict] = {}
    for label, ckpt in candidates.items():
        LOGGER.info("Test eval | %s | %s", label, ckpt.name)
        row = evaluate_checkpoint_on_split(
            test_cfg, ckpt,
            out_dir=run_dir / "evaluation" / "test" / label,
            label=label,
        )
        test_results[label] = row
        by = row.get("by_class") or {}
        hard = by.get("hard_violations") or {}
        if any((hard.get(c) or 0) > 0 for c in hard):
            raise PipelineError(
                f"Hard EVRPTW violations on test for {label}: {hard}"
            )

    methods = {
        k: {
            "macro_served": v.get("macro_served"),
            "macro_feasibility": v.get("macro_feasibility"),
            "macro_objective": v.get("macro_objective"),
            "by_class": v.get("by_class"),
            "checkpoint": v.get("checkpoint"),
        }
        for k, v in test_results.items()
    }
    comparison = {
        "selection_split": "validation" if val_names else "test_fallback",
        "selected_on_validation": selected,
        "validation": {k: {
            "macro_served": v.get("macro_served"),
            "macro_feasibility": v.get("macro_feasibility"),
            "macro_objective": v.get("macro_objective"),
        } for k, v in val_results.items()},
        "methods": methods,
        "selected_test": methods.get(selected),
    }
    out = run_dir / "evaluation" / "checkpoint_comparison.json"
    out.write_text(json.dumps(comparison, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Multi-checkpoint comparison -> %s", out)
    return {
        "selected": selected,
        "comparison": comparison,
        "test_results": test_results,
        "holdout_report": (test_results.get(selected) or {}).get("summary"),
    }


def _write_comparison_html(report: dict, path: Path) -> None:
    """Compact offline HTML table for the pipeline stage comparison."""
    methods = report.get("methods") or {}
    cols = list(methods.keys())
    metric_rows = [
        ("served_ratio", "Served ratio"),
        ("feasibility", "Feasibility"),
        ("objective", "Original objective"),
        ("unserved", "Unserved customers"),
        ("station_action_fraction", "Station-action fraction"),
        ("depot_finishes", "Depot finishes"),
        ("battery_violations", "Battery violations"),
        ("tw_violations", "Customer TW violations"),
        ("depot_lateness", "Depot lateness"),
        ("capacity_violations", "Capacity violations"),
        ("max_step_terminations", "Max-step terminations"),
        ("objective_reconstruction_mismatch", "Obj. recon. mismatch"),
    ]

    def _fmt(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, bool):
            return "yes" if v else "no"
        if isinstance(v, (int, float)):
            return f"{float(v):.4g}"
        return str(v)

    rows_html = []
    for key, label in metric_rows:
        cells = "".join(f"<td>{_fmt((methods.get(c) or {}).get(key))}</td>" for c in cols)
        rows_html.append(f"<tr><th>{label}</th>{cells}</tr>")
    headers = "".join(f"<th>{c}</th>" for c in cols)
    html = (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'/>"
        "<title>Warm-start comparison</title>"
        "<style>body{font-family:Segoe UI,sans-serif;margin:2rem}"
        "table{border-collapse:collapse}th,td{border:1px solid #dde1e6;"
        "padding:.55rem .85rem;text-align:right}th:first-child,td:first-child"
        "{text-align:left}thead th{background:#1e293b;color:#fff}</style>"
        "</head><body><h1>Warm-start pipeline comparison</h1>"
        "<table><thead><tr><th>Metric</th>"
        f"{headers}</tr></thead><tbody>{''.join(rows_html)}</tbody></table>"
        "</body></html>"
    )
    path.write_text(html, encoding="utf-8")


def write_dashboard_bundle(run_dir: Path, stages: dict[str, Any]) -> None:
    dash = run_dir / "dashboard"
    dash.mkdir(parents=True, exist_ok=True)
    expert = stages.get("expert") or {}
    bc = stages.get("bc") or {}
    bc_eval = (bc.get("greedy_eval") or bc) if isinstance(bc, dict) else {}
    rl = stages.get("rl") or {}
    final = stages.get("evaluation") or {}
    holdout = (final.get("holdout_report") or final.get("summary") or final) if isinstance(final, dict) else {}

    comparison = {
        "run_dir": str(run_dir),
        "methods": {
            "expert": {
                "served_ratio": expert.get("mean_served_ratio"),
                "feasibility": expert.get("feasibility_rate"),
                "objective": expert.get("mean_objective"),
                "unserved": expert.get("mean_unserved_customers"),
                "station_action_fraction": expert.get("mean_station_action_fraction"),
                "depot_finishes": expert.get("mean_depot_finish_count"),
                "battery_violations": expert.get("mean_battery_violations"),
                "tw_violations": expert.get("mean_time_window_violations"),
                "depot_lateness": expert.get("mean_depot_lateness"),
                "capacity_violations": expert.get("mean_capacity_violations"),
                "max_step_terminations": expert.get("mean_max_step_terminations"),
                "objective_reconstruction_mismatch": expert.get("mean_relative_objective_mismatch"),
            },
            "bc": {
                "served_ratio": bc_eval.get("mean_served_ratio"),
                "feasibility": bc_eval.get("feasibility_rate"),
                "objective": bc_eval.get("mean_objective"),
                "unserved": bc_eval.get("mean_unserved"),
                "station_action_fraction": bc_eval.get("mean_station_action_fraction"),
                "depot_finishes": bc_eval.get("mean_depot_finish_count"),
                "battery_violations": bc_eval.get("mean_battery_violations"),
                "tw_violations": bc_eval.get("mean_tw_violations"),
                "depot_lateness": bc_eval.get("mean_depot_lateness"),
                "max_step_terminations": bc_eval.get("mean_max_steps"),
            },
            "rl_finetune": {
                "served_ratio": rl.get("final_served_rate") or holdout.get("mean_served_ratio"),
                "feasibility": rl.get("final_feasibility_rate") or holdout.get("feasibility_rate"),
                "objective": rl.get("final_cost") or rl.get("best_cost") or holdout.get("mean_objective"),
            },
            "final_holdout": {
                "served_ratio": holdout.get("mean_served_ratio") or holdout.get("served_rate"),
                "feasibility": holdout.get("feasibility_rate") or holdout.get("feasibility"),
                "objective": holdout.get("mean_objective") or holdout.get("objective"),
                "unserved": holdout.get("mean_unserved_customers") or holdout.get("unserved"),
                "battery_violations": holdout.get("mean_battery_violations"),
                "tw_violations": holdout.get("mean_time_window_violations"),
                "capacity_violations": holdout.get("mean_capacity_violations"),
                "depot_lateness": holdout.get("mean_depot_lateness"),
                "max_step_terminations": holdout.get("mean_max_step_terminations"),
            },
        },
    }
    (dash / "warmstart_comparison.json").write_text(
        json.dumps(comparison, indent=2, default=str), encoding="utf-8",
    )
    (run_dir / "pipeline_summary.json").write_text(
        json.dumps(
            {"stages_completed": list(stages.keys()), "comparison": comparison},
            indent=2, default=str,
        ),
        encoding="utf-8",
    )
    _write_comparison_html({"methods": comparison["methods"]}, dash / "warmstart_comparison.html")
    LOGGER.info("Dashboard bundle -> %s", dash)


def run_training_pipeline(
    config: dict,
    *,
    config_path: str | None = None,
    resume_run: str | Path | None = None,
) -> Path:
    flags = _pipeline_flags(config)
    if resume_run:
        run_dir = Path(resume_run)
        if not run_dir.is_dir():
            raise PipelineError(f"resume-run directory not found: {run_dir}")
        LOGGER.info("Resuming pipeline in existing run directory: %s", run_dir)
        for name in STAGE_DIRS:
            (run_dir / name).mkdir(parents=True, exist_ok=True)
        reuse = bool(flags.get("reuse_cached_stages", True))
    else:
        run_dir = create_run_directory(config)
        # Fresh run directory - nothing to reuse yet.
        reuse = False

    cfg = _wire_run_outputs(config, run_dir)

    stages: dict[str, Any] = {}
    runtime: dict[str, float] = {}
    t0 = time.time()

    from src.data.hanif_dataset import is_hanif_config

    dataset_report: dict = {}
    split_meta: dict = {}
    if is_hanif_config(cfg):
        t_ds = time.time()
        dataset_report, split_meta = stage_hanif_dataset(cfg, run_dir)
        runtime["dataset_stage_s"] = time.time() - t_ds
        stages["dataset"] = {"report": dataset_report, "split": split_meta}
        # Persist wired config after split mutation.
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8",
        )

    if flags.get("run_internal_checks", True):
        t_c = time.time()
        run_internal_checks(cfg)
        runtime["checks_s"] = time.time() - t_c
        stages["checks"] = {"ok": True}

    want_expert = bool(flags.get("run_expert", True))
    want_demos = bool(flags.get("generate_demonstrations", True))

    # One expert rollout pass when both stages are enabled (demos + metrics).
    if want_expert and want_demos:
        LOGGER.info(
            "Pipeline: Stage 1 demos (+ expert metrics) → Stage 2 BC → Stage 3 RL"
        )
        t_d = time.time()
        demo_path = stage_demonstrations(cfg, run_dir, reuse=reuse)
        runtime["expert_generation_time"] = time.time() - t_d
        stages["demonstrations"] = {"path": str(demo_path)}
        stages["expert"] = stage_expert_from_demonstrations(
            cfg, run_dir, demo_path, reuse=reuse,
        )
    elif want_expert:
        stages["expert"] = stage_expert(cfg, run_dir, reuse=reuse)
    elif want_demos:
        t_d = time.time()
        demo_path = stage_demonstrations(cfg, run_dir, reuse=reuse)
        runtime["expert_generation_time"] = time.time() - t_d
        stages["demonstrations"] = {"path": str(demo_path)}
        stages["expert"] = stage_expert_from_demonstrations(
            cfg, run_dir, demo_path, reuse=reuse,
        )

    if flags.get("run_behavior_cloning", True):
        t_bc = time.time()
        stages["bc"] = stage_behavior_cloning(cfg, run_dir, reuse=reuse)
        runtime["bc_stage_s"] = time.time() - t_bc

    if flags.get("run_rl_finetuning", True):
        t_rl = time.time()
        stages["rl"] = stage_rl_finetune(cfg, run_dir)
        runtime["rl_stage_s"] = time.time() - t_rl

    if flags.get("run_final_evaluation", True):
        t_ev = time.time()
        stages["evaluation"] = stage_final_evaluation(cfg, run_dir)
        runtime["final_eval_s"] = time.time() - t_ev

    t_dash = time.time()
    write_dashboard_bundle(run_dir, stages)
    runtime["dashboard_ms"] = (time.time() - t_dash) * 1000.0

    if is_hanif_config(cfg):
        from src.training.final_report import write_final_report

        eval_stage = stages.get("evaluation") or {}
        write_final_report(
            run_dir,
            dataset_report=dataset_report,
            split_meta=split_meta,
            stages=stages,
            checkpoint_comparison=eval_stage.get("comparison") or {},
            selected=str(eval_stage.get("selected") or "unknown"),
            runtime=runtime,
        )

    elapsed = time.time() - t0
    runtime["total_s"] = elapsed
    (run_dir / "metrics" / "runtime_profile.json").write_text(
        json.dumps(runtime, indent=2), encoding="utf-8",
    )
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline complete in %.1f min | run=%s", elapsed / 60.0, run_dir)
    LOGGER.info("Open dashboard: python main_dashboard.py")
    LOGGER.info("=" * 60)
    return run_dir
