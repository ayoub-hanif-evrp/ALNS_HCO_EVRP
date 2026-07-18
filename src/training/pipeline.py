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
        "run_expert": True,
        "generate_demonstrations": True,
        "run_behavior_cloning": True,
        "run_rl_finetuning": True,
        "run_final_evaluation": True,
        "reuse_cached_stages": True,
    }
    out = {**defaults, **raw}
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
    cfg["demonstrations"]["output"] = str(run_dir / "demonstrations" / "demonstrations.pt")
    cfg["demonstrations"]["eval_report"] = str(run_dir / "expert" / "expert_metrics.json")
    cfg["demonstrations"]["input"] = str(run_dir / "demonstrations" / "demonstrations.pt")
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


def run_internal_checks(config: dict) -> None:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 0 - internal sanity checks")
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
    from src.env.evrptw_env import EVRPTWEnv
    from src.env.policy_action import apply_policy_action
    from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert

    ds_path = Path(config["dataset"]["path"])
    small = ds_path / "c101C5"
    if not small.exists():
        small = ds_path / train_names[0]
    inst = load_instance(small)
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


def stage_expert(config: dict, run_dir: Path, *, reuse: bool) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 1 - expert evaluation")
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
    LOGGER.info(
        "Expert done | served=%.3f feas=%.3f -> %s",
        report["mean_served_ratio"], report["feasibility_rate"], metrics_path,
    )
    return report


def stage_demonstrations(config: dict, run_dir: Path, *, reuse: bool) -> Path:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 2 - expert demonstration generation")
    LOGGER.info("=" * 60)
    out_path = Path(config["demonstrations"]["output"])
    summary_path = run_dir / "demonstrations" / "demonstration_summary.json"
    if reuse and out_path.is_file() and summary_path.is_file():
        LOGGER.info("Reusing cached demonstrations: %s", out_path)
        return out_path

    from src.expert.generate_demonstrations import generate_demonstrations

    path = generate_demonstrations(config)
    src_summary = path.with_suffix(".summary.json")
    if src_summary.is_file():
        shutil.copy2(src_summary, summary_path)
    else:
        summary_path.write_text(json.dumps({"path": str(path)}, indent=2), encoding="utf-8")

    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes") or []
    if not episodes:
        raise PipelineError("Demonstration file is empty - cannot continue to BC")
    n_actions = sum(len(ep.get("actions") or []) for ep in episodes)
    if n_actions == 0:
        raise PipelineError("Demonstrations contain zero actions - cannot continue to BC")
    LOGGER.info("Demonstrations OK | episodes=%d actions=%d -> %s", len(episodes), n_actions, path)
    return path


def stage_behavior_cloning(config: dict, run_dir: Path, *, reuse: bool) -> dict:
    LOGGER.info("=" * 60)
    LOGGER.info("Stage 3 - behavior cloning (fixed epochs, no early stopping)")
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

    from src.training.behavior_cloning import train_behavior_cloning

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
    LOGGER.info("Stage 4 - RL fine-tuning from BC")
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
    rl_dir = run_dir / "rl"
    rl_dir.mkdir(parents=True, exist_ok=True)
    if last_src.is_file():
        shutil.copy2(last_src, rl_dir / "rl_last.pt")
        shutil.copy2(last_src, run_dir / "checkpoints" / "rl_last.pt")
    if best_src.is_file():
        shutil.copy2(best_src, rl_dir / "rl_best.pt")
        shutil.copy2(best_src, run_dir / "checkpoints" / "rl_best.pt")
    elif last_src.is_file():
        shutil.copy2(last_src, rl_dir / "rl_best.pt")

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
    LOGGER.info("Stage 5 - final holdout evaluation")
    LOGGER.info("=" * 60)
    ckpt = run_dir / "rl" / "rl_last.pt"
    if not ckpt.is_file():
        ckpt = run_dir / "checkpoints" / "last.pt"
    if not ckpt.is_file():
        raise PipelineError(f"No RL checkpoint for final evaluation: {ckpt}")

    from src.evaluation.evaluate import evaluate

    cfg = json.loads(json.dumps(config, default=str))
    cfg["outputs"] = {
        **cfg["outputs"],
        "results": str(run_dir / "evaluation"),
        "routes": str(run_dir / "evaluation" / "routes"),
        "figures": str(run_dir / "evaluation" / "figures"),
    }
    Path(cfg["outputs"]["results"]).mkdir(parents=True, exist_ok=True)

    summary = evaluate(cfg, checkpoint=str(ckpt))
    eval_path = run_dir / "evaluation" / "final_eval_metrics.json"
    eval_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    results_csv = Path(cfg["outputs"]["results"]) / "results.csv"
    per_inst = run_dir / "evaluation" / "per_instance_eval.csv"
    if results_csv.is_file():
        shutil.copy2(results_csv, per_inst)

    holdout = summary.get("holdout_report") or summary.get("summary") or summary
    batt = float(holdout.get("mean_battery_violations", holdout.get("battery_violations", 0)) or 0)
    tw = float(holdout.get("mean_time_window_violations", holdout.get("time_window_violations", 0)) or 0)
    cap = float(holdout.get("mean_capacity_violations", holdout.get("capacity_violations", 0)) or 0)
    if batt > 0 or tw > 0 or cap > 0:
        LOGGER.error(
            "FINAL EVAL HARD VIOLATIONS | battery=%.4f tw=%.4f capacity=%.4f",
            batt, tw, cap,
        )
    else:
        LOGGER.info("Final eval: no hard EVRPTW violations reported")
    LOGGER.info("Final evaluation -> %s", eval_path)
    return summary


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


def run_training_pipeline(config: dict, *, config_path: str | None = None) -> Path:
    flags = _pipeline_flags(config)
    run_dir = create_run_directory(config)
    cfg = _wire_run_outputs(config, run_dir)
    # Fresh run directory - nothing to reuse yet.
    reuse = False

    stages: dict[str, Any] = {}
    t0 = time.time()

    if flags.get("run_internal_checks", True):
        run_internal_checks(cfg)
        stages["checks"] = {"ok": True}

    if flags.get("run_expert", True):
        stages["expert"] = stage_expert(cfg, run_dir, reuse=reuse)

    if flags.get("generate_demonstrations", True):
        demo_path = stage_demonstrations(cfg, run_dir, reuse=reuse)
        stages["demonstrations"] = {"path": str(demo_path)}

    if flags.get("run_behavior_cloning", True):
        stages["bc"] = stage_behavior_cloning(cfg, run_dir, reuse=reuse)

    if flags.get("run_rl_finetuning", True):
        stages["rl"] = stage_rl_finetune(cfg, run_dir)

    if flags.get("run_final_evaluation", True):
        stages["evaluation"] = stage_final_evaluation(cfg, run_dir)

    write_dashboard_bundle(run_dir, stages)
    elapsed = time.time() - t0
    LOGGER.info("=" * 60)
    LOGGER.info("Pipeline complete in %.1f min | run=%s", elapsed / 60.0, run_dir)
    LOGGER.info("Open dashboard: python main_dashboard.py")
    LOGGER.info("=" * 60)
    return run_dir
