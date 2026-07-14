"""Evaluation pipeline: fleet search per instance, baselines, results, routes, plots."""
from __future__ import annotations

import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data.evrptw_dataset import EVRPTWDataset, split_dataset
from src.data.instance_class import CLASSES, class_filter_for, instance_class
from src.data.instance_loader import load_capacity_from_config
from src.data.training_phases import normalize_training_phase, resolve_training_phase_split
from src.env.cost import CostWeights
from src.evaluation.baselines import (
    NearestFeasiblePolicy,
    RandomFeasiblePolicy,
    run_heuristic_episode,
)
from src.evaluation.fleet_search import FleetCandidate, fleet_search
from src.evaluation.local_search import local_search_repair
from src.evaluation.metrics import build_result_row, build_routes
from src.env.evrptw_env import EVRPTWEnv
from src.training.rollout import run_episode, run_multi_sample
from src.training.train_rl import load_policy
from src.utils.config import load_config
from src.utils.logging_utils import get_logger, write_csv
from src.utils.plotting import (
    plot_feasibility_by_method,
    plot_fleet_search_curve,
    plot_metric_by_instance,
    plot_routes,
)
from src.utils.seed import describe_device, resolve_device, set_seed

LOGGER = get_logger()


def _eval_class_filter(cfg: dict, class_override: str | None = None) -> set[str]:
    if class_override:
        return set(class_filter_for([class_override]))
    ds = cfg.get("dataset") or {}
    if ds.get("class_filter"):
        return set(class_filter_for(ds["class_filter"]))
    return set(CLASSES)


def _load_eval_dataset(config: dict) -> EVRPTWDataset:
    ds_cfg = dict(config["dataset"])
    seed = int(config.get("seed", 42))
    load_cap = load_capacity_from_config(config)

    train_names, eval_names, meta = resolve_training_phase_split(config)
    phase = normalize_training_phase(config)
    if phase == "all_classes":
        ds_cfg["class_filter"] = ["C", "R", "RC"]
        ds_cfg["auto_discover_rc21"] = False

    _train_ds, eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=seed,
        train_filter=ds_cfg.get("train_size_filter") or ["_21"],
        eval_filter=ds_cfg.get("eval_size_filter") or ["_21"],
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_cap,
    )
    LOGGER.info(
        "Eval split phase=%s | holdout=%d | train excluded=%d",
        phase, len(eval_ds), len(train_names),
    )
    _ = meta
    return eval_ds


def _neural_runner(policy, weights, mode, device, n_samples, charging_portions, use_horizon, use_depot_charging, max_steps_factor):
    def runner(instance, m):
        if mode == "multi_sample":
            res = run_multi_sample(
                instance, policy, m, weights, n_samples=n_samples, device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                use_depot_charging=use_depot_charging,
                max_steps_factor=max_steps_factor,
            )
        else:
            res = run_episode(
                instance, policy, m, weights, mode="greedy", device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                use_depot_charging=use_depot_charging,
                max_steps_factor=max_steps_factor,
            )
        return FleetCandidate(m=m, feasible=res.feasible, objective=res.objective, summary=res.summary, result=res)

    return runner


def _baseline_runner(policy_factory, weights, use_horizon, use_depot_charging, max_steps_factor):
    def runner(instance, m):
        env = EVRPTWEnv(
            instance, m=m, use_horizon_check=use_horizon,
            use_depot_charging=use_depot_charging, max_steps_factor=max_steps_factor,
        )
        run_heuristic_episode(env, policy_factory())
        j, _, summary = env.score(weights)
        res = SimpleNamespace(env=env)
        return FleetCandidate(m=m, feasible=env.feasible_solution(), objective=j, summary=summary, result=res)

    return runner


def _write_class_macro_report(rows: list[dict], path: Path, method: str = "model_greedy") -> dict:
    """Per-class means and class-balanced macro-average for the primary method."""
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("method") != method:
            continue
        name = str(row.get("instance_name") or row.get("instance") or "")
        by_class[instance_class(name)].append(row)

    def _mean(vals: list[float]) -> float | None:
        return sum(vals) / len(vals) if vals else None

    report: dict = {"method": method, "per_class": {}, "macro_average": {}}
    keys = ("objective_value", "feasible", "served_ratio", "total_distance", "used_EVs")
    class_means: dict[str, dict] = {}
    for cls in ("C", "R", "RC"):
        crecs = by_class.get(cls) or []
        means = {}
        for k in keys:
            raw = []
            for r in crecs:
                if k == "feasible":
                    raw.append(float(r.get("feasible", r.get("raw_feasible", 0.0))))
                elif k == "served_ratio":
                    raw.append(float(r.get("served_ratio", r.get("served_rate", 0.0))))
                else:
                    raw.append(float(r.get(k, 0.0)))
            means[k] = _mean(raw)
        means["n_instances"] = len(crecs)
        class_means[cls] = means
        report["per_class"][cls] = means
        LOGGER.info(
            "Eval class %s | n=%d | objective=%s | served=%s | feas=%s",
            cls,
            means["n_instances"],
            None if means["objective_value"] is None else f"{means['objective_value']:.1f}",
            None if means["served_ratio"] is None else f"{means['served_ratio']:.3f}",
            None if means["feasible"] is None else f"{means['feasible']:.3f}",
        )

    present = [c for c in ("C", "R", "RC") if class_means[c]["n_instances"] > 0]
    macro = {}
    for k in keys:
        vals = [class_means[c][k] for c in present if class_means[c][k] is not None]
        macro[k] = _mean(vals)
    macro["n_classes"] = len(present)
    report["macro_average"] = macro
    LOGGER.info(
        "Macro-average over %d classes | objective=%s | served=%s | feas=%s",
        macro["n_classes"],
        None if macro["objective_value"] is None else f"{macro['objective_value']:.1f}",
        None if macro["served_ratio"] is None else f"{macro['served_ratio']:.3f}",
        None if macro["feasible"] is None else f"{macro['feasible']:.3f}",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    LOGGER.info("Wrote class macro report: %s", path)
    return report


def evaluate(
    config: dict,
    checkpoint: str | None = None,
    *,
    class_filter: str | None = None,
    macro_average: bool = False,
) -> dict:
    deterministic = bool(config.get("deterministic", False))
    set_seed(int(config.get("seed", 42)), deterministic=deterministic)
    device = resolve_device(config.get("device", "auto"))
    LOGGER.info(f"Evaluation on device: {describe_device(device)}")

    # Ensure evaluation defaults exist even if YAML omitted the block.
    ecfg = dict(config.get("evaluation") or {})
    ecfg.setdefault("multi_sample_n", 16)
    ecfg.setdefault("save_routes", True)
    ecfg.setdefault("save_figures", True)
    ecfg.setdefault("local_search", {"enabled": False})
    config["evaluation"] = ecfg

    eval_ds = _load_eval_dataset(config)
    LOGGER.info(f"Evaluation instances: {len(eval_ds)}")

    eval_class_filter = _eval_class_filter(config, class_filter)
    eval_indices = [
        i for i in range(len(eval_ds))
        if instance_class(eval_ds.names[i]) in eval_class_filter
    ]
    if eval_class_filter and len(eval_indices) < len(eval_ds):
        LOGGER.info(f"Eval class filter {sorted(eval_class_filter)}: {len(eval_indices)} instances")
    if not eval_indices:
        eval_indices = list(range(len(eval_ds)))

    weights = CostWeights.from_config(config["cost"])
    charging_portions = config["problem"]["charging_portions"]
    use_horizon = bool(config["problem"].get("use_horizon_check", True))
    use_depot_charging = bool(config["problem"].get("use_depot_charging", True))
    fcfg = config["fleet_search"]
    extra_margin = int(fcfg.get("extra_margin", 5))
    selection_mode = fcfg.get("selection_mode", "hierarchical")
    max_steps_factor = int(config["decoding"].get("max_steps_factor", 6))
    n_samples = int(ecfg.get("multi_sample_n", 16))

    ls_cfg = ecfg.get("local_search") or {}

    # Resolve checkpoint.
    ckpt_path = checkpoint or str(Path(config["outputs"]["checkpoints"]) / "best.pt")
    methods: dict[str, callable] = {}
    if Path(ckpt_path).exists():
        policy = load_policy(ckpt_path, device=device)
        LOGGER.info(f"Loaded checkpoint: {ckpt_path}")
        methods["model_greedy"] = _neural_runner(
            policy, weights, "greedy", device, n_samples, charging_portions,
            use_horizon, use_depot_charging, max_steps_factor,
        )
        methods["model_sample_bestN"] = _neural_runner(
            policy, weights, "multi_sample", device, n_samples, charging_portions,
            use_horizon, use_depot_charging, max_steps_factor,
        )
    else:
        LOGGER.warning(f"No checkpoint at {ckpt_path}; evaluating baselines only.")

    methods["nearest"] = _baseline_runner(
        lambda: NearestFeasiblePolicy(), weights, use_horizon, use_depot_charging, max_steps_factor,
    )
    methods["random"] = _baseline_runner(
        lambda: RandomFeasiblePolicy(seed=0), weights, use_horizon, use_depot_charging, max_steps_factor,
    )

    out = config["outputs"]
    results_path = Path(out["results"]) / "results.csv"
    routes_dir = Path(out["routes"])
    figures_dir = Path(out["figures"])
    routes_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_routes = bool(ecfg.get("save_routes", True))
    save_figures = bool(ecfg.get("save_figures", True))

    rows: list[dict] = []
    for i in eval_indices:
        instance = eval_ds[i]
        for method_name, runner in methods.items():
            t0 = time.time()
            with torch.no_grad():
                best, candidates = fleet_search(instance, runner, extra_margin, selection_mode)
            runtime = time.time() - t0
            env = best.result.env
            raw_objective = best.objective
            raw_feasible = best.feasible
            ls_improved = False
            ls_diag: dict = {}
            if hasattr(best.result, "env"):
                ls = local_search_repair(env, weights, ls_cfg)
                ls_diag = {
                    "local_search_diagnostic_only": float(ls.diagnostic_only),
                    "local_search_enabled": float(ls.enabled),
                    "local_search_distance_improvement": round(ls.distance_improvement, 4),
                    "local_search_rejection_reason": ls.rejection_reason,
                }
                ls_improved = False
            m_min = candidates[0].m
            m_max = candidates[-1].m
            row = build_result_row(instance, env, best.objective, best.m, m_min, m_max, runtime, method_name)
            row["raw_objective"] = raw_objective
            row["raw_feasible"] = float(raw_feasible)
            row["local_search_improved"] = float(ls_improved)
            row.update(ls_diag)
            rows.append(row)

            # Save routes + fleet-search curve for the primary model method.
            is_primary = method_name == "model_greedy" or (
                "model_greedy" not in methods and method_name == "nearest"
            )
            if save_routes and is_primary:
                route_data = build_routes(instance, env)
                (routes_dir / f"{instance.instance_name}_routes.json").write_text(
                    json.dumps(route_data, indent=2), encoding="utf-8"
                )
            if save_figures and is_primary:
                plot_fleet_search_curve(
                    candidates, best.m, figures_dir / f"fleet_search_{instance.instance_name}.png",
                    instance.instance_name,
                )
                if instance.num_customers <= 25:
                    plot_routes(instance, env, figures_dir / f"routes_{instance.instance_name}.png")

        LOGGER.info(f"[{i + 1}/{len(eval_ds)}] {instance.instance_name} done")

    write_csv(results_path, rows)
    LOGGER.info(f"Wrote results: {results_path}")

    # Archive a timestamped copy so previous evaluation results are preserved.
    archive_dir = Path(out["results"]) / "archive"
    archive_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    archived_csv = archive_dir / f"results_{stamp}.csv"
    shutil.copy2(results_path, archived_csv)
    LOGGER.info(f"Archived results copy: {archived_csv}")

    if save_figures and rows:
        plot_feasibility_by_method(rows, figures_dir / "feasibility_by_method.png")
        plot_metric_by_instance(rows, "total_distance", figures_dir / "distance_by_instance.png")
        plot_metric_by_instance(rows, "used_EVs", figures_dir / "used_evs_by_instance.png")
        plot_metric_by_instance(rows, "objective_value", figures_dir / "objective_by_method.png")

    out_summary: dict = {"results_path": str(results_path), "num_rows": len(rows)}
    if macro_average or normalize_training_phase(config) == "all_classes":
        report = _write_class_macro_report(
            rows, Path(out["results"]) / "class_macro_average.json",
        )
        out_summary["class_macro_report"] = report
    return out_summary


def evaluate_from_config_path(config_path: str, checkpoint: str | None = None) -> dict:
    return evaluate(load_config(config_path), checkpoint)
