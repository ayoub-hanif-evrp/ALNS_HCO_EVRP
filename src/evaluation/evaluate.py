"""Evaluation pipeline: fleet search per instance, baselines, results, routes, plots."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from src.data.evrptw_dataset import EVRPTWDataset, split_dataset
from src.data.instance_class import instance_class
from src.data.instance_loader import load_capacity_from_config
from src.training.curriculum import active_train_classes
from src.training.fixed_mini import (
    all_fixed_names,
    assert_disjoint_train_eval,
    build_fixed_mini_datasets,
    resolve_fixed_train_instances,
)
from src.env.cost import CostWeights
from src.evaluation.baselines import (
    NearestFeasiblePolicy,
    RandomFeasiblePolicy,
    run_heuristic_episode,
)
from src.evaluation.fleet_search import FleetCandidate, fleet_search
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


def _neural_runner(policy, weights, mode, device, n_samples, charging_portions, use_horizon, max_steps_factor):
    def runner(instance, m):
        if mode == "multi_sample":
            res = run_multi_sample(
                instance, policy, m, weights, n_samples=n_samples, device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                max_steps_factor=max_steps_factor,
            )
        else:
            res = run_episode(
                instance, policy, m, weights, mode="greedy", device=device,
                charging_portions=charging_portions, use_horizon_check=use_horizon,
                max_steps_factor=max_steps_factor,
            )
        return FleetCandidate(m=m, feasible=res.feasible, objective=res.objective, summary=res.summary, result=res)

    return runner


def _baseline_runner(policy_factory, weights, use_horizon, max_steps_factor):
    def runner(instance, m):
        env = EVRPTWEnv(instance, m=m, use_horizon_check=use_horizon, max_steps_factor=max_steps_factor)
        run_heuristic_episode(env, policy_factory())
        j, _, summary = env.score(weights)
        res = SimpleNamespace(env=env)
        return FleetCandidate(m=m, feasible=env.feasible_solution(), objective=j, summary=summary, result=res)

    return runner


def evaluate(config: dict, checkpoint: str | None = None) -> dict:
    deterministic = bool(config.get("deterministic", False))
    set_seed(int(config.get("seed", 42)), deterministic=deterministic)
    device = resolve_device(config.get("device", "auto"))
    LOGGER.info(f"Evaluation on device: {describe_device(device)}")

    ds_cfg = config["dataset"]
    seed = int(config.get("seed", 42))
    ccfg = config.get("curriculum") or {}
    if str(ccfg.get("mode", "")).lower() == "fixed_mini":
        full_ds = EVRPTWDataset(
            ds_cfg["path"],
            load_capacity=load_capacity_from_config(config),
            ev_specs_path=ds_cfg.get("ev_specs_path"),
        )
        fixed_by_class = resolve_fixed_train_instances(full_ds.names, config, seed)
        fixed_names = all_fixed_names(fixed_by_class)
        _train_ds, eval_ds = build_fixed_mini_datasets(full_ds, fixed_names)
        assert_disjoint_train_eval(fixed_names, eval_ds.names)
        LOGGER.info(f"Fixed-mini eval: {len(eval_ds)} instances (excludes {len(fixed_names)} train)")
    else:
        _train_ds, eval_ds = split_dataset(
            ds_cfg["path"],
            train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
            seed=seed,
            train_filter=ds_cfg.get("train_size_filter"),
            eval_filter=ds_cfg.get("eval_size_filter"),
            load_capacity=load_capacity_from_config(config),
            ev_specs_path=ds_cfg.get("ev_specs_path"),
        )
    LOGGER.info(f"Evaluation instances: {len(eval_ds)}")

    eval_class_filter = set(active_train_classes(config))
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
    fcfg = config["fleet_search"]
    extra_margin = int(fcfg.get("extra_margin", 5))
    selection_mode = fcfg.get("selection_mode", "hierarchical")
    ecfg = config["evaluation"]
    max_steps_factor = int(config["decoding"].get("max_steps_factor", 6))
    n_samples = int(ecfg.get("multi_sample_n", 16))

    # Resolve checkpoint.
    ckpt_path = checkpoint or str(Path(config["outputs"]["checkpoints"]) / "best.pt")
    methods: dict[str, callable] = {}
    if Path(ckpt_path).exists():
        policy = load_policy(ckpt_path, device=device)
        LOGGER.info(f"Loaded checkpoint: {ckpt_path}")
        methods["model_greedy"] = _neural_runner(
            policy, weights, "greedy", device, n_samples, charging_portions, use_horizon, max_steps_factor
        )
        methods["model_sample_bestN"] = _neural_runner(
            policy, weights, "multi_sample", device, n_samples, charging_portions, use_horizon, max_steps_factor
        )
    else:
        LOGGER.warning(f"No checkpoint at {ckpt_path}; evaluating baselines only.")

    methods["nearest"] = _baseline_runner(lambda: NearestFeasiblePolicy(), weights, use_horizon, max_steps_factor)
    methods["random"] = _baseline_runner(lambda: RandomFeasiblePolicy(seed=0), weights, use_horizon, max_steps_factor)

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
            m_min = candidates[0].m
            m_max = candidates[-1].m
            rows.append(
                build_result_row(instance, env, best.objective, best.m, m_min, m_max, runtime, method_name)
            )

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

    return {"results_path": str(results_path), "num_rows": len(rows)}


def evaluate_from_config_path(config_path: str, checkpoint: str | None = None) -> dict:
    return evaluate(load_config(config_path), checkpoint)
