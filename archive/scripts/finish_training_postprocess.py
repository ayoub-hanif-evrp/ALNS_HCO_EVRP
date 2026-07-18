"""Complete post-training artifacts after a run that finished training but crashed on cleanup.

Usage:
    python scripts/finish_training_postprocess.py --run-dir outputs/runs/run_20260705_194643
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.data.evrptw_dataset import split_dataset
from src.data.instance_loader import load_capacity_from_config
from src.data.rc21_split import resolve_rc21_split
from src.training.cycle_summary import dedupe_metric_rows
from src.training.train_rl import _build_final_summary, load_policy
from src.training.training_dashboard import (
    DashboardReport,
    generate_training_dashboard_from_config,
    regenerate_dashboard_index,
)
from src.utils.logging_utils import get_logger, write_csv
from src.utils.plotting import plot_stacked_composition, plot_training_curves, plot_training_overview
from src.utils.seed import describe_device, resolve_device, set_seed

LOGGER = get_logger()


def _load_train_dataset(config: dict):
    ds_cfg = config["dataset"]
    seed = int(config.get("seed", 42))
    load_cap = load_capacity_from_config(config)
    if ds_cfg.get("auto_discover_rc21"):
        train_names, eval_names, _meta = resolve_rc21_split(config)
        train_ds, _eval_ds = split_dataset(
            ds_cfg["path"],
            train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
            seed=seed,
            train_filter=ds_cfg.get("train_size_filter"),
            eval_filter=ds_cfg.get("eval_size_filter"),
            train_instances=train_names,
            eval_instances=eval_names,
            class_filter=ds_cfg.get("class_filter"),
            load_capacity=load_cap,
        )
        return train_ds
    train_ds, _eval_ds = split_dataset(
        ds_cfg["path"],
        train_ratio=float(ds_cfg.get("train_ratio", 0.8)),
        seed=seed,
        train_filter=ds_cfg.get("train_size_filter"),
        eval_filter=ds_cfg.get("eval_size_filter"),
        train_instances=ds_cfg.get("train_instances"),
        eval_instances=ds_cfg.get("eval_instances"),
        class_filter=ds_cfg.get("class_filter"),
        load_capacity=load_cap,
    )
    return train_ds


def finish_postprocess(
    *,
    config: dict,
    run_dir: Path,
    checkpoint: Path | None = None,
    skip_traces: bool = False,
) -> dict:
    run_dir = Path(run_dir)
    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    metrics_csv = run_dir / "metrics.csv"
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_csv}")

    rows = dedupe_metric_rows(pd.read_csv(metrics_csv).to_dict(orient="records"))
    out_cfg = config["outputs"]
    results_dir = Path(out_cfg["results"])
    figures_dir = Path(out_cfg["figures"]) / "training"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    holdout_meta = {}
    holdout_path = run_dir / "holdout_instances.json"
    if holdout_path.exists():
        holdout_meta = json.loads(holdout_path.read_text(encoding="utf-8"))

    write_csv(results_dir / "training_metrics.csv", rows)
    write_csv(run_dir / "metrics.csv", rows)

    run_id = run_dir.name
    epochs = int(rows[-1].get("epoch", len(rows) - 1)) + 1 if rows else 0
    train_ds = _load_train_dataset(config)
    log_path = Path(out_cfg.get("logs", "outputs/logs")) / "train_log.jsonl"
    summary = _build_final_summary(rows, run_id, epochs, len(train_ds), log_path, run_dir)
    summary["holdout_instances"] = holdout_meta

    dash_report = DashboardReport()
    try:
        dash_report = generate_training_dashboard_from_config(
            config, rows=rows, holdout_meta=holdout_meta, skip_train_count_check=True,
        )
    except Exception as exc:
        LOGGER.error("Training dashboard generation failed: %s", exc)

    tcfg = config.get("training") or {}
    dcfg = config.get("dashboard") or {}
    if (
        not skip_traces
        and dcfg.get("save_representative_traces", True)
        and not bool(tcfg.get("debug_overfit_one_instance", False))
    ):
        try:
            from src.training.solution_diagnostics import run_solution_behavior_diagnostics

            ckpt = checkpoint or Path(out_cfg["checkpoints"]) / "last.pt"
            if not ckpt.exists():
                ckpt = Path(out_cfg["checkpoints"]) / "best.pt"
            device = resolve_device(config.get("device", "auto"))
            set_seed(int(config.get("seed", 42)))
            LOGGER.info("Loading checkpoint %s on %s", ckpt, describe_device(device))
            policy = load_policy(str(ckpt), device=device)
            sol_figs, table_html = run_solution_behavior_diagnostics(
                config,
                pd.DataFrame(rows),
                policy,
                train_ds,
                device,
                figures_dir=figures_dir,
                seed=int(config.get("seed", 42)),
            )
            dash_report.figures.extend(sol_figs)
            regenerate_dashboard_index(dash_report, figures_dir, extra_html=table_html)
            LOGGER.info(
                "Solution behavior diagnostics: %d/%d figures",
                sum(1 for f in sol_figs if f.generated),
                len(sol_figs),
            )
        except Exception as exc:
            LOGGER.warning("Solution behavior diagnostics failed: %s", exc)

    summary["training_dashboard"] = {
        "html": dash_report.html_path,
        "figures_index": dash_report.index_md_path,
        "figures_generated": sum(1 for f in dash_report.figures if f.generated),
        "figures_total": len(dash_report.figures),
    }

    (results_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "final_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_training_curves(rows, figures_dir)
    plot_training_overview(rows, figures_dir / "overview.png")
    plot_stacked_composition(rows, "mean_obj_", "Objective term composition", figures_dir / "objective_composition.png")
    plot_stacked_composition(rows, "mean_pen_", "Penalty composition", figures_dir / "penalty_composition.png")

    LOGGER.info("Post-processing complete for %s", run_dir)
    LOGGER.info("training_summary.json written to %s", results_dir / "training_summary.json")
    if summary.get("training_dashboard", {}).get("html"):
        LOGGER.info("Dashboard: %s", summary["training_dashboard"]["html"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Finish post-training dashboards for a completed run.")
    parser.add_argument("--run-dir", type=str, required=True, help="Path to outputs/runs/run_* directory")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint override")
    parser.add_argument("--skip-traces", action="store_true", help="Skip solution trace diagnostics")
    args = parser.parse_args()

    finish_postprocess(
        config={},
        run_dir=Path(args.run_dir),
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        skip_traces=args.skip_traces,
    )


if __name__ == "__main__":
    main()
