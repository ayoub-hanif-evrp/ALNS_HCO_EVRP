"""Compare Pure RL / Expert / BC / BC+RL on RC holdouts (exact EVRPTW objective)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.utils.config import load_config
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

METRIC_KEYS = [
    ("served_ratio", "Served ratio"),
    ("feasibility", "Feasibility"),
    ("objective", "Original objective"),
    ("unserved", "Unserved customers"),
    ("station_action_fraction", "Station-action fraction"),
    ("depot_finishes", "Depot finishes"),
    ("battery_violations", "Battery violations"),
    ("tw_violations", "Customer TW violations"),
    ("depot_lateness", "Depot lateness"),
    ("max_step_terminations", "Max-step terminations"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Write warm-start comparison report.")
    p.add_argument("--config", default="configs/finetune_RC21_from_BC.yaml")
    p.add_argument("--out", default="outputs/results/warmstart_comparison.json")
    p.add_argument("--pure-rl-summary", default="outputs/runs/run_20260717_161318/final_summary.json")
    p.add_argument("--expert-report", default="outputs/demonstrations/expert_eval_report.json")
    p.add_argument("--bc-summary", default="outputs/bc/bc_summary.json")
    p.add_argument("--finetune-summary", default="outputs/results/training_summary.json")
    return p.parse_args()


def _load_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, (int, float)):
        return f"{float(v):.4g}"
    return str(v)


def _write_html(report: dict, path: Path) -> None:
    methods = report.get("methods") or {}
    cols = list(methods.keys())
    rows_html = []
    for key, label in METRIC_KEYS:
        cells = "".join(
            f"<td>{_fmt((methods.get(c) or {}).get(key))}</td>" for c in cols
        )
        rows_html.append(f"<tr><th>{label}</th>{cells}</tr>")
    headers = "".join(f"<th>{c}</th>" for c in cols)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Warm-start comparison — Transformer MARL EVRPTW</title>
<style>
  body {{ font-family: "Segoe UI", system-ui, sans-serif; margin: 2rem; background: #f7f8fa; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; margin-bottom: 0.25rem; }}
  p.sub {{ color: #555; margin-top: 0; }}
  table {{ border-collapse: collapse; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.08); }}
  th, td {{ border: 1px solid #dde1e6; padding: 0.55rem 0.85rem; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: #1e293b; color: #fff; }}
  tbody th {{ background: #f1f5f9; font-weight: 600; }}
  tr:nth-child(even) td {{ background: #fafbfc; }}
</style>
</head>
<body>
<h1>Warm-start pipeline comparison</h1>
<p class="sub">Pure RL → Expert → BC → BC+RL fine-tune · exact Schneider EVRPTW objective</p>
<table>
<thead><tr><th>Metric</th>{headers}</tr></thead>
<tbody>
{''.join(rows_html)}
</tbody>
</table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def main() -> None:
    args = parse_args()
    _ = load_config(args.config)
    pure = _load_json(args.pure_rl_summary) or {}
    expert = _load_json(args.expert_report) or {}
    bc = _load_json(args.bc_summary) or {}
    ft = _load_json(args.finetune_summary) or {}
    bc_eval = (bc.get("greedy_eval") or {})
    holdout = (ft.get("holdout_evaluation") or {}).get("holdout_report") or {}

    # Fallbacks for pure-RL metrics from run final_summary shapes.
    pure_served = (
        pure.get("final_served_rate")
        or pure.get("served_rate")
        or pure.get("macro_served_rate")
    )
    pure_feas = (
        pure.get("final_feasibility_rate")
        or pure.get("feasibility_rate")
        or pure.get("macro_feasibility")
    )
    pure_obj = pure.get("final_cost") or pure.get("best_cost") or pure.get("cost")

    report = {
        "methods": {
            "pure_rl": {
                "served_ratio": pure_served,
                "feasibility": pure_feas,
                "objective": pure_obj,
                "unserved": pure.get("unserved_customers") or pure.get("mean_unserved_customers"),
                "station_action_fraction": pure.get("station_action_fraction"),
                "depot_finishes": pure.get("depot_finish_count"),
                "battery_violations": pure.get("battery_violations"),
                "tw_violations": pure.get("time_window_violations"),
                "depot_lateness": pure.get("depot_lateness"),
                "max_step_terminations": pure.get("max_step_terminations"),
            },
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
                "max_step_terminations": expert.get("mean_max_step_terminations"),
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
            "bc_rl_finetune": {
                "served_ratio": ft.get("final_served_rate") or holdout.get("mean_served_ratio"),
                "feasibility": ft.get("final_feasibility_rate") or holdout.get("feasibility_rate"),
                "objective": ft.get("final_cost") or ft.get("best_cost") or holdout.get("mean_objective"),
                "unserved": holdout.get("mean_unserved_customers"),
                "station_action_fraction": holdout.get("mean_station_action_fraction"),
                "depot_finishes": holdout.get("mean_depot_finish_count"),
                "battery_violations": holdout.get("mean_battery_violations"),
                "tw_violations": holdout.get("mean_time_window_violations"),
                "depot_lateness": holdout.get("mean_depot_lateness"),
                "max_step_terminations": holdout.get("mean_max_step_terminations"),
                "holdout": holdout or None,
            },
        }
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    html_path = out.with_suffix(".html")
    _write_html(report, html_path)
    LOGGER.info("Wrote warm-start comparison: %s", out)
    LOGGER.info("Wrote warm-start HTML: %s", html_path)
    for name, block in report["methods"].items():
        LOGGER.info(
            "%s | served=%s feas=%s obj=%s",
            name, block.get("served_ratio"), block.get("feasibility"), block.get("objective"),
        )


if __name__ == "__main__":
    main()
