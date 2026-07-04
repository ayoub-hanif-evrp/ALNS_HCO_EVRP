"""Generate focused MARL EVRP training diagnostic dashboard (training metrics only)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logging_utils import get_logger

LOGGER = get_logger()

REQUIRED_RC21_TRAIN_COUNT = 20

DASHBOARD_SECTIONS = [
    "1. Learning progress",
    "2. Customer service and feasibility",
    "3. Reward and penalty diagnosis",
    "4. Action behavior",
    "5. Critic and actor health",
    "6. Per-instance learning",
    "7. EVRP operational metrics",
    "8. Solution behavior diagnostics",
]


@dataclass
class FigureResult:
    name: str
    filename: str
    metrics: list[str]
    section: str
    description: str
    generated: bool
    required: bool = False
    png_path: str | None = None
    svg_path: str | None = None
    skip_reason: str | None = None


@dataclass
class DashboardReport:
    figures: list[FigureResult] = field(default_factory=list)
    html_path: str | None = None
    index_md_path: str | None = None
    num_epochs: int = 0
    num_train_instances: int = 0
    ma_window: int = REQUIRED_RC21_TRAIN_COUNT


def load_training_dataframe(
    *,
    csv_path: str | Path | None = None,
    jsonl_path: str | Path | None = None,
    train_instances: list[str] | None = None,
    holdout_instances: list[str] | None = None,
) -> pd.DataFrame:
    """Load training metrics and keep only training instances (exclude holdout)."""
    rows: list[dict[str, Any]] = []
    if csv_path and Path(csv_path).exists() and Path(csv_path).stat().st_size > 0:
        rows = pd.read_csv(csv_path).to_dict(orient="records")
    elif jsonl_path and Path(jsonl_path).exists():
        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    if "epoch" in df.columns:
        df = df.sort_values("epoch", na_position="last").drop_duplicates(subset=["epoch"], keep="last")
    df = df.reset_index(drop=True)

    holdout = set(holdout_instances or [])
    if train_instances:
        allowed = set(train_instances)
        if "instance_name" in df.columns:
            df = df[df["instance_name"].isin(allowed)]
    elif holdout and "instance_name" in df.columns:
        df = df[~df["instance_name"].isin(holdout)]

    return df.reset_index(drop=True)


def resolve_dashboard_ma_window(
    config: dict,
    df: pd.DataFrame,
    holdout_meta: dict | None = None,
) -> int:
    """MA window = training instances per cycle; fail if RC _21 count != 20."""
    ds = config.get("dataset") or {}
    meta = holdout_meta or {}
    train_names = list(meta.get("train_instances") or ds.get("train_instances") or [])
    if train_names:
        n_train = len(train_names)
    elif "instance_name" in df.columns and not df.empty:
        n_train = int(df["instance_name"].nunique())
    else:
        n_train = int(meta.get("num_train") or ds.get("expected_train_count") or 0)

    if n_train != REQUIRED_RC21_TRAIN_COUNT:
        raise ValueError(
            f"Dashboard requires {REQUIRED_RC21_TRAIN_COUNT} RC _21 training instances per cycle "
            f"(moving-average window = {REQUIRED_RC21_TRAIN_COUNT}), but found {n_train}. "
            f"Train instances: {train_names or '(from metrics)'}"
        )
    return n_train


def _moving_avg(values: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))
    return values.astype(float).rolling(window=w, min_periods=1).mean()


def _epoch_x(df: pd.DataFrame) -> np.ndarray:
    if "epoch" in df.columns:
        return df["epoch"].to_numpy(dtype=float)
    return np.arange(len(df), dtype=float)


def _cycle_col(df: pd.DataFrame, n_inst: int) -> pd.Series:
    if "pass" in df.columns and df["pass"].notna().any():
        return df["pass"].astype(int)
    if "cycle" in df.columns and df["cycle"].notna().any():
        return df["cycle"].astype(int)
    n = max(int(n_inst), 1)
    return pd.Series((np.arange(len(df)) // n) + 1, index=df.index)


def _pick(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _save_fig(fig: plt.Figure, out_dir: Path, filename: str, *, save_svg: bool) -> tuple[Path | None, Path | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / filename
    try:
        fig.savefig(png, dpi=140, bbox_inches="tight")
    except Exception as exc:
        plt.close(fig)
        LOGGER.warning("Failed to save %s: %s", png, exc)
        return None, None
    svg_path = None
    if save_svg:
        svg = png.with_suffix(".svg")
        try:
            fig.savefig(svg, bbox_inches="tight")
            svg_path = svg
        except Exception:
            pass
    plt.close(fig)
    return png, svg_path


def _plot_series_ma(
    df: pd.DataFrame,
    col: str,
    title: str,
    ylabel: str,
    out_dir: Path,
    filename: str,
    *,
    ma_window: int,
    save_svg: bool,
    hline: float | None = None,
) -> tuple[Path | None, Path | None]:
    x = _epoch_x(df)
    y = df[col].astype(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color="#94a3b8", lw=1, alpha=0.65, label="per epoch")
    ax.plot(x, _moving_avg(y, ma_window), color="#2563eb", lw=2, label=f"MA ({ma_window})")
    if hline is not None:
        ax.axhline(hline, color="#ef4444", ls="--", lw=1, alpha=0.8, label=f"ref {hline}")
    ax.set_title(title)
    ax.set_xlabel("training epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_multi_ma(
    df: pd.DataFrame,
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
    out_dir: Path,
    filename: str,
    *,
    ma_window: int,
    save_svg: bool,
    raw: bool = False,
) -> tuple[Path | None, Path | None]:
    x = _epoch_x(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#2563eb", "#dc2626", "#16a34a", "#d97706", "#7c3aed", "#0891b2"]
    for i, (col, label) in enumerate(series):
        y = df[col].astype(float)
        c = colors[i % len(colors)]
        if raw:
            ax.plot(x, y, lw=1, alpha=0.45, color=c)
        ax.plot(x, _moving_avg(y, ma_window), lw=2, color=c, label=label)
    ax.set_title(title)
    ax.set_xlabel("training epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_fig(fig, out_dir, filename, save_svg=save_svg)


def _cycle_aggregate(df: pd.DataFrame, n_inst: int) -> pd.DataFrame:
    work = df.copy()
    work["_cycle"] = _cycle_col(work, n_inst)
    cols = [c for c in [
        "final_objective", "cost", "feasibility_rate", "served_rate", "unserved_customers",
    ] if c in work.columns]
    if not cols:
        return pd.DataFrame()
    return work.groupby("_cycle", as_index=False)[cols].mean(numeric_only=True).rename(columns={"_cycle": "cycle"})


def _heatmap_instance_cycle(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_dir: Path,
    filename: str,
    *,
    n_inst: int,
    save_svg: bool,
) -> tuple[Path | None, Path | None]:
    if "instance_name" not in df.columns or value_col not in df.columns:
        return None, None
    work = df.copy()
    work["_cycle"] = _cycle_col(work, n_inst)
    pivot = work.pivot_table(index="instance_name", columns="_cycle", values=value_col, aggfunc="mean")
    if pivot.empty:
        return None, None
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("training cycle")
    ax.set_ylabel("RC _21 training instance")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return _save_fig(fig, out_dir, filename, save_svg=save_svg)


def _register(
    results: list[FigureResult],
    *,
    name: str,
    filename: str,
    metrics: list[str],
    section: str,
    description: str,
    required: bool,
    png: Path | None,
    svg: Path | None,
    skip_reason: str | None = None,
) -> None:
    results.append(
        FigureResult(
            name=name,
            filename=filename,
            metrics=metrics,
            section=section,
            description=description,
            required=required,
            generated=png is not None,
            png_path=str(png) if png else None,
            svg_path=str(svg) if svg else None,
            skip_reason=skip_reason,
        )
    )


def generate_training_dashboard(
    df: pd.DataFrame,
    out_dir: str | Path,
    *,
    moving_avg_window: int = REQUIRED_RC21_TRAIN_COUNT,
    n_train_instances: int = REQUIRED_RC21_TRAIN_COUNT,
    save_svg: bool = True,
) -> DashboardReport:
    """Build diagnostic training figures, figures_index.md, and index.html."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ma = int(moving_avg_window)
    n_inst = int(n_train_instances)
    report = DashboardReport(num_epochs=len(df), num_train_instances=n_inst, ma_window=ma)
    results: list[FigureResult] = []

    if df.empty:
        report.index_md_path = str(_write_index_md(results, out_dir, ma))
        report.html_path = str(_write_index_html(results, out_dir, ma))
        return report

    def _try(
        name: str,
        filename: str,
        metrics: list[str],
        section: str,
        description: str,
        required: bool,
        fn: Callable[[], tuple[Path | None, Path | None]],
    ) -> None:
        try:
            png, svg = fn()
            reason = None if png else "required metrics missing or empty plot"
            _register(
                results, name=name, filename=filename, metrics=metrics, section=section,
                description=description, required=required, png=png, svg=svg, skip_reason=reason,
            )
        except Exception as exc:
            LOGGER.warning("Figure %s skipped: %s", filename, exc)
            _register(
                results, name=name, filename=filename, metrics=metrics, section=section,
                description=description, required=required, png=None, svg=None, skip_reason=str(exc),
            )

    obj_col = _pick(df, "final_objective", "cost", "cost_raw")
    cyc = _cycle_aggregate(df, n_inst)

    # --- 1. Learning progress ---
    if obj_col:
        _try(
            "Final objective over epochs", "01_objective.png", [obj_col],
            DASHBOARD_SECTIONS[0],
            "Real EVRP objective J per epoch (lower is better). MA window = one full training cycle.",
            True,
            lambda: _plot_series_ma(
                df, obj_col, "Final objective over epochs", "objective J",
                out_dir, "01_objective.png", ma_window=ma, save_svg=save_svg,
            ),
        )
    else:
        _register(results, name="Final objective over epochs", filename="01_objective.png",
                  metrics=["final_objective", "cost"], section=DASHBOARD_SECTIONS[0],
                  description="Real EVRP objective.", required=True, png=None, svg=None,
                  skip_reason="final_objective / cost column missing")

    ret_series = [(c, l) for c, l in [
        ("raw_objective_return", "raw objective return"),
        ("shaped_training_return", "shaped training return"),
        ("reward", "episode return (legacy)"),
    ] if c in df.columns]
    if len(ret_series) >= 2 or (ret_series and ret_series[0][0] in ("raw_objective_return", "shaped_training_return")):
        cols = [c for c, _ in ret_series[:2]] if len(ret_series) >= 2 else [ret_series[0][0]]
        labels = [l for c, l in ret_series[:2]] if len(ret_series) >= 2 else [ret_series[0][1]]
        _try(
            "Raw objective return vs shaped training return", "02_raw_vs_shaped_return.png", cols,
            DASHBOARD_SECTIONS[0],
            "Compares the true objective signal (-J/scale) with the shaped return used for policy updates.",
            True,
            lambda: _plot_multi_ma(
                df, list(zip(cols, labels)), "Raw objective return vs shaped training return",
                "return", out_dir, "02_raw_vs_shaped_return.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    cycle_obj_col = _pick(cyc, "final_objective", "cost") if not cyc.empty else None
    if cycle_obj_col and not cyc.empty:
        def _cycle_obj():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(cyc["cycle"], cyc[cycle_obj_col], marker="o", color="#2563eb", lw=2)
            ax.set_title("Cycle-level mean objective")
            ax.set_xlabel("training cycle")
            ax.set_ylabel("mean objective J")
            fig.tight_layout()
            return _save_fig(fig, out_dir, "03_cycle_objective.png", save_svg=save_svg)
        _try(
            "Cycle-level objective", "03_cycle_objective.png", [cycle_obj_col],
            DASHBOARD_SECTIONS[0],
            f"Mean objective over all {n_inst} training instances per cycle (most interpretable learning curve).",
            True, _cycle_obj,
        )

    # --- 2. Customer service and feasibility ---
    served_col = _pick(df, "served_rate", "served_rate_pct")
    if served_col:
        _try(
            "Served-customer ratio", "04_served_ratio.png", [served_col],
            DASHBOARD_SECTIONS[1], "Fraction of customers served (target 1.0).", True,
            lambda: _plot_series_ma(
                df, served_col, "Served-customer ratio", "served ratio",
                out_dir, "04_served_ratio.png", ma_window=ma, save_svg=save_svg, hline=1.0,
            ),
        )

    if "unserved_customers" in df.columns:
        _try(
            "Unserved customers", "05_unserved_customers.png", ["unserved_customers"],
            DASHBOARD_SECTIONS[1], "Remaining unserved customers per episode (lower is better).", True,
            lambda: _plot_series_ma(
                df, "unserved_customers", "Unserved customers", "count",
                out_dir, "05_unserved_customers.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    if "feasibility_rate" in df.columns:
        _try(
            "Feasibility rate", "06_feasibility_rate.png", ["feasibility_rate"],
            DASHBOARD_SECTIONS[1], "Feasible solution indicator smoothed over epochs.", True,
            lambda: _plot_series_ma(
                df, "feasibility_rate", "Feasibility rate", "feasibility",
                out_dir, "06_feasibility_rate.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    # --- 3. Reward and penalty diagnosis ---
    decomp: list[tuple[str, str]] = []
    for col, label in [
        ("served_bonus_total", "served bonus"),
        ("early_depot_penalty_total", "early depot penalty"),
        ("terminal_unserved_penalty_scaled", "terminal unserved (scaled)"),
        ("mean_pen_unserved", "pen. unserved (objective)"),
        ("mean_pen_infeasible", "pen. infeasible"),
        ("step_penalty_total", "step penalty"),
    ]:
        if col in df.columns and df[col].notna().any() and float(df[col].abs().sum()) > 0:
            decomp.append((col, label))
    if decomp:
        _try(
            "Reward decomposition", "07_reward_decomposition.png", [c for c, _ in decomp],
            DASHBOARD_SECTIONS[2],
            "Training reward components — check whether terminal penalties still dominate.", True,
            lambda: _plot_multi_ma(
                df, decomp[:5], "Reward and penalty decomposition", "magnitude",
                out_dir, "07_reward_decomposition.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    if "terminal_penalty_coef" in df.columns:
        def _tpc():
            x = _epoch_x(df)
            y = df["terminal_penalty_coef"].astype(float)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, y, color="#d97706", lw=2)
            ax.set_title("Terminal unserved penalty schedule")
            ax.set_xlabel("training epoch")
            ax.set_ylabel("terminal penalty coefficient")
            fig.tight_layout()
            return _save_fig(fig, out_dir, "08_terminal_penalty_schedule.png", save_svg=save_svg)
        _try(
            "Terminal penalty schedule", "08_terminal_penalty_schedule.png", ["terminal_penalty_coef"],
            DASHBOARD_SECTIONS[2], "Ramp of terminal unserved penalty during training.", False, _tpc,
        )

    if "relative_objective_mismatch" in df.columns:
        _try(
            "Objective mismatch", "09_objective_mismatch.png", ["relative_objective_mismatch"],
            DASHBOARD_SECTIONS[2],
            "Relative |J - reconstructed J| / |J|. Values > 0.05 mean reward misalignment.", True,
            lambda: _plot_series_ma(
                df, "relative_objective_mismatch", "Relative objective mismatch", "relative mismatch",
                out_dir, "09_objective_mismatch.png", ma_window=ma, save_svg=save_svg, hline=0.05,
            ),
        )

    # --- 4. Action behavior ---
    action_cols = [(c, l) for c, l in [
        ("fraction_customer_actions", "customer"),
        ("fraction_station_actions", "station"),
        ("fraction_depot_actions", "depot"),
    ] if c in df.columns]
    if len(action_cols) >= 2:
        _try(
            "Action type fractions", "10_action_type_fractions.png", [c for c, _ in action_cols],
            DASHBOARD_SECTIONS[3],
            "Policy action mix — rising customer fraction and falling early depot use indicate progress.", True,
            lambda: _plot_multi_ma(
                df, action_cols, "Action type fractions", "fraction",
                out_dir, "10_action_type_fractions.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    if "early_depot_returns" in df.columns:
        _try(
            "Early depot returns", "11_early_depot_returns.png", ["early_depot_returns"],
            DASHBOARD_SECTIONS[3],
            "Depot returns while customers remain unserved (should decrease).", True,
            lambda: _plot_series_ma(
                df, "early_depot_returns", "Early depot returns per episode", "count",
                out_dir, "11_early_depot_returns.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    # --- 5. Critic and actor health ---
    loss_cols = [(c, l) for c, l in [("actor_loss", "actor"), ("critic_loss", "critic")] if c in df.columns]
    if loss_cols:
        _try(
            "Actor and critic loss", "12_actor_critic_loss.png", [c for c, _ in loss_cols],
            DASHBOARD_SECTIONS[4], "Actor-critic optimization stability.", True,
            lambda: _plot_multi_ma(
                df, loss_cols, "Actor and critic loss", "loss",
                out_dir, "12_actor_critic_loss.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    if "critic_explained_variance" in df.columns:
        def _ev():
            x = _epoch_x(df)
            y = df["critic_explained_variance"].astype(float)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, y, color="#7c3aed", lw=1.5)
            ax.axhline(0.0, color="#64748b", ls="--", lw=1)
            ax.set_title("Critic explained variance")
            ax.set_xlabel("training epoch")
            ax.set_ylabel("explained variance")
            fig.tight_layout()
            return _save_fig(fig, out_dir, "13_explained_variance.png", save_svg=save_svg)
        _try(
            "Critic explained variance", "13_explained_variance.png", ["critic_explained_variance"],
            DASHBOARD_SECTIONS[4], "Near zero means the critic is not learning useful values.", True, _ev,
        )

    adv_cols = [(c, l) for c, l in [("advantage", "mean"), ("advantage_std", "std")] if c in df.columns]
    if adv_cols:
        _try(
            "Advantage statistics", "14_advantage_stats.png", [c for c, _ in adv_cols],
            DASHBOARD_SECTIONS[4], "Scale and spread of GAE advantages.", False,
            lambda: _plot_multi_ma(
                df, adv_cols, "Advantage statistics", "advantage",
                out_dir, "14_advantage_stats.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    grad_cols = [(c, l) for c, l in [
        ("actor_grad_norm", "actor grad"),
        ("critic_grad_norm", "critic grad"),
        ("actor_param_delta_norm", "actor Δparam"),
        ("critic_param_delta_norm", "critic Δparam"),
        ("grad_norm", "total grad"),
    ] if c in df.columns and df[c].notna().any()]
    if len(grad_cols) >= 2:
        _try(
            "Gradient and parameter update norms", "15_gradient_param_norms.png", [c for c, _ in grad_cols],
            DASHBOARD_SECTIONS[4],
            "Confirms the actor and critic weights are actually changing each update.", True,
            lambda: _plot_multi_ma(
                df, grad_cols[:4], "Gradient and parameter update norms", "norm",
                out_dir, "15_gradient_param_norms.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    # --- 6. Per-instance learning ---
    heat_obj = obj_col or "cost"
    if heat_obj and "instance_name" in df.columns:
        _try(
            "Per-instance objective heatmap", "16_instance_objective_heatmap.png", [heat_obj],
            DASHBOARD_SECTIONS[5],
            f"Objective per RC _21 instance across cycles ({n_inst} rows).", True,
            lambda: _heatmap_instance_cycle(
                df, heat_obj, "Per-instance objective heatmap",
                out_dir, "16_instance_objective_heatmap.png", n_inst=n_inst, save_svg=save_svg,
            ),
        )
    if served_col and "instance_name" in df.columns:
        _try(
            "Per-instance served-ratio heatmap", "17_instance_served_ratio_heatmap.png", [served_col],
            DASHBOARD_SECTIONS[5],
            "Served ratio per instance across cycles — detects repeating bad patterns.", True,
            lambda: _heatmap_instance_cycle(
                df, served_col, "Per-instance served-ratio heatmap",
                out_dir, "17_instance_served_ratio_heatmap.png", n_inst=n_inst, save_svg=save_svg,
            ),
        )

    # --- 7. EVRP operational metrics (minimal) ---
    if "avg_used_evs" in df.columns:
        _try(
            "Used EVs", "18_used_evs.png", ["avg_used_evs"],
            DASHBOARD_SECTIONS[6], "Fleet size used per episode.", False,
            lambda: _plot_series_ma(
                df, "avg_used_evs", "Used EVs", "count",
                out_dir, "18_used_evs.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    op_cols = [(c, l) for c, l in [
        ("avg_distance", "distance"),
        ("avg_charging_time", "charging time"),
        ("charging_station_visits", "station visits"),
    ] if c in df.columns]
    if len(op_cols) >= 2:
        _try(
            "Distance and charging behavior", "19_distance_charging.png", [c for c, _ in op_cols],
            DASHBOARD_SECTIONS[6], "Minimal operational routing metrics.", False,
            lambda: _plot_multi_ma(
                df, op_cols, "Distance and charging behavior", "value",
                out_dir, "19_distance_charging.png", ma_window=ma, save_svg=save_svg,
            ),
        )

    report.figures = results
    report.index_md_path = str(_write_index_md(results, out_dir, ma))
    report.html_path = str(_write_index_html(results, out_dir, ma))
    return report


def _write_index_md(figures: list[FigureResult], out_dir: Path, ma_window: int) -> Path:
    path = out_dir / "figures_index.md"
    lines = [
        "# MARL EVRP Training Dashboard — Figure Index",
        "",
        "Training metrics only. No validation or held-out test instances.",
        f"Moving average window: **{ma_window}** (= training instances per cycle).",
        "",
        "| Graph | File | Required | Metrics | Status | Explanation |",
        "|-------|------|----------|---------|--------|-------------|",
    ]
    for f in figures:
        req = "yes" if f.required else "no"
        status = "generated" if f.generated else f"skipped ({f.skip_reason or 'unavailable'})"
        metrics = ", ".join(f.metrics) if f.metrics else "—"
        lines.append(
            f"| {f.name} | `{f.filename}` | {req} | {metrics} | {status} | {f.description} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_index_html(
    figures: list[FigureResult],
    out_dir: Path,
    ma_window: int,
    *,
    extra_html: str = "",
) -> Path:
    by_section: dict[str, list[FigureResult]] = {s: [] for s in DASHBOARD_SECTIONS}
    for f in figures:
        by_section.setdefault(f.section, []).append(f)

    parts = [
        "<!DOCTYPE html>",
        "<html lang='en'><head><meta charset='utf-8'>",
        "<title>MARL EVRP Training Dashboard</title>",
        "<style>",
        "body{font-family:system-ui,sans-serif;margin:0;background:#0f172a;color:#e2e8f0}",
        "header{padding:1.5rem 2rem;background:#1e293b;border-bottom:1px solid #334155}",
        "main{padding:1.5rem 2rem;max-width:1200px;margin:0 auto}",
        "h1{margin:0;font-size:1.5rem} .sub{color:#94a3b8;font-size:.9rem;margin-top:.35rem}",
        "section{margin:2rem 0} h2{font-size:1.05rem;color:#93c5fd;border-bottom:1px solid #334155;padding-bottom:.4rem}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:1.25rem;margin-top:1rem}",
        ".card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem}",
        ".req{border-left:3px solid #f59e0b}",
        ".card h3{font-size:.95rem;margin:0 0 .5rem} .card p{font-size:.8rem;color:#94a3b8;margin:.25rem 0 .75rem}",
        ".card img{width:100%;border-radius:4px;background:#fff}",
        ".skip{color:#f87171;font-size:.85rem} .tag{font-size:.7rem;color:#f59e0b}",
        "a{color:#60a5fa}",
        "</style></head><body>",
        "<header><h1>MARL EVRP Training Dashboard</h1>",
        f"<p class='sub'>Schneider RC _21 — training only · MA window = {ma_window} · "
        "no validation or test data</p></header><main>",
    ]
    for title in DASHBOARD_SECTIONS:
        items = by_section.get(title, [])
        if not items:
            continue
        sid = title.split(".", 1)[0]
        parts.append(f"<section id='s{sid}'><h2>{title}</h2><div class='grid'>")
        for f in items:
            cls = "card req" if f.required else "card"
            parts.append(f"<div class='{cls}'>")
            req_tag = " <span class='tag'>critical</span>" if f.required else ""
            parts.append(f"<h3>{f.name}{req_tag}</h3><p>{f.description}</p>")
            if f.generated and f.png_path:
                rel = Path(f.png_path).name
                if rel.endswith(".html"):
                    parts.append(f"<p><a href='{rel}'>Open table: {f.name}</a></p>")
                else:
                    parts.append(f"<img src='{rel}' alt='{f.name}' loading='lazy'>")
                    if f.svg_path:
                        parts.append(f"<p><a href='{Path(f.svg_path).name}'>SVG</a></p>")
            else:
                parts.append(f"<p class='skip'>Skipped — {f.skip_reason or 'metrics missing'}</p>")
            parts.append("</div>")
        parts.append("</div></section>")
    if extra_html:
        parts.append(
            "<section id='s8-table'><h2>8. Solution behavior diagnostics — Final-cycle table</h2>"
            "<style>.diag-table{border-collapse:collapse;width:100%;font-size:12px}"
            "th,td{border:1px solid #334155;padding:4px 8px}th{background:#1e293b;color:#e2e8f0}"
            "td{color:#e2e8f0}</style>"
            f"{extra_html}</section>"
        )
    parts.append(
        "<section><h2>Index</h2><p>See <a href='figures_index.md'>figures_index.md</a>.</p></section>"
    )
    parts.append("</main></body></html>")
    html_path = out_dir / "index.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def regenerate_dashboard_index(
    report: DashboardReport,
    out_dir: Path,
    *,
    extra_html: str = "",
) -> DashboardReport:
    """Rewrite index.html and figures_index.md after appending extra figures."""
    report.index_md_path = str(_write_index_md(report.figures, out_dir, report.ma_window))
    report.html_path = str(
        _write_index_html(report.figures, out_dir, report.ma_window, extra_html=extra_html)
    )
    return report


def generate_training_dashboard_from_config(
    config: dict,
    rows: list[dict] | None = None,
    *,
    holdout_meta: dict | None = None,
    moving_avg_window: int | None = None,
    save_svg: bool = True,
    skip_train_count_check: bool = False,
) -> DashboardReport:
    """Load metrics from config paths / rows and write dashboard artifacts."""
    out_cfg = config.get("outputs") or {}
    figures_dir = Path(out_cfg.get("figures", "outputs/figures")) / "training"
    ds_cfg = config.get("dataset") or {}

    holdout = list((holdout_meta or {}).get("eval_instances") or ds_cfg.get("eval_instances") or [])
    train_inst = list((holdout_meta or {}).get("train_instances") or ds_cfg.get("train_instances") or [])

    if rows:
        df = pd.DataFrame(rows)
        if train_inst and "instance_name" in df.columns:
            df = df[df["instance_name"].isin(train_inst)]
        elif holdout and "instance_name" in df.columns:
            df = df[~df["instance_name"].isin(holdout)]
    else:
        csv_path = Path(out_cfg.get("results", "outputs/results")) / "training_metrics.csv"
        jsonl_path = Path(out_cfg.get("logs", "outputs/logs")) / "train_log.jsonl"
        df = load_training_dataframe(
            csv_path=csv_path, jsonl_path=jsonl_path,
            train_instances=train_inst or None, holdout_instances=holdout or None,
        )

    if skip_train_count_check:
        ma = int(moving_avg_window or ds_cfg.get("expected_train_count") or REQUIRED_RC21_TRAIN_COUNT)
        n_train = ma
    else:
        ma = int(moving_avg_window) if moving_avg_window else resolve_dashboard_ma_window(config, df, holdout_meta)
        n_train = ma

    report = generate_training_dashboard(
        df, figures_dir, moving_avg_window=ma, n_train_instances=n_train, save_svg=save_svg,
    )
    n_gen = sum(1 for f in report.figures if f.generated)
    LOGGER.info(
        "Training dashboard: %d/%d figures → %s (open index.html)",
        n_gen, len(report.figures), figures_dir,
    )
    return report
