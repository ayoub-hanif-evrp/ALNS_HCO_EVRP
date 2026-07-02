"""Generate static MARL EVRP training figures and HTML dashboard from training logs only."""
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

DEFAULT_MA_WINDOW = 20


@dataclass
class FigureResult:
    """Outcome of one dashboard figure."""

    name: str
    filename: str
    metrics: list[str]
    section: str
    description: str
    generated: bool
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


def _moving_avg(values: pd.Series, window: int) -> pd.Series:
    w = max(1, int(window))
    return values.astype(float).rolling(window=w, min_periods=1).mean()


def _epoch_x(df: pd.DataFrame) -> np.ndarray:
    if "epoch" in df.columns:
        return df["epoch"].to_numpy(dtype=float)
    return np.arange(len(df), dtype=float)


def _cycle_col(df: pd.DataFrame) -> pd.Series:
    if "pass" in df.columns and df["pass"].notna().any():
        return df["pass"].astype(int)
    if "cycle" in df.columns and df["cycle"].notna().any():
        return df["cycle"].astype(int)
    n_inst = df["instance_name"].nunique() if "instance_name" in df.columns else 20
    n_inst = max(int(n_inst), 1)
    return pd.Series((df.index // n_inst) + 1, index=df.index)


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
) -> tuple[Path | None, Path | None]:
    x = _epoch_x(df)
    y = df[col].astype(float)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(x, y, color="#94a3b8", lw=1, alpha=0.65, label="per epoch")
    ax.plot(x, _moving_avg(y, ma_window), color="#2563eb", lw=2, label=f"MA ({ma_window})")
    ax.set_title(title)
    ax.set_xlabel("training epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return _save_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_multi_series(
    df: pd.DataFrame,
    series: list[tuple[str, str]],
    title: str,
    ylabel: str,
    out_dir: Path,
    filename: str,
    *,
    ma_window: int,
    save_svg: bool,
) -> tuple[Path | None, Path | None]:
    x = _epoch_x(df)
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#dc2626", "#d97706", "#16a34a", "#7c3aed", "#0891b2", "#64748b"]
    for i, (col, label) in enumerate(series):
        y = df[col].astype(float)
        c = colors[i % len(colors)]
        ax.plot(x, _moving_avg(y, ma_window), lw=2, color=c, label=label)
    ax.set_title(title)
    ax.set_xlabel("training epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return _save_fig(fig, out_dir, filename, save_svg=save_svg)


def _cycle_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    cycles = _cycle_col(df)
    work = df.copy()
    work["_cycle"] = cycles
    agg_cols = {
        c: "mean"
        for c in [
            "cost", "feasibility_rate", "served_rate", "avg_violation",
            "unserved_customers", "avg_lateness", "avg_battery_violation",
            "avg_capacity_violation", "avg_depot_violation", "avg_duplicate_visits",
        ]
        if c in work.columns
    }
    if not agg_cols:
        return pd.DataFrame()
    return work.groupby("_cycle", as_index=False).agg(agg_cols).rename(columns={"_cycle": "cycle"})


def _heatmap_instance_cycle(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_dir: Path,
    filename: str,
    *,
    save_svg: bool,
) -> tuple[Path | None, Path | None]:
    if "instance_name" not in df.columns or value_col not in df.columns:
        return None, None
    work = df.copy()
    work["_cycle"] = _cycle_col(work)
    pivot = work.pivot_table(index="instance_name", columns="_cycle", values=value_col, aggfunc="mean")
    if pivot.empty:
        return None, None
    pivot = pivot.sort_index()
    fig, ax = plt.subplots(figsize=(10, max(4, 0.25 * len(pivot))))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd_r")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=7)
    ax.set_xlabel("training cycle (pass)")
    ax.set_ylabel("training instance")
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
    moving_avg_window: int = DEFAULT_MA_WINDOW,
    save_svg: bool = True,
) -> DashboardReport:
    """Build all training figures, index.md, and index.html."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = DashboardReport(num_epochs=len(df))
    if "instance_name" in df.columns:
        report.num_train_instances = int(df["instance_name"].nunique())

    if df.empty:
        _write_index_md(report.figures, out_dir)
        report.html_path = str(_write_index_html(report.figures, out_dir))
        report.index_md_path = str(out_dir / "figures_index.md")
        return report

    results: list[FigureResult] = []

    def _try(
        name: str,
        filename: str,
        metrics: list[str],
        section: str,
        description: str,
        fn: Callable[[], tuple[Path | None, Path | None]],
    ) -> None:
        try:
            png, svg = fn()
            reason = None if png else "required metrics missing or empty plot"
            _register(
                results, name=name, filename=filename, metrics=metrics, section=section,
                description=description, png=png, svg=svg, skip_reason=reason,
            )
        except Exception as exc:
            LOGGER.warning("Figure %s skipped: %s", filename, exc)
            _register(
                results, name=name, filename=filename, metrics=metrics, section=section,
                description=description, png=None, svg=None, skip_reason=str(exc),
            )

    cost_col = _pick(df, "cost", "mean_J", "cost_raw")
    if cost_col:
        _try(
            "Training objective over epochs", "01_training_objective.png", [cost_col],
            "Learning curves", "Shows whether the MARL policy is reducing the EVRP solution cost.",
            lambda: _plot_series_ma(
                df, cost_col, "Training objective over epochs", "objective J",
                out_dir, "01_training_objective.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )
    else:
        _register(results, name="Training objective over epochs", filename="01_training_objective.png",
                  metrics=["cost"], section="Learning curves", description="Objective cost per epoch.",
                  png=None, svg=None, skip_reason="cost column missing")

    if "reward" in df.columns:
        _try(
            "Episode return over epochs", "02_episode_return.png", ["reward"],
            "Learning curves", "Shows whether the actor-critic learning signal improves over time.",
            lambda: _plot_series_ma(
                df, "reward", "Episode return over epochs", "episode return",
                out_dir, "02_episode_return.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    if "feasibility_rate" in df.columns:
        x = _epoch_x(df)
        y = _moving_avg(df["feasibility_rate"].astype(float), moving_avg_window)
        def _feas():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, y, color="#16a34a", lw=2)
            ax.set_title("Feasibility moving average")
            ax.set_xlabel("training epoch")
            ax.set_ylabel("feasibility rate")
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout()
            return _save_fig(fig, out_dir, "03_feasibility_moving_average.png", save_svg=save_svg)
        _try(
            "Feasibility moving average", "03_feasibility_moving_average.png", ["feasibility_rate"],
            "Feasibility and constraints",
            "Shows whether the model produces feasible EVRP solutions more often.", _feas,
        )

    served_col = _pick(df, "served_rate", "served_rate_pct")
    if served_col:
        ylab = "served rate" if served_col == "served_rate" else "served %"
        _try(
            "Served-customer ratio over epochs", "04_served_customer_ratio.png", [served_col],
            "Feasibility and constraints", "Shows whether the policy learns to serve all customers.",
            lambda: _plot_series_ma(
                df, served_col, "Served-customer ratio over epochs", ylab,
                out_dir, "04_served_customer_ratio.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    viol_series: list[tuple[str, str]] = []
    for col, label in [
        ("unserved_customers", "unserved"),
        ("avg_battery_violation", "battery"),
        ("avg_lateness", "time window"),
        ("avg_capacity_violation", "capacity"),
        ("avg_depot_violation", "depot return"),
        ("avg_duplicate_visits", "duplicate"),
        ("mean_pen_unserved", "pen. unserved"),
        ("mean_pen_battery", "pen. battery"),
        ("mean_pen_late", "pen. late"),
        ("mean_pen_capacity", "pen. capacity"),
        ("mean_pen_depot", "pen. depot"),
        ("mean_pen_duplicate", "pen. duplicate"),
    ]:
        if col in df.columns and df[col].notna().any():
            viol_series.append((col, label))
    if viol_series:
        _try(
            "Constraint violations over training", "05_constraint_violations.png",
            [c for c, _ in viol_series], "Feasibility and constraints",
            "EVRP constraint learning: battery, capacity, time windows, depot return.",
            lambda: _plot_multi_series(
                df, viol_series[:6], "Constraint violations over training", "violation magnitude",
                out_dir, "05_constraint_violations.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    loss_series: list[tuple[str, str]] = []
    for col, label in [
        ("actor_loss", "actor"), ("critic_loss", "critic"), ("total_loss", "total"),
        ("q_critic_loss", "Q critic"), ("loss", "combined loss"),
    ]:
        if col in df.columns and df[col].astype(float).abs().sum() > 0:
            loss_series.append((col, label))
    if loss_series:
        _try(
            "Actor-critic losses", "06_actor_critic_losses.png", [c for c, _ in loss_series],
            "Learning curves", "Training stability of the MARL actor-critic system.",
            lambda: _plot_multi_series(
                df, loss_series, "Actor-critic losses", "loss",
                out_dir, "06_actor_critic_losses.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    if "entropy" in df.columns:
        _try(
            "Policy entropy over epochs", "07_policy_entropy.png", ["entropy"],
            "Learning curves", "Exploration — entropy should not collapse too early.",
            lambda: _plot_series_ma(
                df, "entropy", "Policy entropy over epochs", "entropy",
                out_dir, "07_policy_entropy.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    if "critic_explained_variance" in df.columns and df["critic_explained_variance"].notna().any():
        def _cev():
            x = _epoch_x(df)
            y = df["critic_explained_variance"].astype(float)
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(x, y, color="#7c3aed", lw=1.5)
            ax.set_title("Critic explained variance")
            ax.set_xlabel("training epoch")
            ax.set_ylabel("explained variance")
            fig.tight_layout()
            return _save_fig(fig, out_dir, "08_critic_explained_variance.png", save_svg=save_svg)
        _try(
            "Critic explained variance", "08_critic_explained_variance.png", ["critic_explained_variance"],
            "Learning curves", "Whether the critic learns useful value estimates.", _cev,
        )

    obj_pairs = [
        ("avg_distance", "distance"), ("avg_duration", "duration"),
        ("avg_charging_time", "charging time"), ("avg_charging_price", "charging cost"),
        ("avg_waiting_time", "waiting time"),
    ]
    obj_avail = [(c, l) for c, l in obj_pairs if c in df.columns]
    if len(obj_avail) >= 2:
        scales = [df[c].astype(float).std() for c, _ in obj_avail]
        if max(scales) > 10 * max(min(scales), 1e-9):
            dd = [(c, l) for c, l in obj_avail if c in ("avg_distance", "avg_duration")]
            cw = [(c, l) for c, l in obj_avail if c not in ("avg_distance", "avg_duration")]
            if dd:
                _try("Distance and duration", "09a_distance_duration.png", [c for c, _ in dd],
                     "EVRP operational behavior", "Route distance and duration over training.",
                     lambda: _plot_multi_series(
                         df, dd, "Distance and duration over training", "value",
                         out_dir, "09a_distance_duration.png", ma_window=moving_avg_window, save_svg=save_svg,
                     ))
            if cw:
                _try("Charging and waiting", "09b_charging_waiting.png", [c for c, _ in cw],
                     "EVRP operational behavior", "Charging and waiting effort over training.",
                     lambda: _plot_multi_series(
                         df, cw, "Charging and waiting over training", "value",
                         out_dir, "09b_charging_waiting.png", ma_window=moving_avg_window, save_svg=save_svg,
                     ))
        else:
            _try(
                "Objective components over training", "09_objective_components.png",
                [c for c, _ in obj_avail], "EVRP operational behavior",
                "Which parts of the EVRP cost are improving.",
                lambda: _plot_multi_series(
                    df, obj_avail, "Objective components over training", "value",
                    out_dir, "09_objective_components.png", ma_window=moving_avg_window, save_svg=save_svg,
                ),
            )

    if "avg_used_evs" in df.columns:
        _try(
            "Number of used EVs", "10_used_evs.png", ["avg_used_evs"],
            "EVRP operational behavior", "Whether the model learns efficient fleet usage.",
            lambda: _plot_series_ma(
                df, "avg_used_evs", "Number of used EVs", "used EVs",
                out_dir, "10_used_evs.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    if "charging_station_visits" in df.columns:
        _try(
            "Charging-station visits over training", "11a_station_visits.png", ["charging_station_visits"],
            "EVRP operational behavior", "EV recharging station usage during training.",
            lambda: _plot_series_ma(
                df, "charging_station_visits", "Charging-station visits over training", "visits",
                out_dir, "11a_station_visits.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )
    charge_effort = [(c, l) for c, l in [
        ("avg_charging_time", "charging time"), ("avg_charging_price", "charging cost"),
    ] if c in df.columns]
    if charge_effort:
        _try(
            "Charging effort over training", "11b_charging_effort.png", [c for c, _ in charge_effort],
            "EVRP operational behavior", "Total charging time and monetary charging cost.",
            lambda: _plot_multi_series(
                df, charge_effort, "Charging effort over training", "value",
                out_dir, "11b_charging_effort.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )
    if "min_remaining_battery" in df.columns:
        _try(
            "Minimum battery level over training", "11c_min_battery.png", ["min_remaining_battery"],
            "EVRP operational behavior", "Minimum remaining battery at episode end.",
            lambda: _plot_series_ma(
                df, "min_remaining_battery", "Minimum battery level over training", "battery level",
                out_dir, "11c_min_battery.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    if "route_balance_cv" in df.columns:
        _try(
            "Fleet coordination and route balance", "12_fleet_coordination_balance.png", ["route_balance_cv"],
            "MARL coordination", "Std/mean of per-EV route distance — load balance across agents.",
            lambda: _plot_series_ma(
                df, "route_balance_cv", "Fleet coordination and route balance", "route balance (CV)",
                out_dir, "12_fleet_coordination_balance.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    far_col = _pick(df, "feasible_action_ratio")
    if far_col:
        _try(
            "Feasible action ratio", "13_feasible_action_ratio.png", [far_col],
            "Feasibility and constraints", "How restrictive environment masks are during training.",
            lambda: _plot_series_ma(
                df, far_col, "Feasible action ratio", "ratio",
                out_dir, "13_feasible_action_ratio.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )
    elif "mean_feasible_actions" in df.columns and "num_nodes" in df.columns:
        work = df.copy()
        work["feasible_action_ratio"] = work["mean_feasible_actions"] / work["num_nodes"].clip(lower=1)
        _try(
            "Feasible action ratio", "13_feasible_action_ratio.png",
            ["mean_feasible_actions", "num_nodes"], "Feasibility and constraints",
            "Feasible actions per step relative to graph size.",
            lambda: _plot_series_ma(
                work, "feasible_action_ratio", "Feasible action ratio", "ratio",
                out_dir, "13_feasible_action_ratio.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    cyc = _cycle_aggregate(df)
    if not cyc.empty and "cost" in cyc.columns:
        def _cobj():
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.plot(cyc["cycle"], cyc["cost"], marker="o", color="#2563eb")
            ax.set_title("Cycle-level objective progress")
            ax.set_xlabel("training cycle")
            ax.set_ylabel("mean objective")
            fig.tight_layout()
            return _save_fig(fig, out_dir, "14a_cycle_objective.png", save_svg=save_svg)
        _try(
            "Cycle-level objective progress", "14a_cycle_objective.png", ["cost"],
            "Cyclic schedule analysis", "Mean objective per full pass over training instances.", _cobj,
        )

    if not cyc.empty:
        fs_cols = [c for c in ("feasibility_rate", "served_rate") if c in cyc.columns]
        if fs_cols:
            def _cfs():
                fig, ax = plt.subplots(figsize=(9, 4))
                for col in fs_cols:
                    ax.plot(cyc["cycle"], cyc[col], marker="o", label=col.replace("_", " "))
                ax.set_title("Cycle-level feasibility and served ratio")
                ax.set_xlabel("training cycle")
                ax.legend()
                fig.tight_layout()
                return _save_fig(fig, out_dir, "14b_cycle_feasibility_served.png", save_svg=save_svg)
            _try(
                "Cycle-level feasibility and served ratio", "14b_cycle_feasibility_served.png", fs_cols,
                "Cyclic schedule analysis", "Aggregate feasibility and coverage per cycle.", _cfs,
            )
        vcols = [c for c in (
            "avg_violation", "unserved_customers", "avg_battery_violation", "avg_lateness",
        ) if c in cyc.columns]
        if vcols:
            _try(
                "Cycle-level violations", "14c_cycle_violations.png", vcols,
                "Cyclic schedule analysis", "Mean violations per training cycle.",
                lambda: _plot_multi_series(
                    cyc.rename(columns={"cycle": "epoch"}), [(c, c) for c in vcols],
                    "Cycle-level violations", "mean violation",
                    out_dir, "14c_cycle_violations.png", ma_window=1, save_svg=save_svg,
                ),
            )

    if cost_col and "instance_name" in df.columns:
        _try(
            "Per-instance objective across cycles", "15a_per_instance_objective_heatmap.png", [cost_col],
            "Cyclic schedule analysis", "Objective heatmap: instances × cycles.",
            lambda: _heatmap_instance_cycle(
                df, cost_col, "Per-instance objective across cycles",
                out_dir, "15a_per_instance_objective_heatmap.png", save_svg=save_svg,
            ),
        )
    if "feasibility_rate" in df.columns and "instance_name" in df.columns:
        _try(
            "Per-instance feasibility across cycles", "15b_per_instance_feasibility_heatmap.png",
            ["feasibility_rate"], "Cyclic schedule analysis", "Feasibility heatmap: instances × cycles.",
            lambda: _heatmap_instance_cycle(
                df, "feasibility_rate", "Per-instance feasibility across cycles",
                out_dir, "15b_per_instance_feasibility_heatmap.png", save_svg=save_svg,
            ),
        )

    if cost_col and "instance_name" in df.columns:
        last_cycle = int(_cycle_col(df).max())
        final = df[_cycle_col(df) == last_cycle]
        if not final.empty:
            inst_mean = final.groupby("instance_name")[cost_col].mean().sort_values(ascending=False)
            def _final_bar():
                fig, ax = plt.subplots(figsize=(10, max(4, 0.22 * len(inst_mean))))
                ax.barh(range(len(inst_mean)), inst_mean.to_numpy(), color="#dc2626")
                ax.set_yticks(range(len(inst_mean)))
                ax.set_yticklabels(inst_mean.index, fontsize=7)
                ax.invert_yaxis()
                ax.set_title("Final-cycle objective by training instance")
                ax.set_xlabel("mean objective")
                fig.tight_layout()
                return _save_fig(fig, out_dir, "16_final_cycle_instance_objective.png", save_svg=save_svg)
            _try(
                "Final-cycle objective by training instance", "16_final_cycle_instance_objective.png",
                [cost_col], "Cyclic schedule analysis", "Hardest training instances in the last cycle.",
                _final_bar,
            )

    stab_series: list[tuple[str, str]] = []
    for col, label in [
        ("grad_norm", "grad norm"),
        ("cuda_max_mem_allocated_mb", "peak GPU MB"),
        ("cuda_mem_allocated_mb", "GPU allocated MB"),
    ]:
        if col in df.columns and df[col].notna().any():
            stab_series.append((col, label))
    if stab_series:
        _try(
            "Training stability diagnostics", "17_training_stability_diagnostics.png",
            [c for c, _ in stab_series], "Learning curves",
            "Gradient norms and GPU memory during training.",
            lambda: _plot_multi_series(
                df, stab_series, "Training stability diagnostics", "value",
                out_dir, "17_training_stability_diagnostics.png", ma_window=moving_avg_window, save_svg=save_svg,
            ),
        )

    report.figures = results
    report.index_md_path = str(_write_index_md(results, out_dir))
    report.html_path = str(_write_index_html(results, out_dir))
    return report


def _write_index_md(figures: list[FigureResult], out_dir: Path) -> Path:
    path = out_dir / "figures_index.md"
    lines = [
        "# MARL EVRP Training Figures Index",
        "",
        "Training-only metrics (no validation or held-out test instances).",
        "",
        "| Graph | File | Metrics | Status | Explanation |",
        "|-------|------|---------|--------|-------------|",
    ]
    for f in figures:
        status = "generated" if f.generated else f"skipped ({f.skip_reason or 'unavailable'})"
        metrics = ", ".join(f.metrics) if f.metrics else "—"
        lines.append(f"| {f.name} | `{f.filename}` | {metrics} | {status} | {f.description} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_index_html(figures: list[FigureResult], out_dir: Path) -> Path:
    sections_order = [
        ("Learning curves", "learning"),
        ("Feasibility and constraints", "feasibility"),
        ("EVRP operational behavior", "operations"),
        ("MARL coordination", "coordination"),
        ("Cyclic schedule analysis", "cycles"),
    ]
    by_section: dict[str, list[FigureResult]] = {s: [] for s, _ in sections_order}
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
        "section{margin:2rem 0} h2{font-size:1.1rem;color:#93c5fd;border-bottom:1px solid #334155;padding-bottom:.4rem}",
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:1.25rem;margin-top:1rem}",
        ".card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem}",
        ".card h3{font-size:.95rem;margin:0 0 .5rem} .card p{font-size:.8rem;color:#94a3b8;margin:.25rem 0 .75rem}",
        ".card img{width:100%;border-radius:4px;background:#fff}",
        ".skip{color:#f87171;font-size:.85rem}",
        "a{color:#60a5fa}",
        "</style></head><body>",
        "<header><h1>MARL EVRP Training Dashboard</h1>",
        "<p class='sub'>Training metrics only — Schneider R _21, 20 instances × 20 cycles. "
        "No validation or held-out test data.</p></header><main>",
    ]
    for title, _ in sections_order:
        items = by_section.get(title, [])
        if not items:
            continue
        parts.append(f"<section id='{title.lower().replace(' ','-')}'><h2>{title}</h2><div class='grid'>")
        for f in items:
            parts.append("<div class='card'>")
            parts.append(f"<h3>{f.name}</h3><p>{f.description}</p>")
            if f.generated and f.png_path:
                rel = Path(f.png_path).name
                parts.append(f"<img src='{rel}' alt='{f.name}' loading='lazy'>")
                if f.svg_path:
                    parts.append(f"<p><a href='{Path(f.svg_path).name}'>SVG</a></p>")
            else:
                parts.append(f"<p class='skip'>Unavailable — {f.skip_reason or 'metrics missing'}</p>")
            parts.append("</div>")
        parts.append("</div></section>")
    parts.append(
        "<section><h2>Index</h2><p>See <a href='figures_index.md'>figures_index.md</a> for metrics and status.</p></section>"
    )
    parts.append("</main></body></html>")
    html_path = out_dir / "index.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def generate_training_dashboard_from_config(
    config: dict,
    rows: list[dict] | None = None,
    *,
    holdout_meta: dict | None = None,
    moving_avg_window: int | None = None,
    save_svg: bool = True,
) -> DashboardReport:
    """Load metrics from config paths / rows and write dashboard artifacts."""
    out_cfg = config.get("outputs") or {}
    figures_dir = Path(out_cfg.get("figures", "outputs/figures")) / "training"
    tcfg = config.get("training") or {}
    ds_cfg = config.get("dataset") or {}
    ma = int(moving_avg_window or tcfg.get("dashboard_ma_window", DEFAULT_MA_WINDOW))

    holdout = list((holdout_meta or {}).get("eval_instances") or ds_cfg.get("eval_instances") or [])
    train_inst = list(ds_cfg.get("train_instances") or [])

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

    report = generate_training_dashboard(df, figures_dir, moving_avg_window=ma, save_svg=save_svg)
    n_gen = sum(1 for f in report.figures if f.generated)
    LOGGER.info(
        "Training dashboard: %d/%d figures → %s (open index.html)",
        n_gen, len(report.figures), figures_dir,
    )
    return report
