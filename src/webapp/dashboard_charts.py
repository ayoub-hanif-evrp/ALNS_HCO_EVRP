"""Plotly chart builders for training-progress dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.webapp.dashboard_data import (
    COST_COLUMNS,
    COST_OBJECTIVE_TERMS,
    COST_PENALTY_TERMS,
    cycle_aggregate_df,
    cycle_column,
)
from src.webapp.plot_style import (
    CLASS_COLORS,
    CLASS_ORDER,
    COMPONENT_COLORS,
    CURRICULUM_STAGE_COLORS,
    OVERALL_LINE_COLOR,
    curriculum_stage_color,
)

CHART_LAYOUT = dict(
    template="plotly_white",
    height=340,
    autosize=True,
    margin=dict(t=28, b=40, l=52, r=20),
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.14, x=0, font=dict(size=11)),
)


def _chart_layout(**overrides) -> dict:
    return {**CHART_LAYOUT, **overrides}


def _chart_layout_embedded(**overrides) -> dict:
    """External HTML title — no Plotly title; legend sits below the plot."""
    layout = {**CHART_LAYOUT, **overrides}
    layout["title"] = None
    layout["showlegend"] = True
    layout["legend"] = dict(orientation="h", yanchor="top", y=-0.28, x=0, font=dict(size=10))
    layout["margin"] = overrides.get("margin", dict(t=8, b=62, l=52, r=20))
    return layout


def _x_series(df: pd.DataFrame) -> pd.Series:
    if "episode_index" in df.columns:
        return df["episode_index"]
    if "training_update" in df.columns:
        return df["training_update"]
    return pd.Series(range(1, len(df) + 1), index=df.index)


def _class_col(df: pd.DataFrame) -> str:
    return "instance_class" if "instance_class" in df.columns else ""


def _marker_colors(df: pd.DataFrame) -> list[str]:
    col = _class_col(df)
    if not col:
        return [OVERALL_LINE_COLOR] * len(df)
    return [CLASS_COLORS.get(str(c).upper(), "#64748b") for c in df[col]]


def _x_epoch(df: pd.DataFrame) -> pd.Series:
    if "epoch" in df.columns:
        return df["epoch"]
    return _x_series(df)


def _overall_rolling_col(col: str) -> str:
    """Prepared global rolling-mean column for a metric."""
    aliases = {
        "raw_episode_return": "rolling_raw_return_mean",
        "reward": "rolling_raw_return_mean",
        "served_rate": "rolling_served_ratio_mean",
        "served_ratio": "rolling_served_ratio_mean",
    }
    if col in aliases:
        return aliases[col]
    return f"rolling_{col}_mean"


def _rolling_series(df: pd.DataFrame, col: str, plan: dict) -> pd.Series | None:
    """Rolling mean for ``col``, with on-the-fly fallback when not precomputed."""
    if col not in df.columns:
        return None
    roll_col = _overall_rolling_col(col)
    if roll_col in df.columns:
        return df[roll_col].astype(float)
    alt = f"rolling_{col}_mean"
    if alt in df.columns:
        return df[alt].astype(float)
    w = max(1, int(plan.get("ma_window") or 20))
    return df[col].astype(float).rolling(window=w, min_periods=1).mean()


def add_pass_boundaries(fig: go.Figure, df: pd.DataFrame, *, max_x: float | None = None) -> None:
    """Light vertical lines at training-pass boundaries (no colored bands)."""
    if df.empty or "training_cycle" not in df.columns:
        return
    mx = max_x if max_x is not None else float(_x_epoch(df).max())
    x = _x_epoch(df).to_numpy()
    cycles = df["training_cycle"].astype(int).to_numpy()
    for i in range(1, len(cycles)):
        if cycles[i] != cycles[i - 1]:
            boundary = float(x[i]) - 0.5
            if 0 < boundary <= mx + 0.5:
                fig.add_vline(
                    x=boundary,
                    line_dash="dot",
                    line_color="#cbd5e1",
                    line_width=1,
                    opacity=0.8,
                )


def add_training_regions(fig: go.Figure, df: pd.DataFrame, plan: dict, *, max_x: float | None = None) -> None:
    if plan.get("training_mode") == "flat" and "training_cycle" in df.columns:
        add_pass_boundaries(fig, df, max_x=max_x)
    else:
        add_chart_regions(fig, df, max_x=max_x)


def fig_series_ma(
    df: pd.DataFrame,
    col: str,
    plan: dict,
    *,
    title: str,
    ylabel: str,
    y_range: tuple[float, float] | None = None,
    hline: float | None = None,
) -> go.Figure:
    fig = go.Figure()
    if df.empty or col not in df.columns:
        fig.update_layout(**_chart_layout())
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    x = _x_epoch(df)
    y = df[col].astype(float)
    ma = _rolling_series(df, col, plan)
    fig.add_trace(
        go.Scatter(
            x=x, y=y, mode="markers", name="Per epoch",
            marker=dict(size=4, color="#94a3b8", opacity=0.45),
        )
    )
    if ma is not None:
        fig.add_trace(
            go.Scatter(
                x=x, y=ma, mode="lines", name="Moving average",
                line=dict(color="#2563eb", width=2.5),
            )
        )
    if hline is not None:
        fig.add_hline(
            y=float(hline), line_dash="dash", line_color="#ef4444", line_width=1.5,
            annotation_text=f"ref {hline}", annotation_position="top right",
        )
    add_training_regions(fig, df, plan)
    layout = _chart_layout_embedded(xaxis_title="Training epoch", yaxis_title=ylabel)
    if y_range:
        layout["yaxis"] = dict(range=list(y_range))
    fig.update_layout(**layout)
    return fig


def fig_explained_variance(df: pd.DataFrame, plan: dict, *, title: str) -> go.Figure:
    col = "critic_explained_variance"
    fig = go.Figure()
    if df.empty or col not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    x = _x_epoch(df)
    y = df[col].astype(float)
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="Explained variance", line=dict(color="#7c3aed", width=2))
    )
    fig.add_hline(y=0.0, line_dash="dash", line_color="#64748b", line_width=1.5)
    add_training_regions(fig, df, plan)
    fig.update_layout(**_chart_layout_embedded(xaxis_title="Training epoch", yaxis_title="Explained variance"))
    return fig


def fig_best_objective_so_far(df: pd.DataFrame, plan: dict, *, title: str) -> go.Figure:
    """Best-so-far objective curve (standard RL sample-efficiency / anytime plot)."""
    obj_col = _pick_metric_col(df, "final_objective", "cost", "cost_raw")
    best_col = _pick_metric_col(df, "best_objective_so_far", "cost_best")
    fig = go.Figure()
    if df.empty or not best_col or best_col not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    x = _x_epoch(df)
    if obj_col and obj_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[obj_col].astype(float),
                mode="markers",
                name="Per epoch",
                marker=dict(size=4, color="#94a3b8", opacity=0.4),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=df[best_col].astype(float),
            mode="lines",
            name="Best so far",
            line=dict(color="#16a34a", width=2.8),
        )
    )
    add_training_regions(fig, df, plan)
    fig.update_layout(
        **_chart_layout_embedded(
            xaxis_title="Training epoch",
            yaxis_title="Objective J (lower is better)",
        )
    )
    return fig


def fig_sample_efficiency_quality(df: pd.DataFrame, plan: dict, *, title: str) -> go.Figure:
    """CB-DRL-style dual metric: served coverage % and feasibility rate over training."""
    served_col = _pick_metric_col(df, "served_ratio", "served_rate")
    feas_col = "feasibility_rate"
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty or not served_col or served_col not in df.columns or feas_col not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    x = _x_epoch(df)
    served_ma = _rolling_series(df, served_col, plan)
    served_y = (served_ma if served_ma is not None else df[served_col].astype(float)) * 100.0
    feas_ma = _rolling_series(df, feas_col, plan)
    feas_y = feas_ma if feas_ma is not None else df[feas_col].astype(float)
    fig.add_trace(
        go.Scatter(
            x=x, y=served_y, mode="lines", name="Served % (MA)",
            line=dict(color="#2563eb", width=2.5),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=feas_y, mode="lines", name="Feasibility (MA)",
            line=dict(color="#16a34a", width=2.5, dash="dot"),
        ),
        secondary_y=True,
    )
    add_training_regions(fig, df, plan)
    fig.update_layout(**_chart_layout_embedded(xaxis_title="Training epoch"))
    fig.update_yaxes(title_text="Served (%)", range=[0, 105], secondary_y=False)
    fig.update_yaxes(title_text="Feasibility rate", range=[-0.05, 1.05], secondary_y=True)
    fig.update_layout(title=dict(text=title, x=0.02, xanchor="left"))
    return fig


def _constraint_violation_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Hard-constraint violations only (unserved tracked separately)."""
    return [(c, l) for c, l in _violation_series(df) if c != "unserved_customers"]


def fig_feasibility_ma(df: pd.DataFrame, plan: dict, *, title: str) -> go.Figure:
    col = "feasibility_rate"
    fig = go.Figure()
    if df.empty or col not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    x = _x_epoch(df)
    roll = "rolling_feasibility_rate_mean"
    if roll in df.columns:
        y = df[roll]
    else:
        w = max(1, int(plan.get("ma_window") or 20))
        y = df[col].astype(float).rolling(window=w, min_periods=1).mean()
    fig.add_trace(
        go.Scatter(x=x, y=y, mode="lines", name="Feasibility MA", line=dict(color="#16a34a", width=2.5))
    )
    add_training_regions(fig, df, plan)
    fig.update_layout(
        **_chart_layout_embedded(
            xaxis_title="Training epoch",
            yaxis_title="Feasibility rate",
            yaxis=dict(range=[-0.05, 1.05]),
        )
    )
    return fig


def fig_multi_series_ma(
    df: pd.DataFrame,
    series: list[tuple[str, str]],
    plan: dict,
    *,
    title: str,
    ylabel: str,
) -> go.Figure:
    fig = go.Figure()
    avail = [(c, l) for c, l in series if c in df.columns and df[c].notna().any()]
    if df.empty or not avail:
        fig.update_layout(**_chart_layout())
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig
    x = _x_epoch(df)
    colors = ["#dc2626", "#d97706", "#16a34a", "#7c3aed", "#0891b2", "#64748b"]
    for i, (col, label) in enumerate(avail):
        ma = _rolling_series(df, col, plan)
        y = ma if ma is not None else df[col].astype(float)
        fig.add_trace(
            go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(color=colors[i % len(colors)], width=2))
        )
    add_training_regions(fig, df, plan)
    fig.update_layout(**_chart_layout_embedded(xaxis_title="Training epoch", yaxis_title=ylabel))
    return fig


def fig_cycle_multi_lines(
    df: pd.DataFrame,
    cols: list[tuple[str, str]],
    plan: dict,
    *,
    title: str,
    ylabel: str,
) -> go.Figure:
    del plan
    cyc = cycle_aggregate_df(df)
    avail = [(c, l) for c, l in cols if c in cyc.columns]
    fig = go.Figure()
    if cyc.empty or not avail:
        fig.update_layout(**_chart_layout())
        return fig
    colors = ["#2563eb", "#16a34a", "#d97706", "#7c3aed"]
    for i, (col, label) in enumerate(avail):
        fig.add_trace(
            go.Scatter(
                x=cyc["cycle"], y=cyc[col], mode="lines+markers", name=label,
                line=dict(color=colors[i % len(colors)], width=2),
            )
        )
    fig.update_layout(**_chart_layout_embedded(xaxis_title="Training cycle", yaxis_title=ylabel))
    return fig


def fig_heatmap_instance_cycle(
    df: pd.DataFrame,
    value_col: str,
    plan: dict,
    *,
    title: str,
) -> go.Figure:
    del plan
    fig = go.Figure()
    if df.empty or value_col not in df.columns or "instance_name" not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    work = df.copy()
    work["_cycle"] = cycle_column(work)
    pivot = work.pivot_table(index="instance_name", columns="_cycle", values=value_col, aggfunc="mean")
    if pivot.empty:
        fig.update_layout(**_chart_layout())
        return fig
    pivot = pivot.sort_index()
    fig.add_trace(
        go.Heatmap(
            z=pivot.to_numpy(dtype=float),
            x=[str(c) for c in pivot.columns],
            y=list(pivot.index),
            colorscale="YlOrRd_r",
            colorbar=dict(title=value_col),
        )
    )
    fig.update_layout(
        **_chart_layout(
            title=title,
            xaxis_title="Training cycle",
            yaxis_title="Instance",
            height=max(360, 18 * len(pivot)),
        )
    )
    return fig


def fig_final_cycle_instance_bar(df: pd.DataFrame, cost_col: str, plan: dict, *, title: str) -> go.Figure:
    del plan
    fig = go.Figure()
    if df.empty or cost_col not in df.columns or "instance_name" not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    cycles = cycle_column(df)
    last_cycle = int(cycles.max())
    final = df[cycles == last_cycle]
    if final.empty:
        fig.update_layout(**_chart_layout())
        return fig
    inst_mean = final.groupby("instance_name")[cost_col].mean().sort_values(ascending=True)
    fig.add_trace(
        go.Bar(
            x=inst_mean.to_numpy(),
            y=list(inst_mean.index),
            orientation="h",
            marker_color="#dc2626",
        )
    )
    fig.update_layout(
        **_chart_layout_embedded(
            xaxis_title="Mean objective",
            yaxis_title="Instance",
            height=max(360, 18 * len(inst_mean)),
        )
    )
    return fig


def _violation_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("unserved_customers", "unserved"),
        ("avg_battery_violation", "battery"),
        ("avg_lateness", "time window"),
        ("avg_capacity_violation", "capacity"),
        ("avg_depot_violation", "depot return"),
        ("avg_duplicate_visits", "duplicate"),
    ]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series[:6]


def _loss_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("actor_loss", "actor"),
        ("critic_loss", "critic"),
        ("total_loss", "total"),
        ("q_critic_loss", "Q critic"),
        ("loss", "combined"),
    ]:
        if col in df.columns and df[col].astype(float).abs().sum() > 0:
            series.append((col, label))
    return series


def _stability_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("grad_norm", "grad norm"),
        ("learning_rate", "learning rate"),
        ("cuda_max_mem_allocated_mb", "peak GPU MB"),
        ("cuda_mem_allocated_mb", "GPU allocated MB"),
    ]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series


def _objective_component_specs(df: pd.DataFrame) -> list[dict[str, Any]]:
    pairs = [
        ("avg_distance", "distance"),
        ("avg_duration", "duration"),
        ("avg_charging_time", "charging time"),
        ("avg_charging_price", "charging cost"),
        ("avg_waiting_time", "waiting time"),
    ]
    avail = [(c, l) for c, l in pairs if c in df.columns]
    if len(avail) < 2:
        return []
    scales = [float(df[c].astype(float).std()) for c, _ in avail]
    if max(scales) > 10 * max(min(scales), 1e-9):
        specs: list[dict[str, Any]] = []
        dd = [(c, l) for c, l in avail if c in ("avg_distance", "avg_duration")]
        cw = [(c, l) for c, l in avail if c not in ("avg_distance", "avg_duration")]
        if dd:
            specs.append({
                "type": "multi_series_ma",
                "series": dd,
                "title": "Distance and duration",
                "ylabel": "Value",
                "wide": True,
            })
        if cw:
            specs.append({
                "type": "multi_series_ma",
                "series": cw,
                "title": "Charging and waiting",
                "ylabel": "Value",
                "wide": True,
            })
        return specs
    return [{
        "type": "multi_series_ma",
        "series": avail,
        "title": "Objective components",
        "ylabel": "Value",
        "wide": True,
    }]


def _pick_metric_col(df: pd.DataFrame, *candidates: str) -> str | None:
    for col in candidates:
        if col in df.columns and df[col].notna().any():
            return col
    return None


def _reward_decomp_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("served_bonus_total", "served bonus"),
        ("early_depot_penalty_total", "early depot penalty"),
        ("terminal_unserved_penalty_scaled", "terminal unserved (scaled)"),
        ("mean_pen_unserved", "pen. unserved (objective)"),
        ("mean_pen_infeasible", "pen. infeasible"),
        ("step_penalty_total", "step penalty"),
    ]:
        if col in df.columns and df[col].notna().any() and float(df[col].abs().sum()) > 0:
            series.append((col, label))
    return series[:5]


def _return_comparison_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("raw_objective_return", "raw objective return"),
        ("shaped_training_return", "shaped training return"),
        ("reward", "episode return"),
    ]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series[:2] if len(series) >= 2 else series


def _action_fraction_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("fraction_customer_actions", "customer"),
        ("fraction_station_actions", "station"),
        ("fraction_depot_actions", "depot"),
    ]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series


def _grad_param_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("actor_grad_norm", "actor grad"),
        ("critic_grad_norm", "critic grad"),
        ("actor_param_delta_norm", "actor Δparam"),
        ("critic_param_delta_norm", "critic Δparam"),
        ("grad_norm", "total grad"),
    ]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series[:4]


def _advantage_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [("advantage", "mean"), ("advantage_std", "std")]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series


def _ops_series(df: pd.DataFrame) -> list[tuple[str, str]]:
    series: list[tuple[str, str]] = []
    for col, label in [
        ("avg_distance", "distance"),
        ("avg_charging_time", "charging time"),
        ("charging_station_visits", "station visits"),
    ]:
        if col in df.columns and df[col].notna().any():
            series.append((col, label))
    return series


def build_flat_training_sections(df: pd.DataFrame, plan: dict) -> list[dict[str, Any]]:
    """Paper-aligned MARL EVRP dashboard (TERRAN / CB-DRL / marl-eval conventions)."""
    n_inst = int(plan.get("n_train_instances") or 20)
    obj_col = _pick_metric_col(df, "final_objective", "cost", "cost_raw")
    served_col = _pick_metric_col(df, "served_rate", "served_ratio")
    losses = _loss_series(df)
    returns = _return_comparison_series(df)
    decomp = _reward_decomp_series(df)
    actions = _action_fraction_series(df)
    grads = _grad_param_series(df)
    adv = _advantage_series(df)
    ops = _ops_series(df)
    constraints = _constraint_violation_series(df)
    cycle_obj = [(obj_col, "mean objective")] if obj_col else []
    has_served_feas = (
        served_col
        and served_col in df.columns
        and "feasibility_rate" in df.columns
    )

    return [
        {
            "id": "sample_efficiency",
            "title": "1. Sample efficiency",
            "nav": "Sample eff.",
            "subtitle": "Primary objective J vs training epochs with best-so-far curve (TERRAN / marl-eval style).",
            "charts": [
                *([{
                    "type": "series_ma",
                    "col": obj_col,
                    "title": "Objective J over epochs",
                    "desc": "Real EVRP cost (lower is better). Grey = per epoch, blue = moving average.",
                    "ylabel": "Objective J",
                    "wide": True,
                }] if obj_col else []),
                {
                    "type": "best_objective_so_far",
                    "title": "Best objective so far",
                    "desc": "Anytime / sample-efficiency curve: cumulative minimum of J across training.",
                    "wide": True,
                },
                *([{
                    "type": "cycle_lines",
                    "cols": cycle_obj,
                    "title": "Cycle-level mean objective",
                    "desc": f"Mean J over all {n_inst} training instances per cycle (one full pass).",
                    "ylabel": "Mean objective J",
                    "wide": True,
                }] if cycle_obj else []),
            ],
        },
        {
            "id": "solution_quality",
            "title": "2. Solution quality",
            "nav": "Quality",
            "subtitle": "Customer coverage and feasibility — core metrics in CB-DRL-style EVRP learning curves.",
            "charts": [
                *([{
                    "type": "sample_efficiency_quality",
                    "title": "Served % & feasibility (dual axis)",
                    "desc": "CB-DRL Fig. 3 style: served-customer ratio and feasibility rate should rise together.",
                    "wide": True,
                }] if has_served_feas else []),
                {
                    "type": "served_coverage",
                    "title": "Customers served (%)",
                    "desc": "Rolling mean of served-customer ratio (target 100%).",
                    "wide": True,
                },
                {
                    "type": "feasibility_ma",
                    "title": "Feasibility rate",
                    "desc": "Share of episodes with zero hard-constraint violations (target 1.0).",
                    "wide": True,
                },
                {
                    "type": "series_ma",
                    "col": "unserved_customers",
                    "title": "Unserved customers",
                    "desc": "Remaining unserved customers per episode (should approach 0).",
                    "ylabel": "Count",
                    "wide": True,
                },
            ],
        },
        {
            "id": "constraint_violations",
            "title": "3. Constraint violations",
            "nav": "Constraints",
            "subtitle": "EVRP hard constraints — all should trend to zero for feasible solutions.",
            "charts": [
                *([{
                    "type": "multi_series_ma",
                    "series": constraints,
                    "title": "Constraint violation magnitudes",
                    "desc": "Battery, time windows, capacity, depot return — standard EVRP feasibility diagnostics.",
                    "ylabel": "Violation",
                    "wide": True,
                }] if len(constraints) >= 2 else []),
            ],
        },
        {
            "id": "routing_operations",
            "title": "4. Routing & fleet",
            "nav": "Routing",
            "subtitle": "Operational metrics reported in TERRAN and related EVRP-DRL papers.",
            "charts": [
                {
                    "type": "series_ma",
                    "col": "avg_distance",
                    "title": "Total travel distance",
                    "desc": "Mean route distance per episode (primary routing cost component).",
                    "ylabel": "Distance",
                    "wide": True,
                },
                {
                    "type": "series_ma",
                    "col": "avg_used_evs",
                    "title": "Fleet size (used EVs)",
                    "desc": "Number of vehicles deployed per episode.",
                    "ylabel": "Count",
                },
                *([{
                    "type": "multi_series_ma",
                    "series": ops,
                    "title": "Charging & waiting behavior",
                    "desc": "Charging time, station visits, and related operational costs.",
                    "ylabel": "Value",
                    "wide": True,
                }] if len(ops) >= 2 else []),
            ],
        },
        {
            "id": "rl_optimization",
            "title": "5. RL optimization",
            "nav": "RL opt.",
            "subtitle": "Actor-critic learning signals — shaped vs raw return and loss stability.",
            "charts": [
                *([{
                    "type": "multi_series_ma",
                    "series": returns,
                    "title": "Raw vs shaped return",
                    "desc": "Shaped return drives updates; raw return tracks the true objective signal.",
                    "ylabel": "Return",
                    "wide": True,
                }] if len(returns) >= 2 else []),
                *([{
                    "type": "multi_series_ma",
                    "series": losses[:2],
                    "title": "Actor & critic loss",
                    "desc": "Standard MARL optimization health check.",
                    "ylabel": "Loss",
                    "wide": True,
                }] if len(losses) >= 2 else []),
                {
                    "type": "explained_variance",
                    "title": "Critic explained variance",
                    "desc": "Near zero means the value function is not learning.",
                    "wide": True,
                },
            ],
        },
        {
            "id": "reward_diagnosis",
            "title": "6. Reward shaping",
            "nav": "Reward",
            "subtitle": "Penalty decomposition — terminal unserved should shrink as coverage improves.",
            "charts": [
                *([{
                    "type": "multi_series_ma",
                    "series": decomp,
                    "title": "Reward / penalty decomposition",
                    "desc": "Served bonus vs terminal and step penalties.",
                    "ylabel": "Magnitude",
                    "wide": True,
                }] if decomp else []),
                {
                    "type": "series_ma",
                    "col": "terminal_penalty_coef",
                    "title": "Terminal penalty schedule",
                    "desc": "Ramp of terminal unserved penalty coefficient.",
                    "ylabel": "Coefficient",
                },
            ],
        },
        {
            "id": "action_behavior",
            "title": "7. Policy behavior",
            "nav": "Policy",
            "subtitle": "Action mix and premature route closures.",
            "charts": [
                *([{
                    "type": "multi_series_ma",
                    "series": actions,
                    "title": "Action type fractions",
                    "desc": "Customer vs station vs depot — customer fraction should dominate.",
                    "ylabel": "Fraction",
                    "wide": True,
                }] if len(actions) >= 2 else []),
                {
                    "type": "series_ma",
                    "col": "early_depot_returns",
                    "title": "Premature route closures",
                    "desc": "Route closes while customers remain unserved (route_close events; should decrease).",
                    "ylabel": "Count per episode",
                    "wide": True,
                },
            ],
        },
        {
            "id": "actor_critic_health",
            "title": "8. Actor-critic diagnostics",
            "nav": "AC health",
            "subtitle": "Gradient norms and advantage statistics.",
            "charts": [
                *([{
                    "type": "multi_series_ma",
                    "series": adv,
                    "title": "Advantage statistics",
                    "desc": "Mean and std of GAE advantages.",
                    "ylabel": "Advantage",
                }] if adv else []),
                *([{
                    "type": "multi_series_ma",
                    "series": grads,
                    "title": "Gradient & parameter update norms",
                    "desc": "Confirms weights are changing each update.",
                    "ylabel": "Norm",
                    "wide": True,
                }] if len(grads) >= 2 else []),
            ],
        },
        {
            "id": "per_instance",
            "title": "9. Per-instance learning",
            "nav": "Instances",
            "subtitle": f"Instance × cycle heatmaps ({n_inst} train instances).",
            "charts": [
                *([{
                    "type": "heatmap",
                    "col": obj_col,
                    "title": "Objective heatmap (instance × cycle)",
                    "desc": "Which instances improve and which stall.",
                    "wide": True,
                }] if obj_col else []),
                *([{
                    "type": "heatmap",
                    "col": served_col,
                    "title": "Served-ratio heatmap (instance × cycle)",
                    "desc": "Coverage per instance across training cycles.",
                    "wide": True,
                }] if served_col else []),
                {
                    "type": "heatmap",
                    "col": "feasibility_rate",
                    "title": "Feasibility heatmap (instance × cycle)",
                    "desc": "Feasibility per instance across cycles.",
                    "wide": True,
                },
            ],
        },
    ]


def get_training_chart_sections(df: pd.DataFrame, plan: dict) -> list[dict[str, Any]]:
    """Pick live dashboard sections for the active training mode."""
    if plan.get("training_mode") == "flat":
        return build_flat_training_sections(df, plan)
    return TRAINING_CHART_SECTIONS


def class_contiguous_regions(df: pd.DataFrame) -> list[tuple[float, float, str]]:
    """Return (x0, x1, class) for each contiguous class block in iteration order."""
    if df.empty or "instance_class" not in df.columns:
        return []
    x = _x_series(df).to_numpy()
    classes = df["instance_class"].astype(str).str.upper().to_numpy()
    regions: list[tuple[float, float, str]] = []
    start = 0
    for i in range(1, len(classes) + 1):
        if i == len(classes) or classes[i] != classes[start]:
            regions.append((float(x[start]) - 0.5, float(x[i - 1]) + 0.5, classes[start]))
            start = i
    return regions


def add_class_regions(fig: go.Figure, df: pd.DataFrame, *, max_x: float | None = None) -> None:
    """Shade contiguous C / R / RC blocks with fixed class colors."""
    mx = max_x if max_x is not None else float(_x_series(df).max())
    for x0, x1, cls in class_contiguous_regions(df):
        if x0 > mx + 1:
            break
        color = CLASS_COLORS.get(cls, "#e2e8f0")
        fig.add_vrect(
            x0=x0,
            x1=min(x1, mx + 0.5),
            fillcolor=color,
            opacity=0.07,
            layer="below",
            line_width=0,
        )


def _curriculum_stage_col(df: pd.DataFrame) -> str | None:
    if "curriculum_stage" in df.columns and df["curriculum_stage"].notna().any():
        return "curriculum_stage"
    if "phase_name" in df.columns and df["phase_name"].notna().any():
        return "phase_name"
    return None


def curriculum_contiguous_regions(df: pd.DataFrame) -> list[tuple[float, float, str, int]]:
    """Return (x0, x1, stage_name, stage_index) for each curriculum block."""
    col = _curriculum_stage_col(df)
    if col is None or df.empty:
        return []
    x = _x_series(df).to_numpy()
    stages = df[col].astype(str).to_numpy()
    regions: list[tuple[float, float, str, int]] = []
    start = 0
    stage_idx = 1
    for i in range(1, len(stages) + 1):
        if i == len(stages) or stages[i] != stages[start]:
            regions.append((float(x[start]) - 0.5, float(x[i - 1]) + 0.5, stages[start], stage_idx))
            start = i
            stage_idx += 1
    return regions


def add_curriculum_stage_regions(
    fig: go.Figure,
    df: pd.DataFrame,
    *,
    max_x: float | None = None,
    show_legend: bool = True,
) -> None:
    """Shade curriculum stages with distinct colors and boundary lines."""
    mx = max_x if max_x is not None else float(_x_series(df).max())
    seen_legend: set[str] = set()
    for x0, x1, stage_name, stage_idx in curriculum_contiguous_regions(df):
        if x0 > mx + 1:
            break
        color = curriculum_stage_color(stage_name, stage_idx)
        fig.add_vrect(
            x0=x0,
            x1=min(x1, mx + 0.5),
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
        )
        if x0 > 0.5:
            fig.add_vline(x=x0 + 0.5, line_dash="dash", line_color=color, line_width=1.5, opacity=0.65)
        if show_legend and stage_name not in seen_legend:
            seen_legend.add(stage_name)
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    name=stage_name.replace("_", " "),
                    marker=dict(size=10, color=color, symbol="square"),
                )
            )


def add_chart_regions(fig: go.Figure, df: pd.DataFrame, *, max_x: float | None = None) -> None:
    """Prefer curriculum-stage shading when logged; else class C/R/RC bands."""
    if _curriculum_stage_col(df) is not None:
        add_curriculum_stage_regions(fig, df, max_x=max_x)
    else:
        add_class_regions(fig, df, max_x=max_x)


def _stage_marker_colors(df: pd.DataFrame) -> list[str]:
    col = _curriculum_stage_col(df)
    if col is None:
        return _marker_colors(df)
    colors: list[str] = []
    stage_ids: dict[str, int] = {}
    next_id = 1
    for name in df[col].astype(str):
        if name not in stage_ids:
            stage_ids[name] = next_id
            next_id += 1
        colors.append(curriculum_stage_color(name, stage_ids[name]))
    return colors


def _add_class_markers(fig: go.Figure, df: pd.DataFrame, col: str) -> None:
    """Faint per-step markers colored by instance class (no connecting line)."""
    if df.empty or col not in df.columns:
        return
    fig.add_trace(
        go.Scatter(
            x=_x_series(df),
            y=df[col],
            mode="markers",
            name="Per step",
            marker=dict(size=5, color=_stage_marker_colors(df), opacity=0.35),
            hovertemplate="Step %{x}<br>%{y:.4g}<extra></extra>",
            showlegend=True,
        )
    )


def fig_class_time_series(
    df: pd.DataFrame,
    col: str,
    *,
    title: str,
    y_label: str,
    y_range: tuple[float, float] | None = None,
    log_y: bool = False,
    show_markers: bool = True,
) -> go.Figure:
    """Class-colored markers + one continuous overall rolling-mean line."""
    fig = go.Figure()
    if df.empty or col not in df.columns:
        fig.update_layout(**_chart_layout())
        fig.add_annotation(text="No data", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    if show_markers:
        _add_class_markers(fig, df, col)

    roll_col = _overall_rolling_col(col)
    if roll_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=_x_series(df),
                y=df[roll_col],
                mode="lines",
                name="Rolling mean",
                line=dict(color="#2563eb", width=2.5),
            )
        )

    add_chart_regions(fig, df)
    layout = _chart_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title=y_label,
    )
    if y_range:
        layout["yaxis"] = dict(range=list(y_range))
    if log_y:
        layout["yaxis"] = dict(type="log")
    fig.update_layout(**layout)
    return fig


def fig_metric_with_rolling_band(
    df: pd.DataFrame,
    col: str,
    mean_col: str,
    std_col: str,
    plan: dict,
    *,
    title: str,
    y_label: str,
    y_range: tuple[float, float] | None = None,
    classic: bool = False,
) -> go.Figure:
    """Rolling-mean chart. ``classic=True`` restores the original reward style (grey raw + blue band)."""
    del plan
    if df.empty or col not in df.columns:
        fig = go.Figure()
        fig.update_layout(**_chart_layout())
        return fig

    x = _x_series(df)
    fig = go.Figure()

    if classic:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="lines+markers",
                name="Per step",
                line=dict(color="#94a3b8", width=1),
                marker=dict(size=3, color="#94a3b8"),
                opacity=0.45,
            )
        )
        if mean_col in df.columns:
            mean = df[mean_col]
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=mean,
                    mode="lines",
                    name="Rolling mean",
                    line=dict(color="#2563eb", width=2),
                )
            )
            if std_col and std_col in df.columns:
                std = df[std_col].fillna(0.0)
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean + std,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=mean - std,
                        mode="lines",
                        fill="tonexty",
                        fillcolor="rgba(37,99,235,0.15)",
                        line=dict(width=0),
                        name="±1 std",
                    )
                )
    else:
        _add_class_markers(fig, df, col)
        roll = mean_col if mean_col in df.columns else _overall_rolling_col(col)
        if roll in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[roll],
                    mode="lines",
                    name="Rolling mean",
                    line=dict(color="#2563eb", width=2.5),
                )
            )

    add_chart_regions(fig, df)
    layout = _chart_layout(title=title, xaxis_title="Iteration", yaxis_title=y_label)
    if y_range:
        layout["yaxis"] = dict(range=list(y_range))
    fig.update_layout(**layout)
    return fig


def fig_per_class_bars(
    df: pd.DataFrame,
    metrics: list[tuple[str, str, str | None]],
    *,
    title: str,
) -> go.Figure:
    """Grouped bar chart: one group per class (C/R/RC), bars for each metric.

    metrics: list of (column, label, format) where format is 'pct' or None
    """
    fig = go.Figure()
    if df.empty:
        fig.update_layout(**_chart_layout())
        return fig

    classes = [c for c in CLASS_ORDER if (df["instance_class"] == c).any()] if "instance_class" in df.columns else []
    if not classes:
        fig.update_layout(**_chart_layout())
        fig.add_annotation(text="No class labels in log", showarrow=False, x=0.5, y=0.5, xref="paper", yref="paper")
        return fig

    w = max(5, min(15, len(df) // max(len(classes), 1)))
    x_groups = classes
    n_metrics = len(metrics)
    bar_width = 0.8 / max(n_metrics, 1)

    for mi, (col, label, fmt) in enumerate(metrics):
        if col not in df.columns:
            continue
        vals = []
        for cls in classes:
            g = df[df["instance_class"] == cls].tail(w)
            v = float(g[col].mean()) if not g.empty else 0.0
            if fmt == "pct":
                v *= 100.0
            vals.append(v)
        offset = (mi - (n_metrics - 1) / 2) * bar_width
        fig.add_trace(
            go.Bar(
                x=x_groups,
                y=vals,
                name=label,
                marker_color=[CLASS_COLORS.get(c, "#64748b") for c in classes],
                marker_line=dict(color="#fff", width=0.5),
                offsetgroup=str(mi),
                width=bar_width * 0.9,
                text=[f"{v:.1f}{'%' if fmt == 'pct' else ''}" for v in vals],
                textposition="outside",
                textfont=dict(size=10),
            )
        )

    fig.update_layout(
        **_chart_layout(
            title=title,
            xaxis_title="Instance class",
            yaxis_title="Value",
            barmode="group",
            height=360,
        )
    )
    return fig


def fig_single_metric_bar(
    df: pd.DataFrame,
    col: str,
    *,
    title: str,
    y_label: str,
    as_pct: bool = False,
) -> go.Figure:
    """Simple 3-bar chart (C / R / RC) for one metric — e.g. used vehicles."""
    fmt = "pct" if as_pct else None
    label = y_label
    return fig_per_class_bars(df, [(col, label, fmt)], title=title)


def fig_cycle_bars(df: pd.DataFrame, col: str, *, title: str, y_label: str) -> go.Figure:
    """Bar chart: mean metric per training cycle."""
    fig = go.Figure()
    if df.empty or col not in df.columns or "cycle" not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig
    cycles = sorted(df["cycle"].astype(int).unique())
    vals = [float(df[df["cycle"] == c][col].mean()) for c in cycles]
    fig.add_trace(
        go.Bar(
            x=[f"Cycle {c}" for c in cycles],
            y=vals,
            marker_color="#2563eb",
            text=[f"{v:.2g}" for v in vals],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_chart_layout(title=title, xaxis_title="Cycle", yaxis_title=y_label, height=320)
    )
    return fig


def fig_cost_term(df: pd.DataFrame, spec: dict) -> go.Figure:
    """Dedicated chart for one weighted cost-function term (+ optional raw quantity)."""
    col = spec.get("col", "")
    raw_col = spec.get("raw_col")
    y_label = spec.get("ylabel", "Weighted")
    raw_ylabel = spec.get("raw_ylabel", "Raw")

    use_secondary = bool(raw_col and raw_col in df.columns)
    right_margin = 56 if use_secondary else 20
    fig = make_subplots(specs=[[{"secondary_y": use_secondary}]])
    if df.empty or col not in df.columns:
        fig.update_layout(**_chart_layout_embedded(margin=dict(t=8, b=40, l=52, r=right_margin)))
        fig.add_annotation(
            text="No data yet",
            showarrow=False,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            font=dict(color="#94a3b8", size=12),
        )
        return fig

    x = _x_series(df)
    roll_col = _overall_rolling_col(col)
    has_signal = df[col].abs().to_numpy().sum() > 1e-9

    if has_signal:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="markers",
                name="Per step",
                marker=dict(size=4, color=_stage_marker_colors(df), opacity=0.3),
            ),
            secondary_y=False,
        )
    if roll_col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[roll_col],
                mode="lines",
                name="Weighted",
                line=dict(color="#2563eb", width=2.5),
            ),
            secondary_y=False,
        )
    elif col in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[col],
                mode="lines",
                name="Weighted",
                line=dict(color="#16a34a", width=2),
            ),
            secondary_y=False,
        )

    if use_secondary:
        raw_roll = _overall_rolling_col(str(raw_col))
        raw_y = df[raw_roll] if raw_roll in df.columns else df[raw_col]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=raw_y,
                mode="lines",
                name="Raw",
                line=dict(color="#64748b", width=1.5, dash="dot"),
            ),
            secondary_y=True,
        )

    add_chart_regions(fig, df)
    fig.update_layout(
        **_chart_layout_embedded(
            xaxis_title="Iteration",
            height=260,
            margin=dict(t=8, b=58, l=52, r=right_margin),
        )
    )
    fig.update_yaxes(title_text=y_label, secondary_y=False)
    if use_secondary:
        fig.update_yaxes(title_text=raw_ylabel, secondary_y=True, showgrid=False)
    return fig


def _cost_term_specs(terms: list[dict], *, wide: bool = True) -> list[dict]:
    specs: list[dict] = []
    for term in terms:
        specs.append({
            "type": "cost_term",
            "col": term["col"],
            "raw_col": term.get("raw_col"),
            "title": str(term["short"]),
            "ylabel": "Weighted",
            "raw_ylabel": str(term.get("raw_ylabel") or "Raw"),
            "wide": wide,
            "term_key": term["key"],
        })
    return specs


def _terms_by_keys(terms: list[dict], keys: list[str]) -> list[dict]:
    by_key = {str(t["key"]): t for t in terms}
    return [by_key[k] for k in keys if k in by_key]


def _section_from_cost_terms(
    section_id: str,
    title: str,
    subtitle: str,
    term_keys: list[str],
    terms_list: list[dict],
    *,
    wide: bool = True,
) -> dict[str, Any]:
    return {
        "id": section_id,
        "title": title,
        "subtitle": subtitle,
        "charts": _cost_term_specs(_terms_by_keys(terms_list, term_keys), wide=wide),
    }


def _build_cost_function_sections() -> list[dict[str, Any]]:
    """One dashboard card (div) per cost-function theme — not one giant objective block."""
    return [
        {
            "id": "cost_total",
            "title": "Total objective J(π)",
            "nav": "Total J",
            "subtitle": "R(π) = −J / reward_scale during training. Lower cost is better.",
            "charts": [
                {
                    "type": "rolling_band",
                    "col": "cost",
                    "mean_col": "rolling_cost_mean",
                    "std_col": "",
                    "title": "Total cost J",
                    "desc": "Sum of all weighted objective terms + penalties",
                    "ylabel": "J",
                    "wide": True,
                },
                {
                    "type": "rolling_band",
                    "col": "raw_episode_return",
                    "mean_col": "rolling_raw_return_mean",
                    "std_col": "rolling_raw_return_std",
                    "title": "Training reward (scaled)",
                    "desc": "R = −J / reward_scale",
                    "ylabel": "Reward",
                    "classic": True,
                    "wide": True,
                },
            ],
        },
        _section_from_cost_terms(
            "cost_fleet_routing",
            "Fleet & routing",
            "λ-weighted fleet size, travel distance, and route duration.",
            ["vehicle", "distance", "duration"],
            COST_OBJECTIVE_TERMS,
            wide=True,
        ),
        _section_from_cost_terms(
            "cost_charging",
            "Charging",
            "Charging time and station energy price — separate from route distance.",
            ["charging", "charging_price"],
            COST_OBJECTIVE_TERMS,
            wide=True,
        ),
        _section_from_cost_terms(
            "cost_waiting",
            "Waiting time",
            "Customer waiting due to early arrival at time windows.",
            ["waiting"],
            COST_OBJECTIVE_TERMS,
            wide=True,
        ),
        _section_from_cost_terms(
            "pen_service",
            "Service penalties",
            "Unserved customers, duplicate visits, and global infeasibility — target zero.",
            ["unserved", "duplicate", "infeasible"],
            COST_PENALTY_TERMS,
            wide=False,
        ),
        _section_from_cost_terms(
            "pen_constraints",
            "Constraint penalties",
            "Time windows, battery, capacity, depot return, and reservations — target zero.",
            ["late", "battery", "capacity", "depot", "reserved"],
            COST_PENALTY_TERMS,
            wide=False,
        ),
    ]


def fig_reward_components(df: pd.DataFrame, plan: dict) -> go.Figure:
    del plan
    penalty_keys = [c for c in COST_COLUMNS if c.startswith("mean_pen_")]
    objective_keys = [c for c in COST_COLUMNS if c.startswith("mean_obj_")]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    if df.empty:
        fig.update_layout(**_chart_layout())
        return fig

    x = _x_series(df)
    labels = {
        c: c.replace("mean_pen_", "").replace("mean_obj_", "").replace("_", " ").title()
        for c in COST_COLUMNS
    }

    for i, c in enumerate(penalty_keys):
        if c in df.columns and df[c].abs().to_numpy().sum() > 1e-9:
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[c],
                    mode="lines",
                    name=f"Pen: {labels.get(c, c)}",
                    stackgroup="penalty",
                    line=dict(width=0.5, color=COMPONENT_COLORS[i % len(COMPONENT_COLORS)]),
                    fillcolor=COMPONENT_COLORS[i % len(COMPONENT_COLORS)],
                ),
                secondary_y=False,
            )

    obj_present = [c for c in objective_keys if c in df.columns and df[c].abs().to_numpy().sum() > 1e-9]
    if obj_present:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[obj_present].sum(axis=1),
                mode="lines",
                name="Objective terms",
                line=dict(color="#16a34a", width=1.5),
            ),
            secondary_y=True,
        )
    if "cost" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df["cost"],
                mode="lines",
                name="Total J",
                line=dict(color="#0f172a", width=2, dash="dot"),
            ),
            secondary_y=True,
        )

    add_chart_regions(fig, df)
    fig.update_layout(**_chart_layout(height=400, xaxis_title="Iteration"))
    fig.update_yaxes(title_text="Penalties", secondary_y=False)
    fig.update_yaxes(title_text="Objective / J", secondary_y=True, showgrid=False)
    return fig


def fig_served_coverage_pct(df: pd.DataFrame, plan: dict, *, title: str) -> go.Figure:
    del plan
    fig = go.Figure()
    if df.empty or "served_ratio" not in df.columns:
        fig.update_layout(**_chart_layout())
        return fig

    pct = df["served_ratio"].astype(float) * 100.0
    fig.add_trace(
        go.Scatter(
            x=_x_series(df),
            y=pct,
            mode="markers",
            name="Per step",
            marker=dict(size=6, color=_stage_marker_colors(df), opacity=0.4),
        )
    )
    if "rolling_served_ratio_mean" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=_x_series(df),
                y=df["rolling_served_ratio_mean"].astype(float) * 100.0,
                mode="lines",
                name="Rolling mean",
                line=dict(color="#2563eb", width=2.5),
            )
        )

    add_chart_regions(fig, df)
    fig.add_hline(y=100.0, line_dash="dot", line_color="#cbd5e1")
    fig.update_layout(
        **_chart_layout(
            title=title,
            xaxis_title="Iteration",
            yaxis_title="Served (%)",
            yaxis=dict(range=[0, 105], ticksuffix="%"),
            height=380,
        )
    )
    return fig


def fig_cost_breakdown(df: pd.DataFrame, plan: dict) -> go.Figure:
    del plan
    cols = [c for c in COST_COLUMNS if c in df.columns]
    fig = go.Figure()
    if df.empty or not cols:
        fig.update_layout(**_chart_layout())
        return fig
    labels = {
        "mean_obj_vehicle": "Fleet",
        "mean_obj_distance": "Distance",
        "mean_obj_duration": "Duration",
        "mean_obj_charging": "Charging time",
        "mean_obj_charging_price": "Charging price",
        "mean_obj_waiting": "Waiting",
        "mean_pen_unserved": "Unserved",
        "mean_pen_late": "Lateness",
        "mean_pen_battery": "Battery",
        "mean_pen_capacity": "Capacity",
        "mean_pen_depot": "Depot",
        "mean_pen_reserved": "Reserved",
        "mean_pen_duplicate": "Duplicate",
        "mean_pen_infeasible": "Infeasible",
    }
    x = _x_series(df)
    for i, c in enumerate(cols):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=df[c],
                mode="lines",
                name=labels.get(c, c),
                line=dict(color=COMPONENT_COLORS[i % len(COMPONENT_COLORS)], width=1.5),
            )
        )
    add_chart_regions(fig, df)
    fig.update_layout(
        **_chart_layout(xaxis_title="Iteration", yaxis_title="Weighted cost", height=360)
    )
    return fig


def fig_optimization_panel(df: pd.DataFrame, plan: dict) -> go.Figure:
    """2×3 grid of optimization metrics with continuous rolling-mean lines."""
    del plan
    specs = [
        ("loss", "Loss"),
        ("policy_loss", "Policy loss"),
        ("advantage", "Advantage"),
        ("entropy", "Entropy"),
        ("grad_norm", "Grad norm"),
        ("learning_rate", "Learning rate"),
    ]
    cols = [(c, t) for c, t in specs if c in df.columns]
    if not cols:
        return go.Figure()

    rows = 2
    cols_n = 3
    fig = make_subplots(
        rows=rows,
        cols=cols_n,
        subplot_titles=[t for _, t in cols],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )
    for i, (col, _) in enumerate(cols):
        r, c_idx = divmod(i, cols_n)
        roll_col = _overall_rolling_col(col)
        y = df[roll_col] if roll_col in df.columns else df[col]
        fig.add_trace(
            go.Scatter(
                x=_x_series(df),
                y=y,
                mode="lines",
                name=col,
                showlegend=False,
                line=dict(color="#2563eb", width=2),
            ),
            row=r + 1,
            col=c_idx + 1,
        )
        fig.update_yaxes(title_text="", row=r + 1, col=c_idx + 1)
        fig.update_xaxes(title_text="Iter" if r == rows - 1 else "", row=r + 1, col=c_idx + 1)

    fig.update_layout(
        template="plotly_white",
        height=480,
        autosize=True,
        showlegend=True,
        legend=dict(orientation="h", y=1.06, x=0),
        margin=dict(t=60, b=36, l=40, r=16),
    )
    return fig


def build_chart_figure(df: pd.DataFrame, plan: dict, spec: dict[str, Any]) -> go.Figure | None:
    t = spec.get("type", "metric")
    if t == "series_ma":
        col = spec.get("col", "")
        if not col or col not in df.columns:
            return None
        return fig_series_ma(
            df, col, plan,
            title=spec.get("title", col),
            ylabel=spec.get("ylabel", col),
            y_range=spec.get("y_range"),
            hline=spec.get("hline"),
        )
    if t == "explained_variance":
        return fig_explained_variance(df, plan, title=spec.get("title", "Critic explained variance"))
    if t == "feasibility_ma":
        return fig_feasibility_ma(df, plan, title=spec.get("title", "Feasibility"))
    if t == "best_objective_so_far":
        return fig_best_objective_so_far(df, plan, title=spec.get("title", "Best objective so far"))
    if t == "sample_efficiency_quality":
        return fig_sample_efficiency_quality(df, plan, title=spec.get("title", "Served & feasibility"))
    if t == "multi_series_ma":
        series = spec.get("series") or []
        if not series:
            return None
        return fig_multi_series_ma(
            df, series, plan,
            title=spec.get("title", ""),
            ylabel=spec.get("ylabel", ""),
        )
    if t == "cycle_lines":
        cols = spec.get("cols") or []
        if not cols:
            return None
        return fig_cycle_multi_lines(
            df, cols, plan,
            title=spec.get("title", ""),
            ylabel=spec.get("ylabel", ""),
        )
    if t == "heatmap":
        col = spec.get("col", "")
        if not col or col not in df.columns:
            return None
        return fig_heatmap_instance_cycle(df, col, plan, title=spec.get("title", col))
    if t == "final_cycle_bar":
        col = spec.get("col", "")
        if not col or col not in df.columns:
            return None
        return fig_final_cycle_instance_bar(df, col, plan, title=spec.get("title", col))
    if t == "cost_term":
        return fig_cost_term(df, spec)
    if t == "cost_breakdown":
        return fig_cost_breakdown(df, plan)
    if t == "reward_components":
        return fig_reward_components(df, plan)
    if t == "served_coverage":
        return fig_served_coverage_pct(df, plan, title=spec.get("title", "Customers served (%)"))
    if t == "rolling_band":
        return fig_metric_with_rolling_band(
            df,
            spec.get("col", ""),
            spec.get("mean_col", ""),
            spec.get("std_col", ""),
            plan,
            title=spec.get("title", ""),
            y_label=spec.get("ylabel", ""),
            y_range=spec.get("y_range"),
            classic=bool(spec.get("classic", False)),
        )
    if t == "class_metric":
        return fig_class_time_series(
            df,
            spec.get("col", ""),
            title=spec.get("title", ""),
            y_label=spec.get("ylabel", ""),
            y_range=spec.get("y_range"),
            log_y=spec.get("log_y", False),
        )
    if t == "per_class_bars":
        return fig_per_class_bars(
            df,
            spec.get("metrics", []),
            title=spec.get("title", "Per-class summary"),
        )
    if t == "class_bar":
        return fig_single_metric_bar(
            df,
            spec.get("col", ""),
            title=spec.get("title", ""),
            y_label=spec.get("ylabel", ""),
            as_pct=spec.get("as_pct", False),
        )
    if t == "cycle_bars":
        return fig_cycle_bars(
            df,
            spec.get("col", ""),
            title=spec.get("title", ""),
            y_label=spec.get("ylabel", ""),
        )
    if t == "optimization_panel":
        return fig_optimization_panel(df, plan)
    col = spec.get("col")
    if not col or col not in df.columns:
        return None
    return fig_class_time_series(
        df,
        col,
        title=spec.get("title", col),
        y_label=spec.get("ylabel", col),
        y_range=spec.get("y_range"),
        log_y=spec.get("log_y", False),
    )


TRAINING_CHART_SECTIONS: list[dict[str, Any]] = [
    *_build_cost_function_sections(),
    {
        "id": "coverage",
        "title": "Solution quality",
        "subtitle": "Customer coverage and feasibility — independent of cost weights.",
        "charts": [
            {"type": "served_coverage", "title": "Customers served (%)", "wide": True},
            {
                "type": "class_metric",
                "col": "feasibility_rate",
                "title": "Feasibility rate",
                "desc": "Share of episodes with zero penalty terms",
                "ylabel": "Rate",
                "y_range": (0, 1.05),
                "wide": True,
            },
            {
                "type": "class_metric",
                "col": "served_ratio",
                "title": "Served ratio",
                "ylabel": "Ratio",
                "y_range": (0, 1.05),
            },
        ],
    },
    {
        "id": "fleet_ops",
        "title": "Fleet utilization",
        "subtitle": "Raw operational metrics — EV count, distance per customer, charging share.",
        "charts": [
            {
                "type": "class_metric",
                "col": "avg_used_evs",
                "title": "Used EVs",
                "ylabel": "Count",
                "wide": True,
            },
            {
                "type": "class_metric",
                "col": "distance_per_customer",
                "title": "Distance per customer",
                "ylabel": "Distance",
            },
            {
                "type": "class_metric",
                "col": "charging_time_share",
                "title": "Charging time share",
                "ylabel": "Share",
                "y_range": (0, 1.05),
            },
        ],
    },
    {
        "id": "per_class_reward",
        "title": "Per-class · reward & cost",
        "subtitle": "Trailing-window mean by instance class (C · R · RC).",
        "charts": [
            {
                "type": "per_class_bars",
                "title": "Reward & total J by class",
                "wide": True,
                "metrics": [
                    ("reward", "Reward", None),
                    ("cost", "Cost J", None),
                ],
            },
        ],
    },
    {
        "id": "per_class_coverage",
        "title": "Per-class · coverage & feasibility",
        "subtitle": "Trailing-window mean by instance class (C · R · RC).",
        "charts": [
            {
                "type": "per_class_bars",
                "title": "Coverage & feasibility by class",
                "wide": True,
                "metrics": [
                    ("served_ratio", "Served", "pct"),
                    ("feasibility_rate", "Feasible", "pct"),
                ],
            },
        ],
    },
    {
        "id": "optimization",
        "title": "Policy optimization",
        "subtitle": "Actor-critic diagnostics. Background bands = curriculum stage (colored) or class block.",
        "charts": [
            {"type": "optimization_panel", "title": "Optimization overview", "wide": True},
            {"type": "class_metric", "col": "loss", "title": "Total loss", "ylabel": "Loss", "wide": True},
            {"type": "class_metric", "col": "policy_loss", "title": "Policy loss", "ylabel": "Loss"},
            {"type": "class_metric", "col": "entropy", "title": "Entropy", "ylabel": "nats"},
            {"type": "class_metric", "col": "grad_norm", "title": "Gradient norm", "ylabel": "‖∇‖"},
        ],
    },
]


def section_by_id() -> dict[str, dict[str, Any]]:
    return {s["id"]: s for s in TRAINING_CHART_SECTIONS}


_WIDE_CHART_TYPES = frozenset({
    "cost_breakdown",
    "cost_term",
    "optimization_panel",
    "reward_components",
    "served_coverage",
    "rolling_band",
    "per_class_bars",
    "class_bar",
    "cycle_bars",
    "class_metric",
    "series_ma",
    "feasibility_ma",
    "multi_series_ma",
    "cycle_lines",
    "heatmap",
    "final_cycle_bar",
    "explained_variance",
})


def is_wide_chart(spec: dict[str, Any]) -> bool:
    if spec.get("wide"):
        return True
    return spec.get("type") in _WIDE_CHART_TYPES


TRAINING_CHARTS: list[dict[str, Any]] = [
    spec for section in TRAINING_CHART_SECTIONS for spec in section["charts"]
]
