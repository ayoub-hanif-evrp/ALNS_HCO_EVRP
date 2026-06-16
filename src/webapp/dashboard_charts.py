"""Plotly chart builders for training-progress dashboard."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.webapp.dashboard_data import COST_COLUMNS, COST_OBJECTIVE_TERMS, COST_PENALTY_TERMS
from src.webapp.plot_style import (
    CLASS_COLORS,
    CLASS_ORDER,
    COMPONENT_COLORS,
    OVERALL_LINE_COLOR,
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
            marker=dict(size=5, color=_marker_colors(df), opacity=0.35),
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

    add_class_regions(fig, df)
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


def _overall_rolling_col(col: str) -> str:
    """Prepared global rolling-mean column for a metric."""
    aliases = {
        "raw_episode_return": "rolling_raw_return_mean",
        "reward": "rolling_raw_return_mean",
    }
    if col in aliases:
        return aliases[col]
    return f"rolling_{col}_mean"


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

    add_class_regions(fig, df)
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
                marker=dict(size=4, color=_marker_colors(df), opacity=0.3),
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

    add_class_regions(fig, df)
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
            "Charging time and Hanif station energy price — separate from route distance.",
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

    add_class_regions(fig, df)
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
            marker=dict(size=6, color=_marker_colors(df), opacity=0.4),
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

    add_class_regions(fig, df)
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
    add_class_regions(fig, df)
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
        "subtitle": "REINFORCE diagnostics (not part of J). Background bands = curriculum class blocks.",
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
})


def is_wide_chart(spec: dict[str, Any]) -> bool:
    if spec.get("wide"):
        return True
    return spec.get("type") in _WIDE_CHART_TYPES


TRAINING_CHARTS: list[dict[str, Any]] = [
    spec for section in TRAINING_CHART_SECTIONS for spec in section["charts"]
]
