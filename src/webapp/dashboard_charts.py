"""Chart specifications and Plotly figure builders for the training dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.webapp.plot_style import COLORS, SEQ, base_layout

C = COLORS


def _x(df: pd.DataFrame) -> list:
    if "epoch" in df.columns:
        return df["epoch"].tolist()
    return list(range(len(df)))


def _empty_fig(ylabel: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(**base_layout(ylabel=ylabel))
    fig.add_annotation(
        text="Waiting for training data…",
        showarrow=False,
        font=dict(size=13, color="#94a3b8"),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
    )
    return fig


def _line_fig(df: pd.DataFrame, col: str, ylabel: str, *, color: str | None = None) -> go.Figure:
    if df.empty or col not in df.columns:
        return _empty_fig(ylabel)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=_x(df),
            y=df[col],
            mode="lines",
            name=col,
            line=dict(color=color or C["blue"], width=2.2),
            connectgaps=True,
        )
    )
    fig.update_layout(**base_layout(ylabel=ylabel, showlegend=False))
    return fig


def _multi_line_fig(df: pd.DataFrame, series: list[tuple], ylabel: str) -> go.Figure:
    if df.empty:
        return _empty_fig(ylabel)
    fig = go.Figure()
    added = 0
    for spec in series:
        key = spec[0]
        if key not in df.columns:
            continue
        name = spec[1] if len(spec) > 1 else key
        color = spec[2] if len(spec) > 2 else C["blue"]
        dash = spec[3] if len(spec) > 3 else None
        fig.add_trace(
            go.Scatter(
                x=_x(df),
                y=df[key],
                mode="lines",
                name=name,
                line=dict(color=color, width=2.2, dash=dash),
                connectgaps=True,
            )
        )
        added += 1
    if added == 0:
        return _empty_fig(ylabel)
    fig.update_layout(**base_layout(ylabel=ylabel, showlegend=added > 1))
    return fig


def _area_fig(df: pd.DataFrame, series: list[tuple], ylabel: str) -> go.Figure:
    if df.empty:
        return _empty_fig(ylabel)
    fig = go.Figure()
    added = 0
    for spec in series:
        key = spec[0]
        if key not in df.columns:
            continue
        name = spec[1] if len(spec) > 1 else key
        color = spec[2] if len(spec) > 2 else SEQ[added % len(SEQ)]
        y = df[key].fillna(0.0).abs()
        fig.add_trace(
            go.Scatter(
                x=_x(df),
                y=y,
                name=name,
                mode="lines",
                stackgroup="one",
                line=dict(color=color, width=0.8),
                fillcolor=color,
            )
        )
        added += 1
    if added == 0:
        return _empty_fig(ylabel)
    fig.update_layout(**base_layout(ylabel=ylabel))
    return fig


def build_chart_figure(df: pd.DataFrame, _plan: dict, spec: dict) -> go.Figure | None:
    """Build a Plotly figure from a chart spec and metrics dataframe."""
    ctype = spec.get("type", "line")
    ylabel = spec.get("ylabel", "")

    if ctype == "line":
        col = spec.get("col")
        if not col:
            return None
        return _line_fig(df, col, ylabel, color=spec.get("color"))

    if ctype == "multi_line":
        series = spec.get("series") or []
        return _multi_line_fig(df, series, ylabel)

    if ctype == "area":
        series = spec.get("series") or []
        return _area_fig(df, series, ylabel)

    return None


def _chart(
    *,
    chart_id: str,
    title: str,
    desc: str,
    ctype: str,
    ylabel: str,
    col: str | None = None,
    color: str | None = None,
    series: list[tuple] | None = None,
) -> dict:
    spec = {
        "id": chart_id,
        "type": ctype,
        "title": title,
        "desc": desc,
        "ylabel": ylabel,
    }
    if col is not None:
        spec["col"] = col
    if color is not None:
        spec["color"] = color
    if series is not None:
        spec["series"] = series
    return spec


TRAINING_CHART_SECTIONS: list[dict] = [
    {
        "title": "Learning progress",
        "subtitle": "Reward signal, baseline and optimization dynamics across epochs.",
        "charts": [
            _chart(
                chart_id="reward_baseline",
                title="Reward, baseline and best-so-far",
                desc="Mean batch reward R = −J, the moving-average baseline, and the best reward reached so far.",
                ctype="multi_line",
                ylabel="reward",
                series=[
                    ("reward", "reward", C["blue"], None),
                    ("reward_smooth", "smoothed (EMA)", C["navy"], None),
                    ("baseline", "baseline", C["muted"], "dash"),
                    ("best_reward", "best so far", C["green"], "dot"),
                ],
            ),
            _chart(
                chart_id="reward_band",
                title="Reward spread across the batch",
                desc="Per-epoch minimum, mean and maximum reward in the batch — shows how consistent the sampled solutions are.",
                ctype="multi_line",
                ylabel="reward",
                series=[
                    ("reward_max", "max", C["green"], "dot"),
                    ("reward", "mean", C["blue"], None),
                    ("reward_min", "min", C["red"], "dot"),
                ],
            ),
            _chart(
                chart_id="advantage",
                title="Advantage (reward − baseline)",
                desc="The REINFORCE learning signal. It oscillates around zero as the baseline tracks the policy.",
                ctype="line",
                col="advantage",
                ylabel="advantage",
                color=C["teal"],
            ),
            _chart(
                chart_id="cost",
                title="Objective value J",
                desc="Mean total cost of sampled solutions — the quantity being minimized. Lower is better.",
                ctype="line",
                col="cost",
                ylabel="cost J",
                color=C["orange"],
            ),
            _chart(
                chart_id="loss",
                title="Training loss",
                desc="Total optimized loss and its policy-gradient component.",
                ctype="multi_line",
                ylabel="loss",
                series=[
                    ("loss", "total loss", C["navy"], None),
                    ("policy_loss", "policy loss", C["slate"], "dash"),
                ],
            ),
            _chart(
                chart_id="entropy",
                title="Policy entropy",
                desc="Entropy of the action distribution; a proxy for exploration.",
                ctype="line",
                col="entropy",
                ylabel="entropy (nats)",
                color=C["purple"],
            ),
            _chart(
                chart_id="grad_norm",
                title="Gradient norm",
                desc="L2 norm of gradients after clipping. Spikes can indicate optimization instability.",
                ctype="line",
                col="grad_norm",
                ylabel="‖∇‖",
                color=C["rose"],
            ),
            _chart(
                chart_id="reward_std",
                title="Reward variability (batch std)",
                desc="Standard deviation of reward across the batch.",
                ctype="line",
                col="reward_std",
                ylabel="std(reward)",
                color=C["gold"],
            ),
        ],
    },
    {
        "title": "Feasibility & fleet",
        "subtitle": "Constraint satisfaction and how many vehicles the policy uses.",
        "charts": [
            _chart(
                chart_id="feasibility_served",
                title="Feasibility rate and served-customer rate",
                desc="Fraction of fully-feasible solutions, and the fraction of customers actually served. Both target → 1.0.",
                ctype="multi_line",
                ylabel="fraction",
                series=[
                    ("feasibility_rate", "feasible fraction", C["green"], None),
                    ("served_rate", "served fraction", C["teal"], None),
                ],
            ),
            _chart(
                chart_id="avg_used_evs",
                title="Average vehicles used",
                desc="Mean number of EVs that leave the depot.",
                ctype="line",
                col="avg_used_evs",
                ylabel="EVs",
                color=C["blue"],
            ),
            _chart(
                chart_id="avg_unserved",
                title="Average unserved customers",
                desc="Mean number of customers left unserved per solution. Should converge to zero.",
                ctype="line",
                col="avg_unserved",
                ylabel="customers",
                color=C["red"],
            ),
            _chart(
                chart_id="avg_violation",
                title="Average constraint violation",
                desc="Combined magnitude of battery, capacity, lateness and unserved violations.",
                ctype="line",
                col="avg_violation",
                ylabel="violation magnitude",
                color=C["rose"],
            ),
        ],
    },
    {
        "title": "Route economics",
        "subtitle": "Distance, duration and the time spent charging and waiting.",
        "charts": [
            _chart(
                chart_id="avg_distance",
                title="Average total distance",
                desc="Mean total travelled distance across all vehicles in a solution.",
                ctype="line",
                col="avg_distance",
                ylabel="distance",
                color=C["orange"],
            ),
            _chart(
                chart_id="avg_duration",
                title="Average route duration",
                desc="Mean total operating time across the fleet.",
                ctype="line",
                col="avg_duration",
                ylabel="time",
                color=C["slate"],
            ),
            _chart(
                chart_id="time_split",
                title="Charging vs waiting time",
                desc="How operating time splits between recharging and idle waiting.",
                ctype="multi_line",
                ylabel="time",
                series=[
                    ("avg_charging_time", "charging", C["teal"], None),
                    ("avg_waiting_time", "waiting", C["gold"], None),
                ],
            ),
        ],
    },
    {
        "title": "Efficiency & stability",
        "subtitle": "Normalized route efficiency, time allocation and optimization stability.",
        "charts": [
            _chart(
                chart_id="best_cost",
                title="Best cost reached so far",
                desc="Cumulative minimum of the objective J — a monotone view of the best solution found during training.",
                ctype="line",
                col="cost_best",
                ylabel="best cost J",
                color=C["navy"],
            ),
            _chart(
                chart_id="distance_efficiency",
                title="Distance per EV and per customer",
                desc="Normalized routing efficiency: total distance divided by vehicles used and by customers served.",
                ctype="multi_line",
                ylabel="distance",
                series=[
                    ("distance_per_ev", "per EV", C["orange"], None),
                    ("distance_per_customer", "per customer", C["purple"], None),
                ],
            ),
            _chart(
                chart_id="time_shares",
                title="Time allocation shares",
                desc="Share of active operating time spent charging vs waiting (relative to distance + charging + waiting).",
                ctype="multi_line",
                ylabel="share",
                series=[
                    ("charging_time_share", "charging share", C["teal"], None),
                    ("waiting_time_share", "waiting share", C["gold"], None),
                ],
            ),
            _chart(
                chart_id="learning_rate",
                title="Learning rate",
                desc="Optimizer learning rate per epoch (useful when a scheduler is enabled).",
                ctype="line",
                col="learning_rate",
                ylabel="lr",
                color=C["slate"],
            ),
        ],
    },
    {
        "title": "Objective composition",
        "subtitle": "Weighted objective and penalty term decomposition over training.",
        "charts": [
            _chart(
                chart_id="obj_components",
                title="Objective term composition",
                desc="Weighted contributions to J from fleet, distance, duration, charging and waiting.",
                ctype="area",
                ylabel="weighted cost",
                series=[
                    ("mean_obj_vehicle", "vehicles", SEQ[0]),
                    ("mean_obj_distance", "distance", SEQ[1]),
                    ("mean_obj_duration", "duration", SEQ[2]),
                    ("mean_obj_charging", "charging", SEQ[3]),
                    ("mean_obj_waiting", "waiting", SEQ[4]),
                ],
            ),
            _chart(
                chart_id="pen_components",
                title="Penalty composition",
                desc="Weighted penalty contributions for soft constraints. Should collapse toward zero.",
                ctype="area",
                ylabel="weighted penalty",
                series=[
                    ("mean_pen_unserved", "unserved", SEQ[0]),
                    ("mean_pen_late", "lateness", SEQ[1]),
                    ("mean_pen_battery", "battery", SEQ[2]),
                    ("mean_pen_capacity", "capacity", SEQ[3]),
                    ("mean_pen_depot", "depot", SEQ[4]),
                    ("mean_pen_reserved", "reserved", SEQ[5]),
                    ("mean_pen_duplicate", "duplicate", SEQ[6]),
                    ("mean_pen_infeasible", "infeasible", SEQ[7]),
                ],
            ),
        ],
    },
]
