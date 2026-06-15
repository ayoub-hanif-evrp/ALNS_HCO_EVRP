"""Plotting helpers for training curves, evaluation, and route visualization."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend for batch figure generation
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


def _ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# (key, human title, output filename) for every per-epoch training curve.
_TRAINING_SERIES: list[tuple[str, str, str]] = [
    ("reward", "Reward", "reward_curve.png"),
    ("best_reward", "Best reward so far", "best_reward_curve.png"),
    ("cost", "Cost J", "cost_curve.png"),
    ("advantage", "Advantage (reward - baseline)", "advantage_curve.png"),
    ("loss", "Training loss", "loss_curve.png"),
    ("policy_loss", "Policy loss", "policy_loss_curve.png"),
    ("entropy", "Policy entropy", "entropy_curve.png"),
    ("grad_norm", "Gradient norm", "grad_norm_curve.png"),
    ("learning_rate", "Learning rate", "learning_rate_curve.png"),
    ("reward_std", "Reward std (batch)", "reward_std_curve.png"),
    ("feasibility_rate", "Feasibility rate", "feasibility_curve.png"),
    ("served_rate", "Served-customer rate", "served_rate_curve.png"),
    ("avg_used_evs", "Average used EVs", "used_evs_curve.png"),
    ("avg_unserved", "Average unserved customers", "unserved_curve.png"),
    ("avg_violation", "Average constraint violation", "violation_curve.png"),
    ("avg_distance", "Average distance", "distance_curve.png"),
    ("avg_duration", "Average route duration", "duration_curve.png"),
    ("avg_charging_time", "Average charging time", "charging_curve.png"),
    ("avg_waiting_time", "Average waiting time", "waiting_curve.png"),
    ("distance_per_ev", "Distance per EV", "distance_per_ev_curve.png"),
    ("distance_per_customer", "Distance per customer", "distance_per_customer_curve.png"),
    ("charging_time_share", "Charging time share", "charging_share_curve.png"),
    ("waiting_time_share", "Waiting time share", "waiting_share_curve.png"),
]


def _smooth(values: list[float], window: int = 5) -> list[float]:
    """Trailing moving average that tolerates NaN / None entries."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=float)
    out = []
    for i in range(len(arr)):
        lo = max(0, i - window + 1)
        seg = arr[lo : i + 1]
        seg = seg[~np.isnan(seg)]
        out.append(float(seg.mean()) if seg.size else np.nan)
    return out


def plot_training_curves(log_rows: list[dict], out_dir: str | Path) -> list[Path]:
    """Plot every available per-epoch training metric (raw + smoothed overlay)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not log_rows:
        return []
    epochs = [r.get("epoch", i) for i, r in enumerate(log_rows)]
    saved = []
    for key, title, fname in _TRAINING_SERIES:
        if key not in log_rows[0]:
            continue
        values = [r.get(key, np.nan) for r in log_rows]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, values, color="#94a3b8", lw=1, alpha=0.7, label="per-epoch")
        if len(values) >= 5:
            ax.plot(epochs, _smooth(values), color="#2563eb", lw=2, label="smoothed (w=5)")
            ax.legend(fontsize=8)
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Training: {title}")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_training_overview(log_rows: list[dict], out_path: str | Path) -> Path | None:
    """Single multi-panel figure summarizing the most important training signals."""
    if not log_rows:
        return None
    out_path = _ensure_dir(out_path)
    epochs = [r.get("epoch", i) for i, r in enumerate(log_rows)]
    panels = [
        ("reward", "Reward", "#2563eb"),
        ("cost", "Cost J", "#d97706"),
        ("feasibility_rate", "Feasibility rate", "#16a34a"),
        ("served_rate", "Served rate", "#0f766e"),
        ("avg_used_evs", "Used EVs", "#7c3aed"),
        ("avg_distance", "Distance", "#dc2626"),
        ("avg_violation", "Violation", "#e11d48"),
        ("loss", "Loss", "#1e293b"),
        ("entropy", "Entropy", "#0891b2"),
    ]
    panels = [p for p in panels if p[0] in log_rows[0]]
    ncol = 3
    nrow = (len(panels) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (key, title, color) in zip(axes, panels):
        vals = [r.get(key, np.nan) for r in log_rows]
        ax.plot(epochs, vals, color="#cbd5e1", lw=0.9)
        if len(vals) >= 5:
            ax.plot(epochs, _smooth(vals), color=color, lw=2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch", fontsize=8)
        ax.tick_params(labelsize=8)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("E-VRPTW training overview", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_stacked_composition(
    log_rows: list[dict], prefix: str, title: str, out_path: str | Path
) -> Path | None:
    """Stacked-area decomposition of objective (``mean_obj_*``) or penalty (``mean_pen_*``) terms."""
    if not log_rows:
        return None
    keys = [k for k in log_rows[0] if k.startswith(prefix)]
    if not keys:
        return None
    out_path = _ensure_dir(out_path)
    epochs = [r.get("epoch", i) for i, r in enumerate(log_rows)]
    stacks = [np.abs([r.get(k, 0.0) or 0.0 for r in log_rows]) for k in keys]
    labels = [k[len(prefix):] for k in keys]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.stackplot(epochs, *stacks, labels=labels, alpha=0.85)
    ax.set_xlabel("epoch")
    ax.set_ylabel("weighted contribution")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_validation_curves(val_rows: list[dict], out_dir: str | Path) -> list[Path]:
    """Per-class validation curves: score, cost and feasibility for C / R / RC.

    Each ``val_rows`` record is a flat dict with keys like ``C_cost``,
    ``R_feasibility_rate``, ``overall_score`` and a ``global_epoch`` x value.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not val_rows:
        return []
    x = [r.get("global_epoch", i) for i, r in enumerate(val_rows)]
    classes = ["C", "R", "RC", "overall"]
    colors = {"C": "#2563eb", "R": "#d97706", "RC": "#7c3aed", "overall": "#111827"}
    # (metric suffix, axis title, filename, optional y-limits)
    metrics = [
        ("cost", "Validation cost J", "validation_cost_per_class.png", None),
        ("feasibility_rate", "Validation feasibility rate", "validation_feasibility_per_class.png", (0, 1.02)),
        ("score", "Validation score (cost + 1e4·(1−feas))", "validation_score_per_class.png", None),
        ("avg_used_evs", "Validation used EVs", "validation_used_evs_per_class.png", None),
    ]
    saved = []
    for suffix, title, fname, ylim in metrics:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plotted = False
        for cls in classes:
            key = f"{cls}_{suffix}"
            ys = [r.get(key, np.nan) for r in val_rows]
            if all(np.isnan(v) if isinstance(v, float) else v is None for v in ys):
                continue
            ax.plot(x, ys, marker="o", ms=3, color=colors[cls],
                    lw=2 if cls == "overall" else 1.5,
                    ls="-" if cls != "overall" else "--", label=cls)
            plotted = True
        if not plotted:
            plt.close(fig)
            continue
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Per-class {title}")
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, title="class")
        fig.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=120)
        plt.close(fig)
        saved.append(path)
    return saved


def plot_metric_by_instance(rows: list[dict], metric: str, out_path: str | Path, title: str | None = None) -> Path:
    """Bar chart of a metric across instances, grouped by method."""
    out_path = _ensure_dir(out_path)
    methods = sorted({r["method"] for r in rows})
    instances = sorted({r["instance_name"] for r in rows})
    fig, ax = plt.subplots(figsize=(max(8, len(instances) * 0.6), 5))
    width = 0.8 / max(len(methods), 1)
    x = np.arange(len(instances))
    for i, method in enumerate(methods):
        vals = []
        for inst in instances:
            match = [r for r in rows if r["method"] == method and r["instance_name"] == inst]
            vals.append(match[0].get(metric, np.nan) if match else np.nan)
        ax.bar(x + i * width, vals, width=width, label=method)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(instances, rotation=90)
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} by instance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_feasibility_by_method(rows: list[dict], out_path: str | Path) -> Path:
    out_path = _ensure_dir(out_path)
    methods = sorted({r["method"] for r in rows})
    rates = []
    for method in methods:
        m_rows = [r for r in rows if r["method"] == method]
        rate = np.mean([1.0 if r["feasible"] else 0.0 for r in m_rows]) if m_rows else 0.0
        rates.append(rate)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(methods, rates, color="seagreen")
    ax.set_ylim(0, 1)
    ax.set_ylabel("feasibility rate")
    ax.set_title("Feasibility rate by method")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_fleet_search_curve(candidates, selected_m: int, out_path: str | Path, instance_name: str) -> Path:
    """Objective vs candidate fleet size, marking the selected m."""
    out_path = _ensure_dir(out_path)
    ms = [c.m for c in candidates]
    objs = [c.objective for c in candidates]
    feas = [c.feasible for c in candidates]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ms, objs, marker="o", color="steelblue", label="objective")
    feas_ms = [m for m, f in zip(ms, feas) if f]
    feas_objs = [o for o, f in zip(objs, feas) if f]
    if feas_ms:
        ax.scatter(feas_ms, feas_objs, color="green", s=60, zorder=3, label="feasible")
    ax.axvline(selected_m, color="red", linestyle="--", label=f"selected m={selected_m}")
    ax.set_xlabel("candidate fleet size m")
    ax.set_ylabel("objective J")
    ax.set_title(f"Fleet-size search: {instance_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_routes(instance, env, out_path: str | Path) -> Path:
    """Visualize EV routes in the x-y plane (depot, customers, stations)."""
    out_path = _ensure_dir(out_path)
    fig, ax = plt.subplots(figsize=(8, 8))

    cust_x = [instance.nodes[c].x for c in instance.customer_indices]
    cust_y = [instance.nodes[c].y for c in instance.customer_indices]
    st_x = [instance.nodes[s].x for s in instance.charging_station_indices]
    st_y = [instance.nodes[s].y for s in instance.charging_station_indices]
    depot = instance.nodes[instance.depot_index]

    ax.scatter(cust_x, cust_y, c="steelblue", s=35, label="customers", zorder=2)
    ax.scatter(st_x, st_y, c="limegreen", s=80, marker="^", edgecolors="darkgreen", label="stations", zorder=3)
    ax.scatter([depot.x], [depot.y], c="red", s=220, marker="*", edgecolors="black", label="depot", zorder=4)

    cmap = plt.get_cmap("tab10")
    for ev in env.evs:
        if not ev.used():
            continue
        xs = [instance.nodes[n].x for n in ev.route_nodes]
        ys = [instance.nodes[n].y for n in ev.route_nodes]
        ax.plot(xs, ys, color=cmap(ev.ev_id % 10), linewidth=1.2, alpha=0.8, label=f"EV {ev.ev_id}")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"Routes: {instance.instance_name}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
