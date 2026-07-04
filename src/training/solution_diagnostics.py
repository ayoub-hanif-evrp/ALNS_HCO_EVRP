"""Solution-behavior visual diagnostics from representative rollout traces."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.env.cost import CostWeights
from src.training.flat_schedule import sample_fleet_size
from src.training.rollout_trace import run_traced_episode
from src.training.training_dashboard import FigureResult, _register, _save_fig
from src.utils.logging_utils import get_logger

LOGGER = get_logger()

SOLUTION_SECTION = "8. Solution behavior diagnostics"

SUBSECTIONS = {
    "route_maps": "Route maps",
    "timelines": "EV timelines",
    "battery": "Battery trajectories",
    "unserved": "Unserved over decisions",
    "actions": "Action sequence",
    "feasibility": "Feasibility over decisions",
    "slack": "Time-window slack at service",
    "balance": "Route balance",
    "table": "Final-cycle instance table",
}

EV_COLORS = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2", "#be185d", "#4f46e5"]


@dataclass
class TraceSelection:
    tag: str
    row: dict[str, Any]
    trace_path: Path | None = None
    trace: dict[str, Any] | None = None


def dashboard_cfg(config: dict) -> dict:
    dcfg = dict(config.get("dashboard") or {})
    dcfg.setdefault("save_representative_traces", True)
    dcfg.setdefault("num_random_solution_maps", 1)
    dcfg.setdefault("include_best_final_cycle", True)
    dcfg.setdefault("include_worst_final_cycle", True)
    dcfg.setdefault("include_best_overall", True)
    dcfg.setdefault("include_worst_served_ratio", True)
    dcfg.setdefault("trace_mode", "greedy")
    return dcfg


def select_representative_episodes(df: pd.DataFrame, cfg: dict, seed: int = 42) -> list[TraceSelection]:
    if df.empty:
        return []
    rng = np.random.default_rng(seed)
    obj_col = "final_objective" if "final_objective" in df.columns else "cost"
    if obj_col not in df.columns:
        return []

    out: list[TraceSelection] = []
    seen_epochs: set[int] = set()

    def _add(tag: str, row: pd.Series) -> None:
        ep = int(row.get("epoch", -1))
        if ep in seen_epochs:
            return
        seen_epochs.add(ep)
        out.append(TraceSelection(tag=tag, row=row.to_dict()))

    fc = df[df["cycle"] == df["cycle"].max()].copy() if "cycle" in df.columns else df.copy()
    if len(fc) > 0 and cfg.get("num_random_solution_maps", 1) > 0:
        _add("random_final_cycle", fc.iloc[int(rng.integers(0, len(fc)))])

    if cfg.get("include_best_final_cycle", True) and len(fc) > 0:
        _add("best_final_cycle", fc.loc[fc[obj_col].idxmin()])
    if cfg.get("include_worst_final_cycle", True) and len(fc) > 0:
        _add("worst_final_cycle", fc.loc[fc[obj_col].idxmax()])
    if cfg.get("include_best_overall", True):
        _add("best_overall", df.loc[df[obj_col].idxmin()])
    if cfg.get("include_worst_served_ratio", True) and "served_rate" in df.columns:
        wsr = df.loc[df["served_rate"].idxmin()]
        wo = fc.loc[fc[obj_col].idxmax()] if len(fc) else wsr
        if int(wsr.get("epoch", -2)) != int(wo.get("epoch", -1)):
            _add("worst_served_ratio", wsr)
    return out


def _resolve_instance(train_ds, row: dict):
    if row.get("instance_idx") is not None:
        return train_ds[int(row["instance_idx"])]
    name = row.get("instance_name")
    if name in train_ds.names:
        return train_ds[train_ds.names.index(name)]
    return train_ds[0]


def collect_representative_traces(
    config: dict, df: pd.DataFrame, policy, train_ds, device, *, traces_dir: Path, seed: int = 42,
) -> list[TraceSelection]:
    cfg = dashboard_cfg(config)
    if not cfg.get("save_representative_traces", True):
        return []
    traces_dir.mkdir(parents=True, exist_ok=True)
    selections = select_representative_episodes(df, cfg, seed=seed)
    if not selections:
        return []

    import random
    rng = random.Random(seed)
    weights = CostWeights.from_config(config["cost"])
    prob = config.get("problem") or {}
    decode = config.get("decoding") or {}
    mode = str(cfg.get("trace_mode", decode.get("mode", "greedy")))
    hard_tw = str((config.get("constraints") or {}).get("time_windows_mode", "")) == "hard_original"

    for sel in selections:
        try:
            inst = _resolve_instance(train_ds, sel.row)
            m = max(1, int(round(float(sel.row.get("avg_used_evs", inst.m_min))))) if sel.row.get("avg_used_evs") else sample_fleet_size(inst, 3, rng)
            _, trace = run_traced_episode(
                inst, policy, m, weights, mode=mode, device=device,
                charging_portions=policy.charging_portions,
                use_horizon_check=bool(prob.get("use_horizon_check", True)),
                use_depot_charging=bool(prob.get("use_depot_charging", True)),
                max_steps_factor=int(decode.get("max_steps_factor", 6)),
                hard_time_windows=hard_tw,
                selection_tag=sel.tag, log_row=sel.row,
            )
            path = traces_dir / f"trace_{sel.tag}.json"
            path.write_text(json.dumps(trace, indent=2), encoding="utf-8")
            sel.trace_path = path
            sel.trace = trace
        except Exception as exc:
            LOGGER.warning("Trace %s failed: %s", sel.tag, exc)
    return selections


def _title_from_trace(trace: dict) -> str:
    m = trace.get("meta") or {}
    return (
        f"{m.get('instance_name', '?')} | ep {m.get('epoch', '?')} | cyc {m.get('cycle', '?')} | "
        f"J={float(m.get('objective', 0)):.0f} served={float(m.get('served_ratio', 0)):.1%} "
        f"unserved={m.get('unserved_customers', 0)} feas={m.get('feasible')} EVs={m.get('used_evs', 0)} "
        f"dist={float(m.get('total_distance', 0)):.0f} charge={float(m.get('total_charging_time', 0)):.1f} "
        f"cLat={float(m.get('customer_lateness', 0)):.2f} dLat={float(m.get('depot_lateness', 0)):.2f} "
        f"bViol={float(m.get('battery_violation', 0)):.2f}"
    )


def _subtitle_from_trace(trace: dict) -> str:
    return _title_from_trace(trace)


def _style_diag_axes(ax, *, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_facecolor("#f8fafc")
    ax.grid(True, color="#cbd5e1", linewidth=0.6, alpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, fontweight="bold")


def _set_diag_title(fig, ax, title: str, subtitle: str) -> None:
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98, color="#0f172a")
    ax.set_title(subtitle, fontsize=9, pad=10, color="#475569")


def _save_live_fig(fig: plt.Figure, out_dir: Path, filename: str, *, save_svg: bool, dpi: int = 150) -> Path | None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / filename
    try:
        fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    except Exception as exc:
        plt.close(fig)
        LOGGER.warning("Failed to save %s: %s", png, exc)
        return None
    if save_svg:
        try:
            fig.savefig(png.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
        except Exception:
            pass
    plt.close(fig)
    return png


def _plot_solution_map(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    import matplotlib.patches as mpatches

    nodes = trace.get("nodes") or []
    routes = trace.get("routes") or []
    served = trace.get("served_mask") or []
    meta = trace.get("meta") or {}
    if not nodes:
        return None

    node_by_id = {n["id"]: n for n in nodes}
    xs = {n["id"]: n["x"] for n in nodes}
    ys = {n["id"]: n["y"] for n in nodes}
    depot_ids = {n["id"] for n in nodes if n["type"] == "depot"}

    fig = plt.figure(figsize=(18, 13), facecolor="#f8fafc")
    ax = fig.add_axes([0.06, 0.07, 0.67, 0.86])
    info_ax = fig.add_axes([0.76, 0.07, 0.22, 0.86])
    info_ax.set_axis_off()

    ax.set_facecolor("#f1f5f9")
    ax.grid(True, color="#cbd5e1", linewidth=0.55, alpha=0.75)
    ax.set_aspect("equal", adjustable="box")

    route_patches: list[mpatches.Patch] = []
    active_routes = [r for r in routes if r.get("route_nodes")]

    for i, route in enumerate(active_routes):
        ev_id = route.get("ev_id", i)
        color = EV_COLORS[i % len(EV_COLORS)]
        rnodes = route["route_nodes"]
        n_cust = sum(1 for nid in rnodes if node_by_id.get(nid, {}).get("type") == "customer")

        for k in range(1, len(rnodes)):
            a, b = rnodes[k - 1], rnodes[k]
            ax.plot(
                [xs[a], xs[b]], [ys[a], ys[b]],
                color=color, lw=2.4, alpha=0.88, zorder=1, solid_capstyle="round",
            )
            ax.annotate(
                "",
                xy=(xs[b], ys[b]),
                xytext=(xs[a], ys[a]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=color,
                    lw=2.0,
                    mutation_scale=16,
                    shrinkA=10,
                    shrinkB=12,
                ),
                zorder=2,
            )

        for k, nid in enumerate(rnodes):
            ntype = node_by_id.get(nid, {}).get("type", "")
            ax.annotate(
                str(k),
                (xs[nid], ys[nid]),
                textcoords="offset points",
                xytext=(0, -16),
                fontsize=7,
                fontweight="bold",
                color="white",
                bbox=dict(boxstyle="circle,pad=0.28", facecolor=color, edgecolor="white", linewidth=0.9),
                ha="center",
                va="center",
                zorder=7,
            )
            if nid in depot_ids and k > 0:
                ax.annotate(
                    "depot return",
                    (xs[nid], ys[nid]),
                    xytext=(10, 6),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    fontweight="bold",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9),
                )
            if ntype == "station":
                ax.annotate(
                    "charge",
                    (xs[nid], ys[nid]),
                    xytext=(0, 12),
                    textcoords="offset points",
                    fontsize=7,
                    ha="center",
                    color="#92400e",
                    fontweight="bold",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#fef3c7", edgecolor="#f59e0b"),
                )

        route_patches.append(
            mpatches.Patch(
                color=color,
                label=(
                    f"EV {ev_id}: {n_cust} customers, "
                    f"dist {float(route.get('route_distance', 0)):.0f}, "
                    f"charge {float(route.get('total_charging_time', 0)):.1f}"
                ),
            )
        )

    for n in nodes:
        nid = n["id"]
        if n["type"] == "depot":
            ax.scatter(
                n["x"], n["y"], marker="s", s=450, c="#0f172a",
                edgecolors="white", linewidths=2.5, zorder=5,
            )
            ax.scatter(n["x"], n["y"], marker="s", s=200, c="#475569", zorder=5)
            ax.annotate(
                "DEPOT",
                (n["x"], n["y"]),
                xytext=(0, 18),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                ha="center",
                color="#0f172a",
                zorder=6,
            )
        elif n["type"] == "station":
            ax.scatter(
                n["x"], n["y"], marker="^", s=260, c="#fbbf24",
                edgecolors="#b45309", linewidths=1.6, zorder=4,
            )
            sid = n.get("string_id") or f"S{nid}"
            ax.annotate(
                sid,
                (n["x"], n["y"]),
                xytext=(0, 13),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                color="#92400e",
                fontweight="bold",
                zorder=6,
            )
        elif n["type"] == "customer":
            ok = served[nid] if nid < len(served) else False
            fill = "#10b981" if ok else "#ef4444"
            edge = "#047857" if ok else "#b91c1c"
            ax.scatter(
                n["x"], n["y"], marker="o",
                s=160 if ok else 130, c=fill,
                edgecolors=edge, linewidths=1.4,
                alpha=0.95 if ok else 0.9, zorder=3,
            )
            label = str(n.get("string_id") or nid)
            ax.annotate(
                label,
                (n["x"], n["y"]),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                color=edge,
                fontweight="bold" if not ok else "normal",
                zorder=6,
            )
            if not ok:
                ax.annotate(
                    "UNSERVED",
                    (n["x"], n["y"]),
                    xytext=(0, -13),
                    textcoords="offset points",
                    fontsize=6,
                    ha="center",
                    color="#b91c1c",
                    fontweight="bold",
                    zorder=6,
                )

    node_patches = [
        mpatches.Patch(color="#10b981", label="Served customer"),
        mpatches.Patch(color="#ef4444", label="Unserved customer"),
        mpatches.Patch(color="#fbbf24", label="Charging station"),
        mpatches.Patch(color="#0f172a", label="Depot"),
    ]
    ax.legend(
        handles=node_patches + route_patches,
        loc="upper left",
        fontsize=7.5,
        framealpha=0.96,
        title="Map legend",
        title_fontsize=8,
        borderpad=0.8,
    )
    ax.set_xlabel("X", fontsize=10, fontweight="bold")
    ax.set_ylabel("Y", fontsize=10, fontweight="bold")

    feas = "FEASIBLE" if meta.get("feasible") else "INFEASIBLE"
    inst = meta.get("instance_name", "?")
    fig.suptitle(f"EVRP full solution map · {inst}", fontsize=15, fontweight="bold", y=0.98)
    ax.set_title(
        f"epoch {meta.get('epoch', '?')} · cycle {meta.get('cycle', '?')} · {feas} · "
        f"served {float(meta.get('served_ratio', 0)):.1%} · unserved {meta.get('unserved_customers', 0)} · "
        f"EVs {meta.get('used_evs', 0)} · J={float(meta.get('objective', 0)):.0f}",
        fontsize=10,
        pad=14,
        color="#334155",
    )

    summary_lines = [
        "Solution summary",
        "─────────────────────",
        f"Instance     : {inst}",
        f"Epoch/cycle  : {meta.get('epoch', '?')} / {meta.get('cycle', '?')}",
        f"Objective J  : {float(meta.get('objective', 0)):.1f}",
        f"Served ratio : {float(meta.get('served_ratio', 0)):.1%}",
        f"Unserved     : {meta.get('unserved_customers', 0)}",
        f"Feasible     : {meta.get('feasible')}",
        f"Used EVs     : {meta.get('used_evs', 0)}",
        f"Distance     : {float(meta.get('total_distance', 0)):.1f}",
        f"Charge time  : {float(meta.get('total_charging_time', 0)):.1f}",
        f"Customer TW  : {float(meta.get('customer_lateness', 0)):.2f}",
        f"Depot TW     : {float(meta.get('depot_lateness', 0)):.2f}",
        f"Battery viol.: {float(meta.get('battery_violation', 0)):.2f}",
        "",
        "Route order",
        "─────────────────────",
        "Numbers on map = visit",
        "sequence per EV route.",
        "Arrows show travel",
        "direction.",
    ]
    for route in active_routes:
        ev_id = route.get("ev_id", "?")
        rnodes = route.get("route_nodes") or []
        seq = " → ".join(
            str(node_by_id.get(nid, {}).get("string_id") or nid) for nid in rnodes[:8]
        )
        if len(rnodes) > 8:
            seq += " → …"
        summary_lines.append(f"EV {ev_id}: {seq}")

    info_ax.text(
        0.0, 1.0, "\n".join(summary_lines),
        va="top", ha="left", fontsize=7.5, family="monospace", color="#1e293b",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#94a3b8", linewidth=1.2),
        transform=info_ax.transAxes,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / filename
    fig.savefig(png, dpi=170, bbox_inches="tight", facecolor=fig.get_facecolor())
    if save_svg:
        try:
            fig.savefig(png.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
        except Exception:
            pass
    plt.close(fig)
    return png


def _plot_route_timeline(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    import matplotlib.patches as mpatches

    routes = [r for r in (trace.get("routes") or []) if r.get("visits")]
    nodes = {n["id"]: n for n in (trace.get("nodes") or [])}
    if not routes:
        return None

    seg_colors = {
        "travel": "#94a3b8",
        "waiting": "#fcd34d",
        "service": "#2563eb",
        "charging": "#f59e0b",
        "depot_return": "#0f172a",
    }
    n_ev = len(routes)
    fig, ax = plt.subplots(figsize=(14, max(4.5, n_ev * 1.2 + 1.5)), facecolor="#f8fafc")
    y_ticks, y_labels = [], []

    for route in routes:
        ev_id = route.get("ev_id", 0)
        y = float(ev_id)
        y_ticks.append(y)
        y_labels.append(f"EV {ev_id}")
        visits = route["visits"]

        for k in range(1, len(visits)):
            prev, cur = visits[k - 1], visits[k]
            t0 = float(prev.get("departure") or 0)
            t1 = float(cur.get("arrival") or 0)
            if t1 > t0:
                ax.barh(y, t1 - t0, left=t0, height=0.42, color=seg_colors["travel"], alpha=0.9, edgecolor="white")

        for k, cur in enumerate(visits):
            ntype = nodes.get(cur["node"], {}).get("type", "")
            arr = float(cur.get("arrival") or 0)
            dep = float(cur.get("departure") or 0)
            if ntype == "customer" and dep > arr:
                ax.barh(y, dep - arr, left=arr, height=0.42, color=seg_colors["service"], alpha=0.95, edgecolor="white")
                nid = nodes.get(cur["node"], {}).get("string_id") or cur["node"]
                ax.text(dep, y + 0.24, str(nid), fontsize=7, color="#1e40af", ha="left", va="bottom")
            elif ntype == "station" and dep > arr:
                ax.barh(y, dep - arr, left=arr, height=0.42, color=seg_colors["charging"], alpha=0.95, edgecolor="white")
                ax.text(dep, y + 0.24, "ST", fontsize=7, color="#92400e", ha="left", fontweight="bold")
            elif ntype == "depot" and k > 0 and dep >= arr:
                ax.barh(y, max(0.1, dep - arr), left=arr, height=0.42, color=seg_colors["depot_return"], alpha=0.85, edgecolor="white")
                ax.text(dep, y + 0.24, "DEPOT", fontsize=7, color="#334155", ha="left", fontweight="bold")

        for ch in route.get("charging_actions") or []:
            ct = float(ch.get("charging_time", 0))
            if ct > 0:
                end = float(ch.get("departure", 0))
                ax.barh(y, ct, left=end - ct, height=0.42, color=seg_colors["charging"], alpha=0.75, edgecolor="white", hatch="//")

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=10, fontweight="bold")
    _style_diag_axes(ax, xlabel="Time along route", ylabel="Electric vehicle")
    legend = [
        mpatches.Patch(color=seg_colors["travel"], label="Travel between stops"),
        mpatches.Patch(color=seg_colors["waiting"], label="Waiting before service"),
        mpatches.Patch(color=seg_colors["service"], label="Customer service"),
        mpatches.Patch(color=seg_colors["charging"], label="Charging at station"),
        mpatches.Patch(color=seg_colors["depot_return"], label="Depot return / finish"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, framealpha=0.95, title="Timeline segments")
    _set_diag_title(
        fig, ax,
        "EV route timeline (Gantt view)",
        "Shows how each EV spends time: travel, waiting, service, charging, depot. "
        + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_battery_trajectory(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    routes = [r for r in (trace.get("routes") or []) if r.get("battery_trace")]
    nodes = {n["id"]: n for n in (trace.get("nodes") or [])}
    if not routes:
        return None

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#f8fafc")
    ax.axhline(0, color="#dc2626", ls="--", lw=1.5, label="Empty battery (0)")
    ax.axhspan(-1e9, 0, color="#fee2e2", alpha=0.35, zorder=0)

    for i, route in enumerate(routes):
        ev_id = route.get("ev_id", i)
        color = EV_COLORS[i % len(EV_COLORS)]
        times = route.get("departure_times") or []
        batt = route.get("battery_trace") or []
        rnodes = route.get("route_nodes") or []
        n = min(len(times), len(batt))
        if n < 2:
            continue
        ax.plot(times[:n], batt[:n], marker="o", ms=5, lw=2.2, color=color, label=f"EV {ev_id}", zorder=3)
        for k in range(1, n):
            if batt[k] > batt[k - 1] + 0.5:
                ax.annotate(
                    "",
                    xy=(times[k], batt[k]),
                    xytext=(times[k], batt[k - 1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
                    zorder=4,
                )
                ax.text(times[k], batt[k], " charge", fontsize=7, color=color, va="bottom")
        for k, nid in enumerate(rnodes[:n]):
            ntype = nodes.get(nid, {}).get("type", "")
            if ntype == "customer":
                ax.scatter(times[k], batt[k], s=60, facecolors="white", edgecolors=color, linewidths=1.5, zorder=5)
            elif ntype == "station":
                ax.scatter(times[k], batt[k], s=70, marker="^", c="#f59e0b", edgecolors="#92400e", zorder=5)

    ax.legend(fontsize=9, loc="upper right", framealpha=0.95, title="Battery level per EV")
    _style_diag_axes(ax, xlabel="Time", ylabel="Remaining battery")
    _set_diag_title(
        fig, ax,
        "Battery trajectory per EV",
        "Line = battery over time. Upward jumps = charging. △ = station, ○ = customer. Red zone = below zero. "
        + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_unserved(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    decs = trace.get("decisions") or []
    if not decs:
        return None
    x = [d["decision_index"] for d in decs]
    unserved = [d.get("unserved_after", 0) for d in decs]
    expired = [d.get("expired_after", 0) for d in decs]
    feasible = [d.get("feasible_customers", 0) for d in decs]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#f8fafc")
    ax.fill_between(x, unserved, color="#fecaca", alpha=0.4)
    ax.plot(x, unserved, label="Unserved customers remaining", color="#dc2626", lw=2.5, marker="o", ms=3)
    ax.plot(x, expired, label="Time-window expired (unserved)", color="#d97706", lw=2, marker="s", ms=3)
    ax.plot(x, feasible, label="Feasible customers still reachable", color="#059669", lw=2, marker="^", ms=3)
    ax.legend(fontsize=9, loc="upper right", framealpha=0.95)
    _style_diag_axes(ax, xlabel="Decision step in episode", ylabel="Customer count")
    _set_diag_title(
        fig, ax,
        "Unserved customers over decisions",
        "Should trend down as customers are served. High expired = TW pressure. Low feasible = policy stuck. "
        + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_actions(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    decs = list(trace.get("decisions") or [])
    for tr in trace.get("env_transitions") or []:
        decs.append({"decision_index": tr.get("index", 0), "action_type": tr.get("type", "auto_close")})
    decs.sort(key=lambda d: d.get("decision_index", 0))
    if not decs:
        return None

    ordered = ["customer", "station", "depot_recharge", "depot_finish", "auto_close"]
    y_map = {k: i for i, k in enumerate(ordered)}
    action_styles = {
        "customer": ("Customer visit", "#059669"),
        "station": ("Station charge", "#f59e0b"),
        "depot_recharge": ("Depot recharge", "#2563eb"),
        "depot": ("Depot recharge", "#2563eb"),
        "depot_finish": ("Depot finish route", "#0f172a"),
        "auto_close": ("Auto-close route", "#7c3aed"),
        "auto_close_forced": ("Auto-close route", "#7c3aed"),
    }
    xs = [d["decision_index"] for d in decs]
    types = [d.get("action_type", "customer") for d in decs]

    fig, ax = plt.subplots(figsize=(13, 4.5), facecolor="#f8fafc")
    for x, atype in zip(xs, types):
        _, color = action_styles.get(atype, (atype, "#64748b"))
        key = atype if atype in y_map else ("depot" if atype == "depot" else "auto_close")
        y = y_map[key]
        ax.bar(x, 0.85, bottom=y - 0.42, width=0.9, color=color, edgecolor="white", linewidth=0.6)

    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels([action_styles[k][0] for k in ordered], fontsize=9)
    ax.set_xlim(min(xs) - 0.5, max(xs) + 0.5)
    _style_diag_axes(ax, xlabel="Decision step", ylabel="Action type")
    _set_diag_title(
        fig, ax,
        "Action sequence composition",
        "Each bar = one decision. Look for too many station/depot actions vs customer visits. "
        + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_feasible(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    decs = trace.get("decisions") or []
    if not decs:
        return None
    x = [d["decision_index"] for d in decs]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), facecolor="#f8fafc", sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax_top, ax_bot = axes

    for key, label, color in [
        ("feasible_customers", "Feasible customers", "#059669"),
        ("feasible_stations", "Feasible stations", "#f59e0b"),
        ("feasible_depot", "Feasible depot actions", "#2563eb"),
    ]:
        ax_top.plot(x, [d.get(key, 0) for d in decs], label=label, color=color, lw=2.2, marker="o", ms=2)
    ax_top.legend(fontsize=8, loc="upper right", title="Still available actions")
    _style_diag_axes(ax_top, ylabel="Feasible action count")
    ax_top.set_title("Available actions at each decision", fontsize=10, fontweight="bold", loc="left", color="#334155")

    for key, label, color in [
        ("masked_by_time_window", "Masked by time window", "#dc2626"),
        ("masked_by_battery", "Masked by battery", "#7c3aed"),
        ("masked_by_capacity", "Masked by capacity", "#64748b"),
    ]:
        ax_bot.plot(x, [d.get(key, 0) for d in decs], label=label, color=color, lw=2, ls="--", marker="x", ms=3)
    ax_bot.legend(fontsize=8, loc="upper right", title="Why actions are blocked")
    _style_diag_axes(ax_bot, xlabel="Decision step", ylabel="Masked customer count")

    _set_diag_title(
        fig, ax_bot,
        "Feasible vs masked actions over decisions",
        "Top: what the policy could still choose. Bottom: customers blocked by TW, battery, or capacity. "
        + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_slack(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    served = [
        d for d in (trace.get("decisions") or [])
        if d.get("customer_served") and d.get("service_slack") is not None
    ]
    if not served:
        return None
    slacks = [float(d["service_slack"]) for d in served]
    colors = ["#dc2626" if s < 5 else "#f59e0b" if s < 20 else "#059669" for s in slacks]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#f8fafc")
    bars = ax.bar(range(1, len(slacks) + 1), slacks, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(0, color="#dc2626", lw=2, label="Due time (zero slack = tight)")
    ax.axhspan(-1e9, 0, color="#fee2e2", alpha=0.3)
    for i, (d, bar) in enumerate(zip(served, bars)):
        if i % max(1, len(served) // 12) == 0 or i == len(served) - 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{slacks[i]:.0f}", ha="center", va="bottom", fontsize=7)
    ax.legend(fontsize=9)
    _style_diag_axes(ax, xlabel="Customer visit order in episode", ylabel="Slack = due_time − service_start")
    _set_diag_title(
        fig, ax,
        "Time-window slack at customer service",
        "Green = served early, amber = moderate slack, red = served near due time. Below 0 = late. "
        + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def _plot_balance(trace: dict, out_dir: Path, filename: str, *, save_svg: bool) -> Path | None:
    routes = [r for r in (trace.get("routes") or []) if r.get("route_nodes")]
    nodes = {n["id"]: n for n in (trace.get("nodes") or [])}
    if not routes:
        return None

    labels = [f"EV {r['ev_id']}" for r in routes]
    metrics = {
        "Customers served": [sum(1 for nid in r["route_nodes"] if nodes.get(nid, {}).get("type") == "customer") for r in routes],
        "Route distance": [float(r.get("route_distance", 0)) for r in routes],
        "Charging time": [float(r.get("total_charging_time", 0)) for r in routes],
        "Route duration": [
            float((r.get("departure_times") or [0])[-1] - (r.get("arrival_times") or [0])[0]) for r in routes
        ],
    }
    colors = ["#059669", "#2563eb", "#f59e0b", "#7c3aed"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), facecolor="#f8fafc")
    for ax, (title, values), color in zip(axes.flat, metrics.items(), colors):
        bars = ax.bar(labels, values, color=color, edgecolor="white", linewidth=0.8)
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
        _style_diag_axes(ax, ylabel=title)
        ax.set_title(title, fontsize=10, fontweight="bold", loc="left", color="#334155")
        ax.tick_params(axis="x", labelsize=9)

    _set_diag_title(
        fig, axes[0, 0],
        "Per-EV workload balance",
        "Compare fleet usage: one EV doing all work vs balanced routes. " + _subtitle_from_trace(trace),
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return _save_live_fig(fig, out_dir, filename, save_svg=save_svg)


def build_final_cycle_table_html(df: pd.DataFrame) -> str:
    if df.empty or "cycle" not in df.columns:
        return "<p>No final-cycle data.</p>"
    fc = df[df["cycle"] == df["cycle"].max()].copy()
    obj = "final_objective" if "final_objective" in fc.columns else "cost"
    fc = fc.sort_values(obj, ascending=False)
    cols = [
        ("instance_name", "instance"), (obj, "objective"), ("served_rate", "served"),
        ("unserved_customers", "unserved"), ("feasibility_rate", "feasible"),
        ("avg_used_evs", "EVs"), ("avg_distance", "distance"),
        ("charging_station_visits", "stations"), ("unnecessary_station_actions", "bad st."),
        ("early_depot_returns", "early depot"), ("customer_lateness", "cust TW"),
        ("depot_lateness", "depot TW"), ("avg_battery_violation", "battery"),
        ("relative_objective_mismatch", "mm"),
    ]
    present = [(k, l) for k, l in cols if k in fc.columns]
    header = "".join(f"<th>{l}</th>" for _, l in present)
    body = []
    for _, row in fc.iterrows():
        cells = []
        for key, _ in present:
            v = row[key]
            cells.append(f"<td>{v:.4f}</td>" if isinstance(v, float) and abs(v) < 1000 else f"<td>{v}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<h3>Final-cycle instances (worst objective first)</h3>"
        f"<table class='diag-table'><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table>"
    )


_TRACE_PLOTS = [
    ("random_final_cycle", "20_solution_map_random_final_cycle.png", _plot_solution_map, "route_maps", True),
    ("best_final_cycle", "21_solution_map_best_final_cycle.png", _plot_solution_map, "route_maps", True),
    ("worst_final_cycle", "22_solution_map_worst_final_cycle.png", _plot_solution_map, "route_maps", True),
    ("best_overall", "23_solution_map_best_overall.png", _plot_solution_map, "route_maps", False),
    ("random_final_cycle", "24_route_timeline_random_final_cycle.png", _plot_route_timeline, "timelines", True),
    ("best_final_cycle", "25_route_timeline_best_final_cycle.png", _plot_route_timeline, "timelines", True),
    ("worst_final_cycle", "26_route_timeline_worst_final_cycle.png", _plot_route_timeline, "timelines", True),
    ("random_final_cycle", "27_battery_trajectory_random_final_cycle.png", _plot_battery_trajectory, "battery", True),
    ("best_final_cycle", "28_battery_trajectory_best_final_cycle.png", _plot_battery_trajectory, "battery", True),
    ("worst_final_cycle", "29_battery_trajectory_worst_final_cycle.png", _plot_battery_trajectory, "battery", True),
    ("random_final_cycle", "30_unserved_over_decisions_random.png", _plot_unserved, "unserved", True),
    ("best_final_cycle", "31_unserved_over_decisions_best.png", _plot_unserved, "unserved", True),
    ("worst_final_cycle", "32_unserved_over_decisions_worst.png", _plot_unserved, "unserved", True),
    ("random_final_cycle", "33_action_sequence_random.png", _plot_actions, "actions", True),
    ("best_final_cycle", "34_action_sequence_best.png", _plot_actions, "actions", True),
    ("worst_final_cycle", "35_action_sequence_worst.png", _plot_actions, "actions", True),
    ("random_final_cycle", "36_feasible_actions_random.png", _plot_feasible, "feasibility", True),
    ("best_final_cycle", "37_feasible_actions_best.png", _plot_feasible, "feasibility", True),
    ("worst_final_cycle", "38_feasible_actions_worst.png", _plot_feasible, "feasibility", True),
    ("random_final_cycle", "39_service_slack_random.png", _plot_slack, "slack", False),
    ("best_final_cycle", "40_service_slack_best.png", _plot_slack, "slack", False),
    ("worst_final_cycle", "41_service_slack_worst.png", _plot_slack, "slack", False),
    ("best_final_cycle", "42_ev_route_balance_best.png", _plot_balance, "balance", False),
    ("worst_final_cycle", "43_ev_route_balance_worst.png", _plot_balance, "balance", False),
]

_GROUP_ORDER = [
    "route_maps", "timelines", "battery", "unserved", "actions", "feasibility", "slack", "balance",
]


def _artifact_label(filename: str) -> str:
    return filename.replace(".png", "").replace("_", " ")


def build_solution_diagnostic_groups() -> list[dict[str, Any]]:
    """Grouped artifact manifest shared by static and live dashboards."""
    grouped: dict[str, dict[str, Any]] = {
        key: {"id": key, "title": SUBSECTIONS[key], "items": []} for key in _GROUP_ORDER
    }
    for tag, filename, _plot_fn, subsection, required in _TRACE_PLOTS:
        grouped[subsection]["items"].append({
            "file": filename,
            "tag": tag,
            "title": _artifact_label(filename),
            "required": required,
        })
    return [grouped[key] for key in _GROUP_ORDER]


SOLUTION_DIAGNOSTIC_GROUPS = build_solution_diagnostic_groups()

LIVE_PLOT_SPECS: list[tuple[str, str, Any]] = [
    ("live_solution_map.png", "Route map", _plot_solution_map),
    ("live_route_timeline.png", "Route timeline", _plot_route_timeline),
    ("live_battery_trajectory.png", "Battery trajectory", _plot_battery_trajectory),
    ("live_unserved_over_decisions.png", "Unserved over decisions", _plot_unserved),
    ("live_action_sequence.png", "Action sequence", _plot_actions),
    ("live_feasible_actions.png", "Feasible actions over decisions", _plot_feasible),
    ("live_service_slack.png", "Time-window slack at service", _plot_slack),
    ("live_ev_route_balance.png", "Per-EV route balance", _plot_balance),
]

LIVE_DIAGNOSTIC_GROUPS = [
    {
        "id": "live_snapshot",
        "title": "Live random instance snapshot",
        "items": [
            {"file": f"live/{filename}", "title": title, "required": True}
            for filename, title, _ in LIVE_PLOT_SPECS
        ],
    },
]


def generate_live_diagnostic_figures(trace: dict, live_dir: Path) -> dict[str, Path | None]:
    """Write live-dashboard PNGs for one traced episode under ``live_dir``."""
    live_dir = Path(live_dir)
    live_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path | None] = {}
    for filename, _title, plot_fn in LIVE_PLOT_SPECS:
        rel = f"live/{filename}"
        try:
            png = plot_fn(trace, live_dir, filename, save_svg=False)
            written[rel] = png
        except Exception as exc:
            LOGGER.warning("Live plot %s failed: %s", filename, exc)
            written[rel] = None
    return written


def generate_solution_diagnostic_figures(
    selections: list[TraceSelection], figures_dir: Path, df: pd.DataFrame, *, save_svg: bool = True,
) -> tuple[list[FigureResult], str]:
    figures_dir = Path(figures_dir)
    by_tag = {s.tag: s for s in selections if s.trace}
    results: list[FigureResult] = []
    for tag, filename, plot_fn, subsection, required in _TRACE_PLOTS:
        section = f"{SOLUTION_SECTION} — {SUBSECTIONS[subsection]}"
        name = filename.replace(".png", "").replace("_", " ")
        sel = by_tag.get(tag)
        if not sel:
            _register(results, name=name, filename=filename, metrics=[tag], section=section,
                      description=f"Episode {tag}", required=required, png=None, svg=None,
                      skip_reason=f"no trace for {tag}")
            continue
        try:
            png = plot_fn(sel.trace, figures_dir, filename, save_svg=save_svg)
            svg_p = figures_dir / filename.replace(".png", ".svg") if save_svg and png else None
            _register(results, name=name, filename=filename, metrics=[tag], section=section,
                      description=f"Representative episode: {tag}", required=required,
                      png=png, svg=str(svg_p) if svg_p and svg_p.exists() else None,
                      skip_reason=None if png else "empty plot")
        except Exception as exc:
            _register(results, name=name, filename=filename, metrics=[tag], section=section,
                      description=f"Episode {tag}", required=required, png=None, svg=None, skip_reason=str(exc))
    table_html = build_final_cycle_table_html(df)
    table_path = figures_dir / "44_final_cycle_instance_table.html"
    table_path.write_text(
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<style>.diag-table{border-collapse:collapse;width:100%;font-size:12px}"
        "th,td{border:1px solid #334155;padding:4px 8px}th{background:#1e293b;color:#e2e8f0}</style>"
        f"</head><body>{table_html}</body></html>", encoding="utf-8",
    )
    fr = FigureResult(
        name="Final-cycle instance summary table", filename="44_final_cycle_instance_table.html",
        metrics=["final_cycle"], section=f"{SOLUTION_SECTION} — {SUBSECTIONS['table']}",
        description="Final cycle per-instance metrics.", required=True, generated=True,
        png_path=str(table_path),
    )
    results.append(fr)
    return results, table_html


def run_solution_behavior_diagnostics(
    config: dict, df: pd.DataFrame, policy, train_ds, device, *, figures_dir: Path,
    seed: int = 42, save_svg: bool = True,
) -> tuple[list[FigureResult], str]:
    cfg = dashboard_cfg(config)
    if not cfg.get("save_representative_traces", True):
        return [], ""
    out_cfg = config.get("outputs") or {}
    traces_dir = Path(out_cfg.get("traces", Path(out_cfg.get("root", "outputs")) / "traces"))
    selections = collect_representative_traces(config, df, policy, train_ds, device, traces_dir=traces_dir, seed=seed)
    return generate_solution_diagnostic_figures(selections, figures_dir, df, save_svg=save_svg)
