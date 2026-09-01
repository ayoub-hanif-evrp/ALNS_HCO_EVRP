"""Ten paper figures (PNG only) from a compact Hanif CARE-FI results table."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

DPI = 300
C_ORIG = "#1f4e79"
C_RELAX = "#c45c26"
C_S1 = "#1f4e79"
C_S2 = "#2f6b4f"
C_CUST = "#94a3b8"
C_DEPOT = "#1e293b"
C_STAT = "#c45c26"
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
ROUTE_COLORS = [
    "#1f4e79", "#c45c26", "#2f6b4f", "#7c3aed", "#b45309",
    "#0e7490", "#be123c", "#365314", "#1d4ed8", "#9a3412",
]


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _rows(data: list[dict], tw: str) -> list[dict]:
    key = "relaxed" if tw.startswith("relax") else "original"
    return [r for r in data if str(r.get("tw", "")).startswith(key[:4]) or str(r.get("tw")) == key]


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


def _family_means(rows: list[dict], key: str) -> list[float]:
    out = []
    for fam in FAMILIES:
        sub = [r for r in rows if r.get("family") == fam]
        out.append(_mean(sub, key))
    return out


def fig01_fleet_box(data: list[dict], out: Path) -> Path:
    orig = [float(r["k"]) for r in _rows(data, "original")]
    relax = [float(r["k"]) for r in _rows(data, "relaxed")]
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    bp = ax.boxplot(
        [orig, relax],
        tick_labels=["Original TW", "Relaxed TW"],
        patch_artist=True,
        widths=0.55,
        medianprops={"color": "#111827", "linewidth": 1.4},
        whiskerprops={"color": "#334155"},
        capprops={"color": "#334155"},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "#94a3b8", "markeredgecolor": "none"},
    )
    for patch, color in zip(bp["boxes"], (C_ORIG, C_RELAX)):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
        patch.set_edgecolor(color)
    ax.set_ylabel("Vehicles")
    _style(ax)
    return _save(fig, out / "fig01_fleet_size.png")


def fig02_fleet_family(data: list[dict], out: Path) -> Path:
    o = _family_means(_rows(data, "original"), "k")
    r = _family_means(_rows(data, "relaxed"), "k")
    x = np.arange(len(FAMILIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w / 2, o, w, color=C_ORIG, label="Original TW")
    ax.bar(x + w / 2, r, w, color=C_RELAX, label="Relaxed TW")
    ax.set_xticks(x, FAMILIES)
    ax.set_ylabel("Mean fleet size")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, out / "fig02_fleet_by_family.png")


def fig03_distance_family(data: list[dict], out: Path) -> Path:
    o = _family_means(_rows(data, "original"), "distance")
    r = _family_means(_rows(data, "relaxed"), "distance")
    x = np.arange(len(FAMILIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w / 2, np.array(o) / 1000.0, w, color=C_ORIG, label="Original TW")
    ax.bar(x + w / 2, np.array(r) / 1000.0, w, color=C_RELAX, label="Relaxed TW")
    ax.set_xticks(x, FAMILIES)
    ax.set_ylabel("Mean distance ($\\times 10^3$)")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, out / "fig03_distance_by_family.png")


def fig04_charging_cost(data: list[dict], out: Path) -> Path:
    o1 = _family_means(_rows(data, "original"), "charging_expense")
    o2 = _family_means(_rows(data, "original"), "charging_expense_s2")
    x = np.arange(len(FAMILIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w / 2, o1, w, color=C_S1, label="Stage 1")
    ax.bar(x + w / 2, o2, w, color=C_S2, label="Stage 2")
    ax.set_xticks(x, FAMILIES)
    ax.set_ylabel("Mean charging cost")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, out / "fig04_charging_cost.png")


def fig05_charging_time(data: list[dict], out: Path) -> Path:
    o1 = _family_means(_rows(data, "original"), "charging_time")
    o2 = _family_means(_rows(data, "original"), "charging_time_s2")
    x = np.arange(len(FAMILIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w / 2, o1, w, color=C_S1, label="Stage 1")
    ax.bar(x + w / 2, o2, w, color=C_S2, label="Stage 2")
    ax.set_xticks(x, FAMILIES)
    ax.set_ylabel("Mean charging time")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, out / "fig05_charging_time.png")


def fig06_station_wait(data: list[dict], out: Path) -> Path:
    o1 = _family_means(_rows(data, "original"), "station_waiting")
    o2 = _family_means(_rows(data, "original"), "station_waiting_s2")
    x = np.arange(len(FAMILIES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(x - w / 2, o1, w, color=C_S1, label="Stage 1")
    ax.bar(x + w / 2, o2, w, color=C_S2, label="Stage 2")
    ax.set_xticks(x, FAMILIES)
    ax.set_ylabel("Mean station waiting time")
    ax.legend(frameon=False, fontsize=9)
    _style(ax)
    return _save(fig, out / "fig06_station_waiting.png")


def _draw_map(payload: dict[str, Any], ax) -> None:
    nodes = payload.get("nodes") or []
    paths = payload.get("paths") or []
    cust = [(x, y) for x, y, k in nodes if k == "customer"]
    stat = [(x, y) for x, y, k in nodes if k == "station"]
    depot = [(x, y) for x, y, k in nodes if k == "depot"]
    if cust:
        xs, ys = zip(*cust)
        ax.scatter(xs, ys, s=6, c=C_CUST, zorder=2, linewidths=0)
    if stat:
        xs, ys = zip(*stat)
        ax.scatter(xs, ys, s=28, c=C_STAT, marker="D", zorder=3, linewidths=0)
    if depot:
        ax.scatter([depot[0][0]], [depot[0][1]], s=70, c=C_DEPOT, marker="s", zorder=4)
    for i, path in enumerate(paths):
        if len(path) < 2:
            continue
        xs, ys = zip(*path)
        ax.plot(xs, ys, color=ROUTE_COLORS[i % len(ROUTE_COLORS)], lw=0.7, alpha=0.85, zorder=1)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def fig07_map_clustered(maps: dict, out: Path) -> Path | None:
    payload = maps.get("C1_10_1")
    if not payload:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    _draw_map(payload, ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CUST, markersize=5, label="Customer"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_STAT, markersize=6, label="Station"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_DEPOT, markersize=7, label="Depot"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    return _save(fig, out / "fig07_routes_clustered.png")


def fig08_map_random(maps: dict, out: Path) -> Path | None:
    payload = maps.get("R1_10_1")
    if not payload:
        return None
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    _draw_map(payload, ax)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_CUST, markersize=5, label="Customer"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_STAT, markersize=6, label="Station"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_DEPOT, markersize=7, label="Depot"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper right", fontsize=8)
    return _save(fig, out / "fig08_routes_random.png")


def fig09_paired_fleet(data: list[dict], out: Path) -> Path:
    orig = {r["instance"]: float(r["k"]) for r in _rows(data, "original")}
    relax = {r["instance"]: float(r["k"]) for r in _rows(data, "relaxed")}
    names = sorted(set(orig) & set(relax))
    xs = [orig[n] for n in names]
    ys = [relax[n] for n in names]
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.scatter(xs, ys, s=18, c=C_ORIG, alpha=0.75, linewidths=0, zorder=2)
    lo = min(xs + ys)
    hi = max(xs + ys)
    ax.plot([lo, hi], [lo, hi], color="#94a3b8", lw=1.0, zorder=1)
    ax.set_xlabel("Vehicles, original TW")
    ax.set_ylabel("Vehicles, relaxed TW")
    ax.set_aspect("equal", adjustable="box")
    _style(ax)
    return _save(fig, out / "fig09_paired_fleet.png")


def fig10_time_composition(data: list[dict], out: Path) -> Path:
    rows = _rows(data, "original")
    charge, wait, cust, rest = [], [], [], []
    for fam in FAMILIES:
        sub = [r for r in rows if r.get("family") == fam]
        ct = _mean(sub, "charging_time_s2")
        sw = _mean(sub, "station_waiting_s2")
        cw = _mean(sub, "customer_waiting_s2")
        dur = _mean(sub, "duration_s2")
        residual = max(0.0, dur - ct - sw - cw)
        charge.append(ct)
        wait.append(sw)
        cust.append(cw)
        rest.append(residual)
    x = np.arange(len(FAMILIES))
    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ax.bar(x, rest, color="#cbd5e1", label="Travel and service")
    ax.bar(x, cust, bottom=rest, color="#64748b", label="Customer waiting")
    b2 = np.array(rest) + np.array(cust)
    ax.bar(x, wait, bottom=b2, color=C_RELAX, label="Station waiting")
    b3 = b2 + np.array(wait)
    ax.bar(x, charge, bottom=b3, color=C_S2, label="Charging")
    ax.set_xticks(x, FAMILIES)
    ax.set_ylabel("Mean route duration")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    _style(ax)
    return _save(fig, out / "fig10_time_composition.png")


def generate_all(data: list[dict], maps: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = [
        fig01_fleet_box(data, out_dir),
        fig02_fleet_family(data, out_dir),
        fig03_distance_family(data, out_dir),
        fig04_charging_cost(data, out_dir),
        fig05_charging_time(data, out_dir),
        fig06_station_wait(data, out_dir),
        fig09_paired_fleet(data, out_dir),
        fig10_time_composition(data, out_dir),
    ]
    for fn in (fig07_map_clustered, fig08_map_random):
        p = fn(maps, out_dir)
        if p is not None:
            saved.append(p)
    return [p for p in saved if p is not None]
