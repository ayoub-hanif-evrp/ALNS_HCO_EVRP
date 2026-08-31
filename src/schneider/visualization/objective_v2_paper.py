"""Publication tables and the seven main-paper EVRPTW figures for objective_v2.

Does not put dataset names, seeds, or search-debug labels in figure titles.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from src.schneider.data.hanif_convert import VARIANT_SPECS
from src.schneider.data.instance_loader import load_instance
from src.schneider.evaluation.paper_objective import expected_price_s, expected_wait_s, charging_rate_s
from src.schneider.evaluation.stochastic_robustness import evaluate_solution_stochastic
from src.schneider.search.route_replay import replay_route, replay_solution

ROOT = Path(__file__).resolve().parents[3]
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
MODES = [
    ("classical", "Classical fleet-distance"),
    ("expected", "Expected heterogeneous"),
    ("risk_aware", "Risk-aware heterogeneous"),
]
TW = [("original_tw", "Original TW", "A"), ("relaxed_tw", "Relaxed TW", "B")]
DPI = 300
C_CLASS = "#4a4a4a"
C_EXP = "#1f4e79"
C_RISK = "#c45c26"
COLORS = {"classical": C_CLASS, "expected": C_EXP, "risk_aware": C_RISK}
REPRESENTATIVE = "C1_10_1"


def _save_fig(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    svg = stem.with_suffix(".svg")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        parsed = dict(row)
        for k, v in row.items():
            if v is None or v == "":
                continue
            try:
                if v.lower() in {"true", "false"}:
                    parsed[k] = v.lower() == "true"
                elif "." in v or "e" in v.lower():
                    parsed[k] = float(v)
                else:
                    parsed[k] = int(v)
            except (ValueError, AttributeError):
                parsed[k] = v
        out.append(parsed)
    return out


def _family(name: str) -> str:
    return str(name).split("_")[0]


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if key in r and r[key] not in (None, "")]
    return float(np.mean(vals)) if vals else float("nan")


def load_mode_tables(exp_dir: Path) -> dict[str, dict[str, list[dict]]]:
    data: dict[str, dict[str, list[dict]]] = {}
    for mode, _ in MODES:
        data[mode] = {}
        for tw, _lab, _var in TW:
            data[mode][tw] = _read_csv(exp_dir / mode / tw / "results.csv")
    return data


def _routes(exp_dir: Path, mode: str, tw: str, name: str) -> list | None:
    path = exp_dir / mode / tw / "routes" / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(list(r["nodes"]), list(r.get("labels") or [None] * len(r["nodes"]))) for r in payload["routes"]]


def _instance(variant: str, name: str):
    root = Path(VARIANT_SPECS[variant]["output"])
    return load_instance(root / name)


def write_tables(exp_dir: Path, tables_dir: Path, data: dict, sto: dict | None) -> list[Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def row_for(mode: str, tw: str) -> dict[str, Any]:
        rows = data[mode][tw]
        feas = [r for r in rows if r.get("feasible")]
        n = max(len(rows), 1)
        served_ratios = [
            float(r.get("served") or 0) / max(float(r.get("n_customers") or 1), 1.0)
            for r in rows
        ]
        return {
            "objective": dict(MODES)[mode],
            "tw_variant": tw,
            "feasible_instances": sum(1 for r in rows if r.get("feasible")),
            "n": len(rows),
            "served_ratio": float(np.mean(served_ratios)) if served_ratios else float("nan"),
            "mean_vehicles": _mean(feas, "used_k") if feas else _mean(rows, "used_k"),
            "mean_distance": _mean(feas, "distance") if feas else _mean(rows, "distance"),
            "mean_duration": _mean(feas, "duration") if feas else float("nan"),
            "charging_visits": _mean(feas, "charging_visits") if feas else _mean(rows, "charging_visits"),
            "charged_energy": _mean(feas, "charged_energy") if feas else _mean(rows, "charged_energy"),
            "charging_time": _mean(feas, "charging_time") if feas else _mean(rows, "charging_time"),
            "expected_charging_cost": _mean(feas, "expected_charging_cost") if feas else _mean(rows, "expected_charging_cost"),
            "station_waiting_time": _mean(feas, "station_waiting_time") if feas else _mean(rows, "station_waiting_time"),
            "minimum_soc": _mean(feas, "minimum_soc") if feas else _mean(rows, "minimum_soc"),
            "runtime": _mean(rows, "runtime_s"),
        }

    main_rows = [row_for(m, tw) for tw, *_ in TW for m, _ in MODES]
    main_path = tables_dir / "objective_comparison.csv"
    _write_csv(main_path, main_rows)
    written.append(main_path)

    fam_rows = []
    for tw, _lab, _v in TW:
        for fam in FAMILIES:
            for mode, label in MODES:
                sub = [r for r in data[mode][tw] if _family(str(r.get("instance", ""))) == fam]
                feas = [r for r in sub if r.get("feasible")]
                use = feas or sub
                fam_rows.append({
                    "tw_variant": tw,
                    "family": fam,
                    "objective": label,
                    "n": len(sub),
                    "feasible": len(feas),
                    "mean_vehicles": _mean(use, "used_k"),
                    "mean_distance": _mean(use, "distance"),
                    "station_waiting_time": _mean(use, "station_waiting_time"),
                    "expected_charging_cost": _mean(use, "expected_charging_cost"),
                    "charging_time": _mean(use, "charging_time"),
                })
    fam_path = tables_dir / "family_objective_comparison.csv"
    _write_csv(fam_path, fam_rows)
    written.append(fam_path)

    per = []
    for tw, *_ in TW:
        for mode, label in MODES:
            for r in data[mode][tw]:
                per.append({"objective": label, "tw_variant": tw, **r})
    per_path = tables_dir / "per_instance_objective_supplement.csv"
    _write_csv(per_path, per)
    written.append(per_path)

    if sto:
        sto_rows = []
        for tw, *_ in TW:
            for mode, label in MODES:
                block = ((sto.get(mode) or {}).get(tw) or {})
                if not block:
                    continue
                sto_rows.append({"objective": label, "tw_variant": tw, **block})
        if sto_rows:
            sp = tables_dir / "stochastic_robustness.csv"
            _write_csv(sp, sto_rows)
            written.append(sp)
    return written


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def fig01_objective_comparison(out: Path, data: dict, sto: dict | None) -> list[Path]:
    metrics = [
        ("used_k", "Vehicles", "a"),
        ("distance", "Distance", "b"),
        ("expected_charging_cost", "Expected charging cost", "c"),
        ("station_waiting_time", "Station waiting time", "d"),
        ("charging_time", "Charging time", "e"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4))
    axes = axes.ravel()
    x = np.arange(len(TW))
    width = 0.24
    for ax, (key, ylab, letter) in zip(axes, metrics):
        for i, (mode, _lab) in enumerate(MODES):
            vals = [_mean(data[mode][tw], key) for tw, *_ in TW]
            ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[mode], label=_lab if letter == "a" else None)
        ax.set_xticks(x)
        ax.set_xticklabels(["Original TW", "Relaxed TW"])
        ax.set_ylabel(ylab)
        ax.set_title(f"({letter})")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax = axes[5]
    if sto:
        for i, (mode, _lab) in enumerate(MODES):
            vals = [float(((sto.get(mode) or {}).get(tw) or {}).get("tw_reliability") or np.nan) for tw, *_ in TW]
            ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[mode], label=_lab)
        ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(["Original TW", "Relaxed TW"])
    ax.set_ylabel("TW reliability")
    ax.set_title("(f)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, [lab for _, lab in MODES], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Operational performance under alternative routing objectives", y=1.08, fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, out / "fig01_objective_comparison")


def fig02_family_performance(out: Path, data: dict) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    panels = [
        (0, 0, "original_tw", "used_k", "Vehicles — Original TW"),
        (0, 1, "original_tw", "distance", "Distance — Original TW"),
        (0, 2, "original_tw", "expected_charging_cost", "Charging cost — Original TW"),
        (1, 0, "relaxed_tw", "used_k", "Vehicles — Relaxed TW"),
        (1, 1, "relaxed_tw", "distance", "Distance — Relaxed TW"),
        (1, 2, "relaxed_tw", "station_waiting_time", "Station waiting — Relaxed TW"),
    ]
    x = np.arange(len(FAMILIES))
    width = 0.24
    for r, c, tw, key, title in panels:
        ax = axes[r][c]
        for i, (mode, lab) in enumerate(MODES):
            vals = []
            for fam in FAMILIES:
                sub = [row for row in data[mode][tw] if _family(str(row.get("instance", ""))) == fam]
                vals.append(_mean(sub, key))
            ax.bar(x + (i - 1) * width, vals, width=width, color=COLORS[mode], label=lab if r == 0 and c == 0 else None)
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILIES, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.legend([lab for _, lab in MODES], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Performance across instance families", y=1.08, fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, out / "fig02_family_performance")


def fig03_representative_map(out: Path, exp_dir: Path, mode: str) -> list[Path]:
    routes = _routes(exp_dir, mode, "original_tw", REPRESENTATIVE)
    if not routes:
        return []
    inst = _instance("A", REPRESENTATIVE)
    xs = [float(n.x) for n in inst.nodes]
    ys = [float(n.y) for n in inst.nodes]
    cust = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    st = list(inst.charging_station_indices)
    depot = int(inst.depot_index)
    fig, ax = plt.subplots(figsize=(7.2, 7.0))
    ax.scatter([xs[i] for i in cust], [ys[i] for i in cust], s=4, c="#94a3b8", alpha=0.35, linewidths=0, label="Customers")
    ax.scatter([xs[i] for i in st], [ys[i] for i in st], s=18, c="#b45309", marker="^", alpha=0.55, label="Charging stations")
    cmap = plt.get_cmap("tab20")
    used_st = set()
    n_draw = 0
    for j, (nodes, _lab) in enumerate(routes):
        if not any(inst.nodes[n].is_customer for n in nodes):
            continue
        n_draw += 1
        ax.plot([xs[n] for n in nodes], [ys[n] for n in nodes], color=cmap(j % 20), lw=0.7, alpha=0.55)
        for n in nodes:
            if inst.nodes[n].is_charging_station:
                used_st.add(int(n))
    if used_st:
        ax.scatter([xs[i] for i in used_st], [ys[i] for i in used_st], s=36, c="#b45309", marker="^",
                   edgecolors="k", linewidths=0.4, zorder=5, label="Charging visits")
    ax.scatter([xs[depot]], [ys[depot]], s=140, c="#111111", marker="*", zorder=6, label="Depot")
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Representative EVRPTW solution")
    ax.legend(loc="upper right", fontsize=7, frameon=False)
    axin = inset_axes(ax, width="38%", height="38%", loc="lower left", borderpad=1.2)
    span = 70.0
    axin.scatter([xs[i] for i in cust], [ys[i] for i in cust], s=3, c="#94a3b8", alpha=0.4, linewidths=0)
    axin.scatter([xs[i] for i in st], [ys[i] for i in st], s=12, c="#b45309", marker="^", alpha=0.6)
    for j, (nodes, _lab) in enumerate(routes):
        if not any(inst.nodes[n].is_customer for n in nodes):
            continue
        axin.plot([xs[n] for n in nodes], [ys[n] for n in nodes], color=cmap(j % 20), lw=0.6, alpha=0.6)
    axin.scatter([xs[depot]], [ys[depot]], s=80, c="#111111", marker="*")
    axin.set_xlim(xs[depot] - span, xs[depot] + span)
    axin.set_ylim(ys[depot] - span, ys[depot] + span)
    axin.set_xticks([])
    axin.set_yticks([])
    axin.set_aspect("equal")
    return _save_fig(fig, out / "fig03_representative_map")


def fig04_battery_soc(out: Path, exp_dir: Path, mode: str) -> list[Path]:
    routes = _routes(exp_dir, mode, "original_tw", REPRESENTATIVE)
    if not routes:
        return []
    inst = _instance("A", REPRESENTATIVE)
    q = float(inst.vehicle.battery_capacity)
    charged = []
    for i, (nodes, labels) in enumerate(routes):
        if not any(inst.nodes[n].is_charging_station for n in nodes):
            continue
        rr = replay_route(inst, nodes, labels)
        if any(float(ev.charged_energy or 0.0) > 1e-9 for ev in rr.route_events):
            charged.append((i, rr))
        if len(charged) >= 4:
            break
    if not charged:
        return []
    fig, axes = plt.subplots(len(charged), 1, figsize=(8.6, 2.15 * len(charged)), sharex=True)
    if len(charged) == 1:
        axes = [axes]
    for ax, (ri, rr) in zip(axes, charged):
        times = [0.0]
        soc = [1.0]
        for ev in rr.route_events:
            times.append(float(ev.arrival))
            soc.append(float(ev.battery_before) / q)
            times.append(float(ev.departure))
            soc.append(float(ev.battery_after) / q)
            if float(ev.charged_energy or 0.0) > 1e-9:
                ax.axvspan(ev.arrival, ev.departure, color="#b45309", alpha=0.18)
                ax.scatter([ev.arrival], [ev.battery_before / q], c="#b45309", s=28, zorder=5, marker="v")
                ax.scatter([ev.departure], [ev.battery_after / q], c="#1f4e79", s=28, zorder=5, marker="^")
        ax.plot(times, soc, color="#1f4e79", lw=1.6)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("SoC")
        ax.set_title(f"Route {ri + 1}", loc="left", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[-1].set_xlabel("Time")
    fig.suptitle("Battery state-of-charge along representative routes", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, out / "fig04_battery_soc")


def fig05_time_composition(out: Path, exp_dir: Path, mode: str) -> list[Path]:
    labels = ["Original TW", "Relaxed TW"]
    keys = ["travel", "service", "customer_wait", "station_wait", "charging"]
    palette = ["#1f4e79", "#64748b", "#94a3b8", "#c45c26", "#b45309"]
    stacked = []
    for tw, _lab, var in TW:
        routes = _routes(exp_dir, mode, tw, REPRESENTATIVE)
        if not routes:
            stacked.append({k: 0.0 for k in keys})
            continue
        inst = _instance(var, REPRESENTATIVE)
        res = replay_solution(inst, routes, m=len(routes))
        service = 0.0
        for route in res.route_results:
            for ev in route.route_events:
                nd = inst.nodes[ev.node]
                if nd.is_customer:
                    service += float(nd.service_time)
        travel = max(0.0, float(res.duration) - service - float(res.customer_early_waiting_time)
                     - float(res.station_waiting_time) - float(res.charging_time))
        stacked.append({
            "travel": travel,
            "service": service,
            "customer_wait": float(res.customer_early_waiting_time),
            "station_wait": float(res.station_waiting_time),
            "charging": float(res.charging_time),
        })
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    x = np.arange(2)
    bottom = np.zeros(2)
    pretty = {
        "travel": "Travel time",
        "service": "Service time",
        "customer_wait": "Customer early waiting",
        "station_wait": "Station waiting",
        "charging": "Charging time",
    }
    for k, color in zip(keys, palette):
        vals = np.array([s[k] for s in stacked])
        ax.bar(x, vals, bottom=bottom, color=color, width=0.55, label=pretty[k])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Time")
    ax.set_title("Operational time composition")
    ax.legend(frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _save_fig(fig, out / "fig05_time_composition")


def fig06_station_selection(out: Path, exp_dir: Path, mode: str) -> list[Path]:
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4))
    attrs = [
        ("rate", "Charging rate"),
        ("price", "Expected price"),
        ("wait", "Expected waiting time"),
    ]
    for row, (tw, twlab, var) in enumerate(TW):
        inst = _instance(var, REPRESENTATIVE)
        all_s = list(inst.charging_station_indices)
        all_rate = [charging_rate_s(inst, s) for s in all_s]
        all_price = [expected_price_s(inst, s) for s in all_s]
        all_wait = [expected_wait_s(inst, s) for s in all_s]
        used = []
        routes = _routes(exp_dir, mode, tw, REPRESENTATIVE)
        if routes:
            res = replay_solution(inst, routes, m=len(routes))
            for route in res.route_results:
                for ev in route.route_events:
                    if float(ev.charged_energy or 0) > 1e-12 and inst.nodes[ev.node].is_charging_station:
                        used.append(int(ev.node))
        used_rate = [charging_rate_s(inst, s) for s in used] or [np.nan]
        used_price = [expected_price_s(inst, s) for s in used] or [np.nan]
        used_wait = [expected_wait_s(inst, s) for s in used] or [np.nan]
        series = [(all_rate, used_rate), (all_price, used_price), (all_wait, used_wait)]
        for col, ((avail, sel), (_k, title)) in enumerate(zip(series, attrs)):
            ax = axes[row][col]
            parts = ax.violinplot([avail, sel], positions=[1, 2], showmeans=True, showextrema=False)
            for i, body in enumerate(parts["bodies"]):
                body.set_facecolor(C_EXP if i == 0 else C_RISK)
                body.set_alpha(0.65)
            ax.set_xticks([1, 2])
            ax.set_xticklabels(["All stations", "Selected"])
            ax.set_title(f"{title} — {twlab}", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    fig.suptitle("Charging-station characteristics and utilization", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, out / "fig06_station_selection")


def fig07_stochastic_robustness(out: Path, sto: dict | None) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.8))
    if not sto:
        axes[0].text(0.5, 0.5, "Stochastic evaluation not yet available", ha="center")
        fig.suptitle("Robustness to charging-station uncertainty")
        return _save_fig(fig, out / "fig07_stochastic_robustness")
    x = np.arange(len(MODES))
    width = 0.36
    # waiting
    ax = axes[0]
    for j, (tw, twlab, _v) in enumerate(TW):
        means = [float(((sto.get(m) or {}).get(tw) or {}).get("mean_station_waiting") or np.nan) for m, _ in MODES]
        p95 = [float(((sto.get(m) or {}).get(tw) or {}).get("p95_station_waiting") or np.nan) for m, _ in MODES]
        cvar = [float(((sto.get(m) or {}).get(tw) or {}).get("cvar95_station_waiting") or np.nan) for m, _ in MODES]
        offset = (j - 0.5) * 0.08
        ax.plot(x + offset, means, marker="o", color=C_EXP if j == 0 else C_RISK, label=f"Mean {twlab}")
        ax.plot(x + offset, p95, marker="s", linestyle="--", color=C_EXP if j == 0 else C_RISK, label=f"P95 {twlab}")
        ax.plot(x + offset, cvar, marker="^", linestyle=":", color=C_EXP if j == 0 else C_RISK, label=f"CVaR95 {twlab}")
    ax.set_xticks(x)
    ax.set_xticklabels(["Classical", "Expected", "Risk-aware"], fontsize=8)
    ax.set_title("(a) Station waiting")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[1]
    for j, (tw, twlab, _v) in enumerate(TW):
        means = [float(((sto.get(m) or {}).get(tw) or {}).get("mean_charging_expense") or np.nan) for m, _ in MODES]
        p95 = [float(((sto.get(m) or {}).get(tw) or {}).get("p95_charging_expense") or np.nan) for m, _ in MODES]
        cvar = [float(((sto.get(m) or {}).get(tw) or {}).get("cvar95_charging_expense") or np.nan) for m, _ in MODES]
        offset = (j - 0.5) * 0.08
        ax.plot(x + offset, means, marker="o", color=C_EXP if j == 0 else C_RISK, label=twlab)
        ax.plot(x + offset, p95, marker="s", linestyle="--", color=C_EXP if j == 0 else C_RISK)
        ax.plot(x + offset, cvar, marker="^", linestyle=":", color=C_EXP if j == 0 else C_RISK)
    ax.set_xticks(x)
    ax.set_xticklabels(["Classical", "Expected", "Risk-aware"], fontsize=8)
    ax.set_title("(b) Charging expense")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax = axes[2]
    width = 0.36
    for j, (tw, twlab, _v) in enumerate(TW):
        vals = [float(((sto.get(m) or {}).get(tw) or {}).get("tw_reliability") or np.nan) for m, _ in MODES]
        ax.bar(x + (j - 0.5) * width, vals, width=width, color=C_EXP if j == 0 else C_RISK, label=twlab)
    ax.set_xticks(x)
    ax.set_xticklabels(["Classical", "Expected", "Risk-aware"], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("(c) TW reliability")
    ax.legend(frameon=False, fontsize=7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Robustness to charging-station uncertainty", fontsize=13)
    fig.tight_layout()
    return _save_fig(fig, out / "fig07_stochastic_robustness")


def evaluate_stochastic_all(exp_dir: Path, data: dict, *, n_scenarios: int, names: list[str]) -> dict:
    """CRN: same scenario seed for A/B/C on each instance."""
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    acc: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for tw, _lab, var in TW:
        for name in names:
            inst = _instance(var, name)
            for mode, _ in MODES:
                routes = _routes(exp_dir, mode, tw, name)
                if not routes:
                    continue
                summary = evaluate_solution_stochastic(
                    inst, routes, n_scenarios=n_scenarios, seed=0,
                )
                acc[(mode, tw)].append(summary.as_dict())
    for (mode, tw), rows in acc.items():
        keys = rows[0].keys()
        out[mode][tw] = {k: float(np.mean([float(r[k]) for r in rows])) for k in keys}
    return dict(out)


def proposed_mode(data: dict, sto: dict | None = None) -> str:
    """C is proposed only if fleet/feasibility hold and station metrics improve."""
    def feas(mode: str) -> float:
        vals = []
        for tw, *_ in TW:
            rows = data[mode][tw]
            if rows:
                vals.append(sum(1 for r in rows if r.get("feasible")) / max(len(rows), 1))
        return float(np.mean(vals)) if vals else 0.0

    def veh(mode: str) -> float:
        return float(np.nanmean([_mean(data[mode][tw], "used_k") for tw, *_ in TW if data[mode][tw]]))

    def cost(mode: str) -> float:
        return float(np.nanmean([_mean(data[mode][tw], "expected_charging_cost") for tw, *_ in TW if data[mode][tw]]))

    def wait(mode: str) -> float:
        return float(np.nanmean([_mean(data[mode][tw], "station_waiting_time") for tw, *_ in TW if data[mode][tw]]))

    def rel(mode: str) -> float:
        if not sto:
            return float("nan")
        vals = [float(((sto.get(mode) or {}).get(tw) or {}).get("tw_reliability") or np.nan) for tw, *_ in TW]
        return float(np.nanmean(vals))

    if not any(data["classical"][tw] for tw, *_ in TW):
        return "classical"
    f_c, k_c, c_c, w_c, r_c = feas("classical"), veh("classical"), cost("classical"), wait("classical"), rel("classical")

    def acceptable(mode: str) -> bool:
        if not any(data[mode][tw] for tw, *_ in TW):
            return False
        if feas(mode) + 1e-9 < f_c:
            return False
        if veh(mode) > k_c + 0.5:
            return False
        return True

    def improves(mode: str) -> bool:
        better = False
        for tw, *_ in TW:
            rows_m = data[mode][tw]
            rows_c = data["classical"][tw]
            if not rows_m or not rows_c:
                continue
            if _mean(rows_m, "expected_charging_cost") < _mean(rows_c, "expected_charging_cost") - 1.0:
                better = True
            if _mean(rows_m, "station_waiting_time") < _mean(rows_c, "station_waiting_time") - 1.0:
                better = True
            if _mean(rows_m, "charging_time") < _mean(rows_c, "charging_time") - 1.0:
                better = True
            if sto:
                r_m = float(((sto.get(mode) or {}).get(tw) or {}).get("tw_reliability") or np.nan)
                r_c = float(((sto.get("classical") or {}).get(tw) or {}).get("tw_reliability") or np.nan)
                if not np.isnan(r_m) and not np.isnan(r_c):
                    if r_m > r_c + 0.01:
                        better = True
                    if r_m < r_c - 0.05:
                        return False
        return better

    if acceptable("risk_aware") and improves("risk_aware"):
        return "risk_aware"
    if acceptable("expected") and improves("expected"):
        return "expected"
    return "classical"


def generate_all(
    exp_dir: Path,
    *,
    figures_dir: Path,
    tables_dir: Path,
    n_scenarios: int = 100,
    names: list[str] | None = None,
) -> dict[str, Any]:
    data = load_mode_tables(exp_dir)
    names = names or sorted({
        str(r["instance"]) for mode, _ in MODES for tw, *_ in TW for r in data[mode][tw]
    })
    sto = None
    if names and any(data[m][tw] for m, _ in MODES for tw, *_ in TW):
        try:
            sto = evaluate_stochastic_all(exp_dir, data, n_scenarios=n_scenarios, names=names)
        except Exception as exc:
            sto = None
            print(f"stochastic eval skipped: {exc}", flush=True)
    written_tables = write_tables(exp_dir, tables_dir, data, sto)
    mode = proposed_mode(data, sto)
    figs: list[Path] = []
    figs.extend(fig01_objective_comparison(figures_dir, data, sto))
    figs.extend(fig02_family_performance(figures_dir, data))
    figs.extend(fig03_representative_map(figures_dir, exp_dir, mode))
    figs.extend(fig04_battery_soc(figures_dir, exp_dir, mode))
    figs.extend(fig05_time_composition(figures_dir, exp_dir, mode))
    figs.extend(fig06_station_selection(figures_dir, exp_dir, mode))
    figs.extend(fig07_stochastic_robustness(figures_dir, sto))
    return {
        "tables": [str(p).replace("\\", "/") for p in written_tables],
        "figures": [str(p).replace("\\", "/") for p in figs],
        "proposed_mode": mode,
        "stochastic_scenarios": n_scenarios if sto else 0,
        "evaluation": "post-optimization stochastic robustness evaluation",
        "chance_constrained": False,
    }
