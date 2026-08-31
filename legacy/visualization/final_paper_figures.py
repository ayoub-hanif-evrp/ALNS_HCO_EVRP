"""One-figure / one-message EVRPTW paper figures.

No subplot grids. No dataset/seed/warm-start titles.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from src.data.hanif_convert import VARIANT_SPECS
from src.data.instance_loader import load_instance
from src.evaluation.paper_objective import charging_rate_s, expected_wait_s
from src.search.route_replay import replay_route, replay_solution

ROOT = Path(__file__).resolve().parents[2]
DPI = 300
C_A = "#1f4e79"
C_B = "#c45c26"
C_GRAY = "#94a3b8"
C_DARK = "#1e293b"
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]


def _style(ax) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


def _save(fig, stem: Path) -> list[Path]:
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
        rows = []
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for k, v in raw.items():
                try:
                    if v is None or v == "":
                        continue
                    if v.lower() in {"true", "false"}:
                        row[k] = v.lower() == "true"
                    elif any(c in v for c in ".eE"):
                        row[k] = float(v)
                    else:
                        row[k] = int(v)
                except (ValueError, AttributeError):
                    row[k] = v
            rows.append(row)
        return rows


def _routes(exp: Path, method: str, tw: str, name: str):
    path = exp / method / tw / "routes" / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(list(r["nodes"]), list(r.get("labels") or [None] * len(r["nodes"]))) for r in payload["routes"]]


def load_results(exp: Path) -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {}
    for method in ("classical", "proposed"):
        out[method] = {
            "original_tw": _read_csv(exp / method / "original_tw" / "results.csv"),
            "relaxed_tw": _read_csv(exp / method / "relaxed_tw" / "results.csv"),
        }
    return out


def select_representative(rows: list[dict]) -> str | None:
    feas = [r for r in rows if r.get("feasible")]
    if not feas:
        return None
    ks = [float(r["used_k"]) for r in feas]
    med = float(np.median(ks))
    feas.sort(key=lambda r: (abs(float(r["used_k"]) - med), str(r["instance"])))
    return str(feas[0]["instance"])


def _vals(rows: list[dict], key: str) -> list[float]:
    return [float(r[key]) for r in rows if key in r and r.get(key) not in (None, "")]


def fig01_fleet(out: Path, proposed: dict[str, list[dict]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    data = [_vals(proposed["original_tw"], "used_k"), _vals(proposed["relaxed_tw"], "used_k")]
    try:
        bp = ax.boxplot(data, tick_labels=["Original TW", "Relaxed TW"], patch_artist=True, widths=0.55)
    except TypeError:
        bp = ax.boxplot(data, labels=["Original TW", "Relaxed TW"], patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], [C_A, C_B]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(C_DARK)
    for med in bp["medians"]:
        med.set_color(C_DARK)
    ax.set_ylabel("Used electric vehicles")
    ax.set_title("Fleet size under the two time-window settings", fontsize=10)
    _style(ax)
    return _save(fig, out / "fig01_fleet_size")


def fig02_distance(out: Path, proposed: dict[str, list[dict]]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    data = [_vals(proposed["original_tw"], "distance"), _vals(proposed["relaxed_tw"], "distance")]
    try:
        bp = ax.boxplot(data, tick_labels=["Original TW", "Relaxed TW"], patch_artist=True, widths=0.55)
    except TypeError:
        bp = ax.boxplot(data, labels=["Original TW", "Relaxed TW"], patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], [C_A, C_B]):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(C_DARK)
    for med in bp["medians"]:
        med.set_color(C_DARK)
    ax.set_ylabel("Travel distance")
    ax.set_title("Travel distance under the two time-window settings", fontsize=10)
    _style(ax)
    return _save(fig, out / "fig02_distance")


def _pick_highlight_routes(instance, routes) -> list[int]:
    nonempty = [
        i for i, (nodes, _) in enumerate(routes)
        if any(instance.nodes[n].is_customer for n in nodes)
    ]
    charged = [
        i for i in nonempty
        if any(instance.nodes[n].is_charging_station for n in routes[i][0])
    ]
    chosen = []
    if charged:
        charged.sort(key=lambda i: len(routes[i][0]))
        chosen.append(charged[len(charged) // 2])
        most = max(charged, key=lambda i: sum(1 for n in routes[i][0] if instance.nodes[n].is_charging_station))
        if most not in chosen:
            chosen.append(most)
    for i in nonempty:
        if i not in chosen:
            chosen.append(i)
        if len(chosen) >= 3:
            break
    return chosen[:3]


def fig03_map(out: Path, exp: Path, name: str, meta: dict) -> list[Path]:
    routes = _routes(exp, "proposed", "original_tw", name)
    if not routes:
        return []
    inst = load_instance(Path(VARIANT_SPECS["A"]["output"]) / name)
    xs = [float(n.x) for n in inst.nodes]
    ys = [float(n.y) for n in inst.nodes]
    cust = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    st = list(inst.charging_station_indices)
    depot = int(inst.depot_index)
    highlight = _pick_highlight_routes(inst, routes)
    meta["representative_instance"] = name
    meta["highlighted_routes"] = highlight
    fig, ax = plt.subplots(figsize=(7.0, 6.6))
    ax.scatter([xs[i] for i in cust], [ys[i] for i in cust], s=3, c=C_GRAY, alpha=0.35, linewidths=0, zorder=1)
    ax.scatter([xs[i] for i in st], [ys[i] for i in st], s=6, c="#d6b89c", marker="^", alpha=0.22, zorder=2)
    colors = [C_A, C_B, "#2f6b4f"]
    visited = set()
    for j, ri in enumerate(highlight):
        nodes, _ = routes[ri]
        ax.plot([xs[n] for n in nodes], [ys[n] for n in nodes], color=colors[j % 3], lw=1.8, zorder=3)
        for n in nodes:
            if inst.nodes[n].is_charging_station:
                visited.add(int(n))
    if visited:
        ax.scatter([xs[i] for i in visited], [ys[i] for i in visited], s=36, c=C_B, marker="^",
                   edgecolors="k", linewidths=0.4, zorder=4)
    ax.scatter([xs[depot]], [ys[depot]], s=160, c="k", marker="*", zorder=5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Representative electric-vehicle routing and charging decisions", fontsize=10)
    _style(ax)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_GRAY, markersize=5, label="Customer"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#d6b89c", markersize=7, label="Charging station"),
            Line2D([0], [0], marker="*", color="w", markerfacecolor="k", markersize=12, label="Depot"),
            Line2D([0], [0], color=C_A, lw=1.8, label="Selected route"),
        ],
        frameon=False, fontsize=8, loc="upper right",
    )
    return _save(fig, out / "fig03_representative_route")


def fig04_soc(out: Path, exp: Path, name: str, meta: dict) -> list[Path]:
    routes = _routes(exp, "proposed", "original_tw", name)
    if not routes:
        return []
    inst = load_instance(Path(VARIANT_SPECS["A"]["output"]) / name)
    q = float(inst.vehicle.battery_capacity)
    nonempty = [
        i for i, (nodes, _) in enumerate(routes)
        if any(inst.nodes[n].is_customer for n in nodes)
    ]
    charged = [
        i for i in nonempty
        if any(inst.nodes[n].is_charging_station for n in routes[i][0])
    ]
    if charged:
        ri = max(charged, key=lambda i: sum(1 for n in routes[i][0] if inst.nodes[n].is_charging_station))
    else:
        ri = nonempty[0] if nonempty else 0
    meta["soc_routes"] = [ri]
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    nodes, labels = routes[ri]
    rr = replay_route(inst, nodes, labels)
    t, soc = [0.0], [100.0]
    for ev in rr.route_events:
        t.append(float(ev.arrival))
        soc.append(100.0 * float(ev.battery_before) / q)
        t.append(float(ev.departure))
        soc.append(100.0 * float(ev.battery_after) / q)
        nd = inst.nodes[ev.node]
        if nd.is_customer:
            ax.scatter([ev.arrival], [100.0 * ev.battery_before / q], s=14, c=C_A, zorder=4)
        if float(ev.charged_energy or 0) > 1e-9 and float(ev.battery_after) > float(ev.battery_before) + 1e-9:
            ax.scatter([ev.arrival], [100.0 * ev.battery_before / q], s=36, marker="v", c=C_B, zorder=5)
            ax.scatter([ev.departure], [100.0 * ev.battery_after / q], s=36, marker="^", c="#2f6b4f", zorder=5)
    ax.plot(t, soc, "-", color=C_A, lw=1.8, label=f"Route {ri + 1}")
    ax.set_ylim(-2, 105)
    ax.set_xlabel("Elapsed route time")
    ax.set_ylabel("State of charge (%)")
    ax.set_title("Battery state of charge along a charging-intensive route", fontsize=10)
    ax.legend(
        handles=[
            Line2D([0], [0], color=C_A, lw=1.8, label=f"Route {ri + 1}"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor=C_A, markersize=6, label="Customer arrival"),
            Line2D([0], [0], marker="v", color="w", markerfacecolor=C_B, markersize=7, label="SoC before charging"),
            Line2D([0], [0], marker="^", color="w", markerfacecolor="#2f6b4f", markersize=7, label="SoC after charging"),
        ],
        frameon=False, fontsize=8,
    )
    _style(ax)
    return _save(fig, out / "fig04_soc_profile")


def fig05_stations(out: Path, exp: Path, name: str) -> list[Path]:
    routes = _routes(exp, "proposed", "original_tw", name)
    if not routes:
        return []
    inst = load_instance(Path(VARIANT_SPECS["A"]["output"]) / name)
    res = replay_solution(inst, routes, m=len(routes))
    used: dict[int, int] = {}
    for route in res.route_results:
        for ev in route.route_events:
            if float(ev.charged_energy or 0) > 1e-12 and inst.nodes[ev.node].is_charging_station:
                used[int(ev.node)] = used.get(int(ev.node), 0) + 1
    all_s = list(inst.charging_station_indices)
    waits = [expected_wait_s(inst, s) for s in all_s]
    visits = [float(used.get(s, 0)) for s in all_s]
    rates = [charging_rate_s(inst, s) for s in all_s]
    rmax = max(rates) if rates else 1.0
    sizes = [18 + 70 * (r / rmax) for r in rates]
    fig, ax = plt.subplots(figsize=(3.6, 3.4))
    unused = [i for i, s in enumerate(all_s) if used.get(s, 0) == 0]
    selected = [i for i, s in enumerate(all_s) if used.get(s, 0) > 0]
    if unused:
        ax.scatter(
            [waits[i] for i in unused], [visits[i] for i in unused],
            s=[sizes[i] for i in unused], c=C_GRAY, alpha=0.35, linewidths=0, label="Unused",
        )
    if selected:
        ax.scatter(
            [waits[i] for i in selected], [visits[i] for i in selected],
            s=[sizes[i] for i in selected], c=C_B, edgecolors="k", linewidths=0.4,
            label="Used", zorder=3,
        )
    ax.set_xlabel("Expected station waiting time")
    ax.set_ylabel("Charging visits")
    ax.set_title("Charging visits versus expected station waiting", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, out / "fig05_station_selection")


def fig06_time(out: Path, data: dict[str, dict[str, list[dict]]]) -> list[Path]:
    keys = ["travel_time", "service_time", "customer_early_waiting_time", "station_waiting_time", "charging_time"]
    pretty = {
        "travel_time": "Travel",
        "service_time": "Service",
        "customer_early_waiting_time": "Customer waiting",
        "station_waiting_time": "Station waiting",
        "charging_time": "Charging",
    }
    colors = [C_A, C_GRAY, "#cbd5e1", C_B, "#8a5a2b"]
    stacked = []
    for tw in ("original_tw", "relaxed_tw"):
        rows = data.get("proposed", {}).get(tw) or []
        totals = {k: sum(float(r.get(k) or 0.0) for r in rows) for k in keys}
        tot = sum(max(v, 0.0) for v in totals.values()) or 1.0
        stacked.append({k: 100.0 * max(totals[k], 0.0) / tot for k in keys})
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    x = np.arange(2)
    bottom = np.zeros(2)
    for k, color in zip(keys, colors):
        vals = np.array([s[k] for s in stacked])
        ax.bar(x, vals, bottom=bottom, color=color, width=0.55, label=pretty[k])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["Original TW", "Relaxed TW"])
    ax.set_ylabel("Share of operating time (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Composition of vehicle operating time", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    _style(ax)
    return _save(fig, out / "fig06_time_composition")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _stats(rows: list[dict], key: str) -> tuple[float, float]:
    xs = np.array(_vals(rows, key), dtype=float)
    if xs.size == 0:
        return float("nan"), float("nan")
    return float(np.mean(xs)), float(np.std(xs, ddof=0))


def write_tables(tables: Path, data: dict) -> None:
    t1 = []
    for method, label in (("classical", "Classical"), ("proposed", "Proposed")):
        for tw, twlab in (("original_tw", "Original TW"), ("relaxed_tw", "Relaxed TW")):
            rows = data[method][tw]
            feas = [r for r in rows if r.get("feasible")]
            use = feas or rows
            def pair(key):
                m, s = _stats(use, key)
                return m, s
            t1.append({
                "row": f"{label} — {twlab}",
                "feasible": sum(1 for r in rows if r.get("feasible")),
                "n": len(rows),
                "served_pct_mean": 100.0 * _stats(use, "served_ratio")[0] if any("served_ratio" in r for r in use) else 100.0 * np.mean([
                    float(r.get("served") or 0) / max(float(r.get("n_customers") or 1), 1) for r in use
                ]),
                "vehicles_mean": pair("used_k")[0], "vehicles_std": pair("used_k")[1],
                "distance_mean": pair("distance")[0], "distance_std": pair("distance")[1],
                "charging_visits_mean": pair("charging_visits")[0],
                "charged_energy_mean": pair("charged_energy")[0] if use and "charged_energy" in use[0] else pair("energy_consumed")[0],
                "charging_time_mean": pair("charging_time")[0],
                "station_waiting_mean": pair("station_waiting_time")[0],
                "customer_waiting_mean": pair("customer_early_waiting_time")[0],
                "charging_expense_mean": pair("expected_charging_cost")[0],
                "minimum_soc_mean": pair("minimum_soc")[0],
                "runtime_mean": pair("runtime_s")[0], "runtime_std": pair("runtime_s")[1],
            })
    _write_csv(tables / "table1_overall.csv", t1)

    t2 = []
    for tw, twlab in (("original_tw", "Original TW"), ("relaxed_tw", "Relaxed TW")):
        for fam in FAMILIES:
            for method, label in (("classical", "Classical"), ("proposed", "Proposed")):
                sub = [r for r in data[method][tw] if str(r.get("instance", "")).split("_")[0] == fam]
                t2.append({
                    "tw_variant": twlab,
                    "family": fam,
                    "method": label,
                    "n": len(sub),
                    "vehicles": _stats(sub, "used_k")[0],
                    "distance": _stats(sub, "distance")[0],
                    "charging_visits": _stats(sub, "charging_visits")[0],
                    "charging_time": _stats(sub, "charging_time")[0],
                    "station_waiting": _stats(sub, "station_waiting_time")[0],
                    "charging_expense": _stats(sub, "expected_charging_cost")[0],
                    "runtime": _stats(sub, "runtime_s")[0],
                })
    _write_csv(tables / "table2_family.csv", t2)

    t3 = []
    for tw, twlab in (("original_tw", "Original TW"), ("relaxed_tw", "Relaxed TW")):
        c, p = data["classical"][tw], data["proposed"][tw]
        def delta(key):
            ac, _ = _stats(c, key)
            bc, _ = _stats(p, key)
            abs_d = bc - ac
            pct = 0.0 if abs(ac) < 1e-9 else 100.0 * abs_d / ac
            return abs_d, pct
        dk, _ = delta("used_k")
        t3.append({
            "tw_variant": twlab,
            "Delta_vehicles": dk,
            "Delta_distance_%": delta("distance")[1],
            "Delta_station_waiting_%": delta("station_waiting_time")[1],
            "Delta_charging_time_%": delta("charging_time")[1],
            "Delta_charging_expense_%": delta("expected_charging_cost")[1],
        })
    _write_csv(tables / "table3_objective_comparison.csv", t3)

    per = []
    for method, label in (("classical", "Classical"), ("proposed", "Proposed")):
        for tw, twlab in (("original_tw", "Original TW"), ("relaxed_tw", "Relaxed TW")):
            for r in data[method][tw]:
                per.append({"method": label, "tw_variant": twlab, **r})
    _write_csv(tables / "per_instance.csv", per)


def generate(exp: Path, paper: Path, *, frozen_final: bool = False) -> dict[str, Any]:
    data = load_results(exp)
    figures = paper / "figures"
    tables = paper / "tables"
    for d in (figures, tables):
        d.mkdir(parents=True, exist_ok=True)
    write_tables(tables, data)
    meta: dict[str, Any] = {
        "representative_rule": "Proposed Original-TW feasible instance whose fleet size is closest to the median; ties broken by instance name.",
        "soc_route_rule": "Within that instance: the route with the most charging visits.",
        "time_composition_rule": "Shares are summed over all Proposed instances in each TW setting.",
        "frozen_final": bool(frozen_final),
    }
    name = select_representative(data["proposed"]["original_tw"])
    saved = []
    saved += fig01_fleet(figures, data["proposed"])
    saved += fig02_distance(figures, data["proposed"])
    saved += fig06_time(figures, data)
    if name:
        saved += fig03_map(figures, exp, name, meta)
        saved += fig04_soc(figures, exp, name, meta)
        saved += fig05_stations(figures, exp, name)
        (paper / "representative_routes.json").write_text(
            json.dumps(meta, indent=2) + "\n", encoding="utf-8",
        )
    (paper / "manuscript_values.json").write_text(json.dumps({
        "experiment": str(exp).replace("\\", "/"),
        "representative_instance": name,
        "n_main_figures": 6,
        "frozen_final": bool(frozen_final),
        "titles": [
            "Fleet size under the two time-window settings",
            "Travel distance under the two time-window settings",
            "Representative electric-vehicle routing and charging decisions",
            "Battery state of charge along a charging-intensive route",
            "Charging visits versus expected station waiting",
            "Composition of vehicle operating time",
        ],
        **meta,
    }, indent=2) + "\n", encoding="utf-8")
    return {"figures": [str(p).replace("\\", "/") for p in saved], "representative": name, **meta}
