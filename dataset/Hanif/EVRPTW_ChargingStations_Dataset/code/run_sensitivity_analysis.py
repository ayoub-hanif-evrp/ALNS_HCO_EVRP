# OAT sensitivity on generation params: regenerate subsets, run HC, write CSV and optional matplotlib plots.

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import shutil
from copy import deepcopy

from algorithm1_stations import generate_stations, write_stations_csv
from algorithm2_relax_tw import relax_time_windows
from evrptw_solvers import (
    load_ev_specs,
    load_station_rates,
    parse_instance,
    prepare_instance,
    run_meta_solver,
)
from utils import (
    INPUT_DIR,
    euclidean_distance,
    get_instance_files,
    parse_instance as parse_homberger,
    write_instance,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_PARAMS_PATH = os.path.join(_ROOT, "params.json")
_OUTPUT_ROOT = os.path.join(_ROOT, "sensitivity_runs")
_RESULTS_CSV = os.path.join(_SCRIPT_DIR, "sensitivity_analysis_results.csv")
_PLOTS_DIR = os.path.join(_ROOT, "sensitivity_graph_results")

INSTANCE_CLASSES = ("C1", "C2", "R1", "R2", "RC1", "RC2")

VARIANT_A_NAME = "variant_A_original_TW"
VARIANT_B_NAME = "variant_B_relaxed_TW"


# Take the first per_class sorted .txt names for each Homberger class prefix (C1_, C2_, …).
def select_instances_per_class(all_files: list[str], per_class: int) -> list[str]:
    sorted_files = sorted(all_files)
    selected: list[str] = []
    for cls in INSTANCE_CLASSES:
        prefix = cls + "_"
        matches = [f for f in sorted_files if f.startswith(prefix)]
        selected.extend(matches[:per_class])
    return selected


# Load root params.json.
def load_default_params():
    with open(_PARAMS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# Widen or tighten r/p/w intervals around their midpoints by width_factor (copy of params).
def scale_attr_ranges(params: dict, width_factor: float) -> dict:
    out = deepcopy(params)
    for lo_k, hi_k in (("r_min", "r_max"), ("p_min", "p_max"), ("w_min", "w_max")):
        lo, hi = float(out[lo_k]), float(out[hi_k])
        mid = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * width_factor
        nlo, nhi = mid - half, mid + half
        if lo_k == "p_min":
            nlo = max(0.01, nlo)
            nhi = max(nlo + 1e-3, nhi)
        if lo_k == "w_min":
            nlo = max(0.01, nlo)
            nhi = max(nlo + 1e-3, nhi)
        if lo_k == "r_min":
            nlo = max(0.01, nlo)
            nhi = max(nlo + 1e-3, nhi)
        out[lo_k], out[hi_k] = nlo, nhi
    return out


# Flat metadata + parameter snapshot for one sensitivity case row in the results CSV.
def param_row(case_id: str, oat_group: str, level: str, p: dict) -> dict:
    return {
        "case_id": case_id,
        "oat_group": oat_group,
        "oat_level": level,
        "seed": p["seed"],
        "num_stations": p["num_stations"],
        "d_sep": p["d_sep"],
        "R_factor": p["R_factor"],
        "K": p["K"],
        "r_min": p["r_min"],
        "r_max": p["r_max"],
        "p_min": p["p_min"],
        "p_max": p["p_max"],
        "w_min": p["w_min"],
        "w_max": p["w_max"],
        "cv_price": p["cv_price"],
        "cv_wait": p["cv_wait"],
        "R_near": p["R_near"],
        "R_medium": p["R_medium"],
        "R_far": p["R_far"],
        "attr_range_width_factor": p.get("_attr_width_factor", ""),
    }


# Generate variant-A .txt and _stations.csv for listed Homberger bases into out_variant_a.
def generate_variant_a(params: dict, homberger_dir: str, out_variant_a: str, fnames: list[str]):
    os.makedirs(out_variant_a, exist_ok=True)
    for fname in fnames:
        fpath = os.path.join(homberger_dir, fname)
        instance_name, vehicle_info, depot, customers = parse_homberger(fpath)
        stations, attributes = generate_stations(depot, customers, params)
        stem = os.path.splitext(fname)[0]
        out_txt = os.path.join(out_variant_a, stem + ".txt")
        out_csv = os.path.join(out_variant_a, stem + "_stations.csv")
        write_instance(out_txt, instance_name, vehicle_info, depot, customers, stations)
        write_stations_csv(out_csv, attributes)


# Run time-window relaxation from variant A to B with stdout suppressed.
def build_variant_b(variant_a_dir: str, variant_b_dir: str, params: dict):
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull):
            relax_time_windows(
                variant_a_dir,
                variant_b_dir,
                params["R_near"],
                params["R_medium"],
                params["R_far"],
            )


# Mean over stations of min distance from each station to any customer (Euclidean).
def mean_nn_station_customer_distance(txt_path: str) -> float | None:
    _, _, depot, customers = parse_homberger(txt_path)
    fleet_size, load_capacity, depot2, customers2, stations = parse_instance(txt_path)
    if not stations or not customers:
        return None
    cust_pts = [(c["x"], c["y"]) for c in customers]
    dists = []
    for s in stations:
        sx, sy = s[1], s[2]
        dmin = min(euclidean_distance(sx, sy, cx, cy) for cx, cy in cust_pts)
        dists.append(dmin)
    return sum(dists) / len(dists) if dists else None


# Pool charging_rate, wait_mu, price_mu, density_score, and NN distances across station CSV rows.
def aggregate_generation_stats(variant_a_dir: str, fnames: list[str]):
    rates, wmu, pmu, nnd, dscore = [], [], [], [], []
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        txt = os.path.join(variant_a_dir, stem + ".txt")
        csvp = os.path.join(variant_a_dir, stem + "_stations.csv")
        if not os.path.isfile(txt) or not os.path.isfile(csvp):
            continue
        with open(csvp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rates.append(float(row["charging_rate"]))
                wmu.append(float(row["wait_mu"]))
                pmu.append(float(row["price_mu"]))
                dscore.append(float(row["density_score"]))
        nn = mean_nn_station_customer_distance(txt)
        if nn is not None:
            nnd.append(nn)
    n_st = len(rates)
    return {
        "gen_n_station_rows": n_st,
        "gen_avg_charging_rate": sum(rates) / n_st if n_st else float("nan"),
        "gen_std_charging_rate": _pstdev(rates),
        "gen_avg_wait_mu": sum(wmu) / n_st if n_st else float("nan"),
        "gen_avg_price_mu": sum(pmu) / n_st if n_st else float("nan"),
        "gen_avg_density_score": sum(dscore) / n_st if n_st else float("nan"),
        "gen_avg_nn_station_customer": sum(nnd) / len(nnd) if nnd else float("nan"),
        "gen_std_nn_station_customer": _pstdev(nnd),
    }


# Sample standard deviation (pandas-style); nan if len(xs) < 2.
def _pstdev(xs: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return round(math.sqrt(v), 6)


# Run HC (run_meta_solver) on each instance in variant_dir for each Q in q_values; return detail rows.
def run_hc_on_folder(
    variant: str,
    variant_dir: str,
    fnames: list[str],
    q_values: list[int],
    energy_rate: float,
    progress_prefix: str = "",
):
    rows = []
    valid = []
    for fname in fnames:
        stem = os.path.splitext(fname)[0]
        txt = os.path.join(variant_dir, stem + ".txt")
        csvp = os.path.join(variant_dir, stem + "_stations.csv")
        if os.path.isfile(txt) and os.path.isfile(csvp):
            valid.append((fname, stem, txt, csvp))
    total_runs = len(valid) * len(q_values)
    run_i = 0
    for fname, stem, txt, csvp in valid:
        fleet_size, load_capacity, depot, customers, stations = parse_instance(txt)
        inst = prepare_instance(
            fleet_size,
            load_capacity,
            depot,
            customers,
            stations,
            load_station_rates(csvp),
        )
        for Q in q_values:
            run_i += 1
            print(
                f"{progress_prefix}    HC variant {variant} [{run_i}/{total_runs}] {stem} Q={Q} …",
                flush=True,
            )
            r = run_meta_solver(
                inst,
                "HC",
                instance_name=f"{variant}_{stem}",
                battery_capacity=float(Q),
                energy_rate=energy_rate,
                fleet_cap=None,
                fast=True,
                baseline_fast=True,
            )
            print(
                f"{progress_prefix}    HC variant {variant} [{run_i}/{total_runs}] {stem} Q={Q} "
                f"done  feasible={bool(r['feasible'])}  {r['seconds']}s",
                flush=True,
            )
            rows.append(
                {
                    "variant": variant,
                    "instance": stem,
                    "Q": Q,
                    "feasible": 1 if r["feasible"] else 0,
                    "num_routes": int(r["num_routes"]),
                    "num_charges": int(r["num_charges"]),
                    "seconds": float(r["seconds"]),
                    "total_distance": r["total_distance"],
                    "served": int(r["served"]),
                    "num_customers": int(inst["num_customers"]),
                }
            )
    return rows


# Mean feasible fraction, routes, charges, runtime, and mean distance over feasible runs.
def _summarize(rows: list[dict]):
    n = len(rows)
    if n == 0:
        return {}
    feas = sum(r["feasible"] for r in rows)
    avg_routes = sum(r["num_routes"] for r in rows) / n
    avg_chg = sum(r["num_charges"] for r in rows) / n
    avg_sec = sum(r["seconds"] for r in rows) / n
    fd = [r["total_distance"] for r in rows if r["feasible"] and r["total_distance"] != float("inf")]
    avg_dist = sum(fd) / len(fd) if fd else float("nan")
    served = [r["served"] for r in rows]
    return {
        "n_runs": n,
        "feasible_count": feas,
        "feasible_frac": round(feas / n, 6),
        "avg_routes": round(avg_routes, 4),
        "avg_charges": round(avg_chg, 4),
        "avg_runtime_s": round(avg_sec, 4),
        "avg_distance_feas": round(avg_dist, 4) if not math.isnan(avg_dist) else "",
        "avg_served": round(sum(served) / n, 4),
    }


# Split solver rows by variant A/B and add pooled hc_* keys for CSV export.
def summarize_by_variant(solver_rows: list[dict]):
    a = [r for r in solver_rows if r["variant"] == "A"]
    b = [r for r in solver_rows if r["variant"] == "B"]
    sa, sb = _summarize(a), _summarize(b)
    out = {}
    for k, v in sa.items():
        out[f"hc_{k}_A"] = v
    for k, v in sb.items():
        out[f"hc_{k}_B"] = v
    out.update({f"hc_{k}": v for k, v in _summarize(solver_rows).items()})
    return out


# One sensitivity case: wipe case_dir, regenerate A/B, aggregate gen stats, run HC on both variants.
def run_case(
    case_id: str,
    params: dict,
    homberger_dir: str,
    fnames: list[str],
    q_values: list[int],
    energy_rate: float,
    progress_prefix: str = "",
):
    case_dir = os.path.join(_OUTPUT_ROOT, case_id)
    va = os.path.join(case_dir, VARIANT_A_NAME)
    vb = os.path.join(case_dir, VARIANT_B_NAME)
    if os.path.isdir(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir, exist_ok=True)

    print(f"{progress_prefix}  generate variant A …", flush=True)
    generate_variant_a(params, homberger_dir, va, fnames)
    print(f"{progress_prefix}  relax variant B …", flush=True)
    build_variant_b(va, vb, params)

    gen_stats = aggregate_generation_stats(va, fnames)
    solver_rows = []
    print(f"{progress_prefix}  HC variant A (original TW) …", flush=True)
    solver_rows.extend(
        run_hc_on_folder("A", va, fnames, q_values, energy_rate, progress_prefix=progress_prefix)
    )
    print(f"{progress_prefix}  HC variant B (relaxed TW) …", flush=True)
    solver_rows.extend(
        run_hc_on_folder("B", vb, fnames, q_values, energy_rate, progress_prefix=progress_prefix)
    )
    sol_sum = summarize_by_variant(solver_rows)

    return {**gen_stats, **sol_sum, "output_dir": case_dir}


# Build the nine OAT cases: num_stations, R_factor, attr_ranges (each low/default/high).
def build_cases(defaults: dict):
    cases = []
    for label, val, lev in (
        ("num_stations_200", 200, "low"),
        ("num_stations_333", 333, "default"),
        ("num_stations_400", 400, "high"),
    ):
        p = deepcopy(defaults)
        p["num_stations"] = val
        cases.append((label, "num_stations", lev, p))

    for label, val, lev in (
        ("R_factor_low", 0.05, "low"),
        ("R_factor_default", defaults["R_factor"], "default"),
        ("R_factor_high", 0.15, "high"),
    ):
        p = deepcopy(defaults)
        p["R_factor"] = val
        cases.append((label, "R_factor", lev, p))

    for label, fac, lev in (
        ("attr_ranges_tight", 0.9, "low"),
        ("attr_ranges_default", 1.0, "default"),
        ("attr_ranges_loose", 1.1, "high"),
    ):
        if fac == 1.0:
            p = deepcopy(defaults)
            p["_attr_width_factor"] = 1.0
        else:
            p = scale_attr_ranges(defaults, fac)
            p["_attr_width_factor"] = fac
        cases.append((label, "attr_ranges", lev, p))

    return cases


# Save sensitivity_overview.png and sensitivity_feasible_by_group.png if matplotlib is available.
def save_plots(rows: list[dict], q_str: str, n_inst: int):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    os.makedirs(_PLOTS_DIR, exist_ok=True)
    by_group: dict[str, list[dict]] = {}
    for r in rows:
        g = r["oat_group"]
        by_group.setdefault(g, []).append(r)

    order = {"low": 0, "default": 1, "high": 2}

    # Order rows within an OAT group: low, default, high.
    def sort_key(r):
        return order.get(r["oat_level"], 9)

    metrics = [
        ("feasible_frac", "HC feasible fraction (all runs)"),
        ("avg_routes", "Mean routes (all runs)"),
        ("avg_charges", "Mean charging events (all runs)"),
        ("avg_runtime_s", "Mean runtime (s)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    titles = {
        "num_stations": "num_stations",
        "R_factor": "R_factor",
        "attr_ranges": "attr. ranges (simple scale)",
    }

    for ax_idx, (mkey, ylab) in enumerate(metrics):
        ax = axes[ax_idx]
        hc_key = f"hc_{mkey}" if mkey != "feasible_frac" else "hc_feasible_frac"
        if mkey == "feasible_frac":
            hc_key = "hc_feasible_frac"
        x_base = 0
        ticks, ticklabels = [], []
        group_width = 4.0
        bar_w = 0.8
        colors = ("#4477aa", "#ee6677", "#228833")
        for gi, gname in enumerate(["num_stations", "R_factor", "attr_ranges"]):
            grp = sorted(by_group.get(gname, []), key=sort_key)
            vals = []
            labels = []
            for r in grp:
                v = r.get(hc_key, "")
                if v == "":
                    vals.append(0.0)
                else:
                    vals.append(float(v))
                if gname == "num_stations":
                    labels.append(str(int(r["num_stations"])))
                elif gname == "R_factor":
                    labels.append(str(r["R_factor"]))
                else:
                    labels.append(r["oat_level"][:3])
            xs = [x_base + i * bar_w for i in range(len(vals))]
            for i, (x, v) in enumerate(zip(xs, vals)):
                ax.bar(x, v, width=bar_w * 0.9, color=colors[i % 3])
            ticks.extend(xs)
            ticklabels.extend(labels)
            x_base += group_width

        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylab, fontsize=8)
        ax.set_title(metrics[ax_idx][1], fontsize=9)
        ax.axvline(3.5, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(7.5, color="gray", linewidth=0.5, linestyle="--")

    fig.suptitle(f"Sensitivity (HC), n_inst={n_inst}, Q={q_str}", fontsize=10)
    fig.tight_layout()
    p1 = os.path.join(_PLOTS_DIR, "sensitivity_overview.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Second figure: one subplot per OAT group, line plot
    fig2, axs2 = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, gname in zip(axs2, ["num_stations", "R_factor", "attr_ranges"]):
        grp = sorted(by_group.get(gname, []), key=sort_key)
        xs = list(range(len(grp)))
        y_feas = [float(r["hc_feasible_frac"]) for r in grp]
        ax.plot(xs, y_feas, "o-", color="#4477aa", label="feasible_frac")
        ax.set_xticks(xs)
        if gname == "num_stations":
            ax.set_xticklabels([str(int(r["num_stations"])) for r in grp])
        elif gname == "R_factor":
            ax.set_xticklabels([str(r["R_factor"]) for r in grp])
        else:
            ax.set_xticklabels([r["oat_level"] for r in grp])
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(titles[gname])
        ax.set_ylabel("Feasible frac")
        ax.grid(True, alpha=0.3)
    fig2.tight_layout()
    p2 = os.path.join(_PLOTS_DIR, "sensitivity_feasible_by_group.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    return True


# CLI: run all cases, write sensitivity_analysis_results.csv, optional plots.
def main():
    ap = argparse.ArgumentParser(
        description="Full sensitivity pipeline in one go (default: 12 instances, Q=600, CSV + graphs).",
        epilog="No flags required. Run: python run_sensitivity_analysis.py",
    )
    ap.add_argument(
        "--per-class",
        type=int,
        default=2,
        help="optional: instances per class (default 2 → 12 total)",
    )
    ap.add_argument("--q", type=int, default=600, help="optional: battery Q (default 600)")
    ap.add_argument("--no-plots", action="store_true", help="optional: skip PNG graphs")
    args = ap.parse_args()

    q_values = [int(args.q)]
    q_str = str(q_values[0])
    defaults = load_default_params()
    _, energy_rate = load_ev_specs()

    homberger_dir = os.path.abspath(os.path.join(_SCRIPT_DIR, INPUT_DIR))
    all_files = get_instance_files(homberger_dir)
    fnames = select_instances_per_class(all_files, args.per_class)
    expect_n = args.per_class * len(INSTANCE_CLASSES)
    if len(fnames) != expect_n:
        print(
            f"Warning: expected {expect_n} instance files ({args.per_class} x {len(INSTANCE_CLASSES)} classes), "
            f"got {len(fnames)} (some classes may have fewer files).",
            flush=True,
        )

    os.makedirs(_OUTPUT_ROOT, exist_ok=True)
    cases = build_cases(defaults)

    print(
        f"Sensitivity: {len(fnames)} instances (up to {args.per_class} per class), "
        f"Q={q_str}, {len(cases)} cases, HC only.",
        flush=True,
    )

    csv_rows = []
    for ci, (case_id, oat_group, level, params) in enumerate(cases, start=1):
        prog = f"Progress {ci}/{len(cases)}: {case_id}"
        print(prog, flush=True)
        base = param_row(case_id, oat_group, level, params)
        base["homberger_instances_used"] = len(fnames)
        base["instance_selection"] = f"{args.per_class}_per_class_{'_'.join(INSTANCE_CLASSES)}"
        base["homberger_list"] = ";".join(fnames)
        base["Q_values"] = q_str
        base["energy_rate_h"] = energy_rate
        base["method"] = "HC"

        stats = run_case(
            case_id,
            params,
            homberger_dir,
            fnames,
            q_values,
            energy_rate,
            progress_prefix=prog,
        )
        base.update(stats)
        csv_rows.append(base)
        print(f"  finished {case_id}", flush=True)

    # CSV field order: metadata + params + generation + HC pooled + HC A/B
    priority = [
        "case_id",
        "oat_group",
        "oat_level",
        "method",
        "instance_selection",
        "homberger_instances_used",
        "Q_values",
        "energy_rate_h",
        "seed",
        "num_stations",
        "d_sep",
        "R_factor",
        "K",
        "r_min",
        "r_max",
        "p_min",
        "p_max",
        "w_min",
        "w_max",
        "cv_price",
        "cv_wait",
        "R_near",
        "R_medium",
        "R_far",
        "attr_range_width_factor",
        "output_dir",
        "gen_n_station_rows",
        "gen_avg_charging_rate",
        "gen_std_charging_rate",
        "gen_avg_wait_mu",
        "gen_avg_price_mu",
        "gen_avg_density_score",
        "gen_avg_nn_station_customer",
        "gen_std_nn_station_customer",
        "hc_n_runs",
        "hc_feasible_count",
        "hc_feasible_frac",
        "hc_avg_routes",
        "hc_avg_charges",
        "hc_avg_runtime_s",
        "hc_avg_distance_feas",
        "hc_avg_served",
        "hc_n_runs_A",
        "hc_feasible_count_A",
        "hc_feasible_frac_A",
        "hc_avg_routes_A",
        "hc_avg_charges_A",
        "hc_avg_runtime_s_A",
        "hc_avg_distance_feas_A",
        "hc_avg_served_A",
        "hc_n_runs_B",
        "hc_feasible_count_B",
        "hc_feasible_frac_B",
        "hc_avg_routes_B",
        "hc_avg_charges_B",
        "hc_avg_runtime_s_B",
        "hc_avg_distance_feas_B",
        "hc_avg_served_B",
    ]
    all_keys = set()
    for row in csv_rows:
        all_keys.update(row.keys())
    ordered = [k for k in priority if k in all_keys]
    rest = sorted(k for k in all_keys if k not in ordered and k != "homberger_list")
    fieldnames = ordered + rest
    if "homberger_list" in all_keys:
        fieldnames.append("homberger_list")

    with open(_RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    plots_ok = False
    if not args.no_plots:
        plots_ok = save_plots(csv_rows, q_str, len(fnames))

    if args.no_plots:
        extra = " (graphs skipped)"
    elif plots_ok:
        extra = ""
    else:
        extra = " (graphs skipped: pip install matplotlib)"
    print("Done.", flush=True)
    print(f"  CSV:   {_RESULTS_CSV}", flush=True)
    print(f"  Runs:  {_OUTPUT_ROOT}", flush=True)
    if plots_ok:
        print(f"  Graphs: {_PLOTS_DIR}", flush=True)
    elif extra:
        print(f"  {extra.strip()}", flush=True)


if __name__ == "__main__":
    main()
