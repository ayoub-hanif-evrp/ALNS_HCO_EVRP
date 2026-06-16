# Read baseline_solver_results.csv and print Q-sweep or legacy difficulty summaries to stdout.

import csv
import os
from collections import defaultdict

INPUT_CSV = os.path.join(os.path.dirname(__file__), "baseline_solver_results.csv")

REF_METHOD = "HC"
METHOD_ORDER = ["CON", "HC"]


# Homberger class prefix from instance stem (e.g. C1_10_3 -> C1).
def instance_class(name):
    return name.split("_")[0]


# Load CSV rows and normalize method_label from method or method_label column.
def load_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        ml = (r.get("method_label") or r.get("method") or "").strip()
        r["method_label"] = ml
    return rows


# Parse total_distance from a row, or None if missing.
def row_distance(row):
    v = (row.get("total_distance") or "").strip()
    if not v:
        return None
    return float(v)


# Parse vehicle count from num_vehicles_used or num_routes.
def row_num_vehicles(row):
    v = (row.get("num_vehicles_used") or row.get("num_routes") or "").strip()
    if not v:
        return None
    return int(float(v))


# Parse pct_fleet_used as float, or None.
def row_pct_fleet(row):
    v = (row.get("pct_fleet_used") or "").strip()
    if not v:
        return None
    return float(v)


# Legacy CSV key for effective fleet cap; default "file".
def row_fleet_effective_key(row):
    v = (row.get("fleet_effective") or "").strip()
    if v:
        return v
    return "file"


# Sort order for fleet_effective keys in legacy reports.
def fleet_sort_key(key):
    if key == "file":
        return -1
    try:
        return int(key)
    except ValueError:
        return 99999


# Subset of rows with feasible == 1.
def feasible_only(rs):
    return [r for r in rs if int(r["feasible"])]


# Count feasible runs and means of distance/time/vehicles/%fleet over feasible rows only.
def summarize_rows(rs):
    n = len(rs)
    feas = sum(int(r["feasible"]) for r in rs)
    fr = feasible_only(rs)
    if not fr:
        return n, feas, None, None, None, None
    dists = [row_distance(r) for r in fr]
    dists = [d for d in dists if d is not None]
    secs = [float(r["seconds"]) for r in fr]
    ve = [row_num_vehicles(r) for r in fr]
    ve = [x for x in ve if x is not None]
    pcts = [row_pct_fleet(r) for r in fr]
    pcts = [p for p in pcts if p is not None]
    mean_d = sum(dists) / len(dists) if dists else None
    mean_t = sum(secs) / len(secs) if secs else None
    mean_v = sum(ve) / len(ve) if ve else None
    mean_p = sum(pcts) / len(pcts) if pcts else None
    return n, feas, mean_d, mean_t, mean_v, mean_p


# Human-readable line for mean stats from summarize_rows.
def fmt_means(mean_d, mean_t, mean_v, mean_p):
    if mean_d is None:
        return "mean dist n/a | mean time n/a | mean veh n/a | mean %fleet n/a"
    return (
        f"mean dist {mean_d:.0f} | mean time {mean_t:.3f} s | "
        f"mean veh {mean_v:.1f} | mean %fleet {mean_p:.1f}"
    )


# True if CSV has Q column (new baseline format).
def has_q_sweep_columns(rows):
    return bool(rows) and "Q" in rows[0]


# Total charging events from row (num_charges_total or num_charges).
def _row_num_charges_total(r):
    for key in ("num_charges_total", "num_charges"):
        v = (r.get(key) or "").strip()
        if v:
            try:
                return float(v)
            except ValueError:
                return 0.0
    return 0.0


# Average charges per route from column or derived from totals.
def _row_avg_charges_per_vehicle(r):
    v = (r.get("avg_charges_per_vehicle") or "").strip()
    if v:
        try:
            return float(v)
        except ValueError:
            pass
    tot = _row_num_charges_total(r)
    nr_s = (r.get("num_routes") or "").strip()
    if nr_s:
        try:
            nr = float(nr_s)
            if nr > 0:
                return tot / nr
        except ValueError:
            pass
    return 0.0


# Aggregate feasible count and mean metrics for one (variant, Q, method) group.
def aggregate_table3_row(rs):
    n = len(rs)
    if n == 0:
        return None
    feas = sum(int(r["feasible"]) for r in rs)
    avg_routes = sum(float(r["num_routes"] or 0) for r in rs) / n
    avg_chg_tot = sum(_row_num_charges_total(r) for r in rs) / n
    avg_chg_pv = sum(_row_avg_charges_per_vehicle(r) for r in rs) / n
    avg_sec = sum(float(r["seconds"] or 0) for r in rs) / n
    fr = feasible_only(rs)
    dists = [row_distance(r) for r in fr]
    dists = [d for d in dists if d is not None]
    mean_d = sum(dists) / len(dists) if dists else None
    return n, feas, avg_routes, avg_chg_tot, avg_chg_pv, avg_sec, mean_d


# Print sections 1–5 for Q-sweep baseline CSV.
def print_q_sweep_analysis(rows):
    by_key = defaultdict(list)
    for r in rows:
        try:
            qv = int(float(r["Q"]))
        except (TypeError, ValueError):
            continue
        key = (r["variant"], qv, r["method_label"])
        by_key[key].append(r)

    q_values = sorted({int(float(r["Q"])) for r in rows if (r.get("Q") or "").strip()})
    methods = [m for m in METHOD_ORDER if any(
        (r["method_label"] == m) for r in rows
    )]
    if not methods:
        methods = sorted({r["method_label"] for r in rows})

    h_sample = rows[0].get("h", "").strip() if rows else ""
    print("\n1) Table-3-style summary (metaheuristic Q-sweep)")
    print("-" * 100)
    print(
        "  avg_chg_tot: mean total charging events per instance; "
        "avg_chg/veh: mean (charges / routes) per instance.\n"
        "  Means over ALL rows unless noted; avg_dist_feas uses feasible rows only."
    )
    if h_sample:
        print(f"  h column in CSV: {h_sample}")
    print()

    for variant in ("A", "B"):
        print(f"  Variant {variant}")
        hdr = (
            f"{'Q':>4} {'Method':<6} {'feas/n':>8} {'avg_routes':>11} "
            f"{'avg_chg_tot':>12} {'chg/veh':>10} {'avg_sec':>10} {'avg_dist_feas':>14}"
        )
        print("  " + hdr)
        print("  " + "-" * len(hdr))
        for Q in q_values:
            for method in methods:
                rs = by_key.get((variant, Q, method), [])
                agg = aggregate_table3_row(rs)
                if agg is None:
                    continue
                n, feas, ar, act, acpv, asec, md = agg
                ds = f"{md:.1f}" if md is not None else "n/a"
                print(
                    f"  {Q:>4} {method:<6} {feas:>4}/{n:<3} {ar:>11.1f} "
                    f"{act:>12.1f} {acpv:>10.2f} {asec:>10.3f} {ds:>14}"
                )
        print()

    # --- Pooled by method (all Q) ---
    print("2) Each method (all variants, all Q pooled)")
    print("-" * 60)
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method_label"]].append(r)
    for method in sorted(by_method.keys()):
        rs = by_method[method]
        n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
        print(f"  {method}: feasible {feas}/{n} | {fmt_means(mean_d, mean_t, mean_v, mean_p)}")

    # --- By variant (pooled Q) ---
    print("\n3) By variant (all Q pooled)")
    print("-" * 60)
    for variant in ("A", "B"):
        for method in sorted(by_method.keys()):
            rs = [r for r in by_method[method] if r["variant"] == variant]
            if not rs:
                continue
            n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
            print(
                f"  variant {variant} | {method}: feasible {feas}/{n} | "
                f"{fmt_means(mean_d, mean_t, mean_v, mean_p)}"
            )

    # --- By instance class (REF_METHOD, all Q) ---
    print(f"\n4) By instance class ({REF_METHOD}; all Q pooled)")
    print("-" * 60)
    class_rows = defaultdict(list)
    for r in rows:
        if r["method_label"] != REF_METHOD:
            continue
        class_rows[instance_class(r["instance"])].append(r)
    for cls in sorted(class_rows.keys()):
        rs = class_rows[cls]
        n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
        print(f"  {cls}: feasible {feas}/{n} | {fmt_means(mean_d, mean_t, mean_v, mean_p)}")

    print(f"\n5) Solve time spread for {REF_METHOD} (feasible rows, all Q)")
    print("-" * 60)
    times = sorted(
        float(r["seconds"])
        for r in rows
        if r["method_label"] == REF_METHOD and int(r["feasible"])
    )
    if times:
        mid = times[len(times) // 2]
        print(f"  min {times[0]:.4f} | max {times[-1]:.4f} | median {mid:.4f}")
    else:
        print("  (no feasible rows)")


# Print legacy fleet-cap style sections.
def print_legacy_analysis(rows):
    print("Solver difficulty summary (legacy CSV / fleet-cap experiments)")
    print("  Mean distance / time / vehicles / %fleet: feasible rows only.")
    print("=" * 60)

    by_method = defaultdict(list)
    by_fleet = defaultdict(list)
    for r in rows:
        by_method[r["method_label"]].append(r)
        by_fleet[row_fleet_effective_key(r)].append(r)

    print("\n1) By effective fleet limit (all methods pooled)")
    print("-" * 60)
    for fk in sorted(by_fleet.keys(), key=fleet_sort_key):
        rs = by_fleet[fk]
        n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
        print(f"  fleet_effective={fk}: feasible {feas}/{n} | {fmt_means(mean_d, mean_t, mean_v, mean_p)}")

    print("\n2) Each method (all instances, all fleet caps)")
    print("-" * 60)
    for method in sorted(by_method.keys()):
        rs = by_method[method]
        n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
        print(f"  {method}: feasible {feas}/{n} | {fmt_means(mean_d, mean_t, mean_v, mean_p)}")

    print("\n3) By time-window variant (A=original, B=relaxed)")
    print("-" * 60)
    for variant in ("A", "B"):
        for method in sorted(by_method.keys()):
            rs = [r for r in by_method[method] if r["variant"] == variant]
            if not rs:
                continue
            n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
            print(
                f"  variant {variant} | {method}: feasible {feas}/{n} | "
                f"{fmt_means(mean_d, mean_t, mean_v, mean_p)}"
            )

    print(f"\n4) By fleet limit and variant ({REF_METHOD} only)")
    print("-" * 60)
    for fk in sorted(by_fleet.keys(), key=fleet_sort_key):
        for variant in ("A", "B"):
            rs = [
                r
                for r in by_fleet[fk]
                if r["method_label"] == REF_METHOD and r["variant"] == variant
            ]
            if not rs:
                continue
            n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
            print(
                f"  fleet={fk} variant {variant}: feasible {feas}/{n} | "
                f"{fmt_means(mean_d, mean_t, mean_v, mean_p)}"
            )

    print(f"\n5) By instance class ({REF_METHOD}; all fleet caps pooled)")
    print("-" * 60)
    class_rows = defaultdict(list)
    for r in rows:
        if r["method_label"] != REF_METHOD:
            continue
        class_rows[instance_class(r["instance"])].append(r)
    for cls in sorted(class_rows.keys()):
        rs = class_rows[cls]
        n, feas, mean_d, mean_t, mean_v, mean_p = summarize_rows(rs)
        print(f"  {cls}: feasible {feas}/{n} | {fmt_means(mean_d, mean_t, mean_v, mean_p)}")

    print(f"\n6) By instance class and fleet_effective ({REF_METHOD} only)")
    print("-" * 60)
    for fk in sorted(by_fleet.keys(), key=fleet_sort_key):
        class_f = defaultdict(list)
        for r in by_fleet[fk]:
            if r["method_label"] != REF_METHOD:
                continue
            class_f[instance_class(r["instance"])].append(r)
        if not class_f:
            continue
        parts = []
        for cls in sorted(class_f.keys()):
            rs = class_f[cls]
            n, feas, _md, _mt, _mv, _mp = summarize_rows(rs)
            parts.append(f"{cls}:{feas}/{n}")
        print(f"  fleet_effective={fk}: " + " | ".join(parts))

    print(f"\n7) Spread of solve time (seconds) for {REF_METHOD}, feasible rows")
    print("-" * 60)
    times = sorted(
        float(r["seconds"])
        for r in rows
        if r["method_label"] == REF_METHOD and int(r["feasible"])
    )
    if times:
        mid = times[len(times) // 2]
        print(f"  min {times[0]:.4f} | max {times[-1]:.4f} | median {mid:.4f}")
    else:
        print("  (no feasible rows)")


# Load CSV and dispatch to Q-sweep or legacy printer.
def main():
    if not os.path.isfile(INPUT_CSV):
        print(f"Missing {INPUT_CSV}")
        print("Run: python run_baseline_solvers.py")
        return

    rows = load_rows(INPUT_CSV)
    if not rows:
        print("CSV is empty.")
        return

    print("Solver difficulty / baseline summary")
    print("=" * 60)

    if has_q_sweep_columns(rows):
        print_q_sweep_analysis(rows)
    else:
        print_legacy_analysis(rows)

    print("=" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
