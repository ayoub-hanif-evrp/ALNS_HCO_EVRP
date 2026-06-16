# Table-style feasibility check: constructive heuristic + evaluate_routes over Q grid (variants A and B).

import os

from evrptw_solvers import (
    constructive_feasibility_summary,
    load_station_rates,
    parse_instance,
    prepare_instance,
)

VARIANT_A_DIR = os.path.join(os.path.dirname(__file__), "..", "variant_A_original_TW")
VARIANT_B_DIR = os.path.join(os.path.dirname(__file__), "..", "variant_B_relaxed_TW")
Q_VALUES = [200, 300, 400, 500, 600, 700]
ENERGY_RATE = 1.0


# Run constructive_feasibility_summary for each instance in variant_dir across Q_VALUES; return per-Q result lists.
def run_variant(variant_name, variant_dir):
    files = sorted(f for f in os.listdir(variant_dir) if f.lower().endswith(".txt"))
    total = len(files)
    results = {Q: [] for Q in Q_VALUES}

    print(f"Processing {variant_name} ({total} instances)...")

    for i, filename in enumerate(files):
        instance_name = filename.replace(".txt", "").replace(".TXT", "")
        csv_path = os.path.join(variant_dir, instance_name + "_stations.csv")
        if not os.path.exists(csv_path):
            continue

        fleet_size, load_capacity, depot, customers, stations = parse_instance(
            os.path.join(variant_dir, filename)
        )
        station_rates = load_station_rates(csv_path)
        instance = prepare_instance(
            fleet_size, load_capacity, depot, customers, stations, station_rates
        )

        status_parts = []
        for Q in Q_VALUES:
            summ = constructive_feasibility_summary(instance, Q, ENERGY_RATE)
            results[Q].append(
                {
                    "feasible": summ["feasible"],
                    "routes": summ["num_routes"],
                    "charges": summ["num_charges"],
                    "avg_charges_per_vehicle": summ["avg_charges_per_vehicle"],
                }
            )
            if summ["feasible"]:
                status_parts.append(f"Q={Q}: OK")
            else:
                n_c = instance["num_customers"]
                status_parts.append(f"Q={Q}: FAIL({n_c - summ['served']})")

        print(f"  [{i+1:>2}/{total}] {instance_name:<15}  {'  '.join(status_parts)}")

    print(f"Done.\n")
    return results


# Print aggregated feasibility counts and averages for variants A and B.
def print_table(results_a, results_b):
    print("=" * 80)
    print("Table 3 — Feasibility under different battery capacities (h = 1.0)")
    print("  (feasible = full partition passes evaluate_routes)")
    print("=" * 80)
    print(
        f"{'Q':<6} {'Feasible A':>12} {'Feasible B':>12} "
        f"{'Avg Veh A':>11} {'Avg Veh B':>11} "
        f"{'Avg chg/veh A':>14} {'Avg chg/veh B':>14}"
    )
    print("-" * 80)

    for Q in Q_VALUES:
        a_list = results_a[Q]
        b_list = results_b[Q]
        feas_a = sum(r["feasible"] for r in a_list)
        feas_b = sum(r["feasible"] for r in b_list)
        avg_veh_a = sum(r["routes"] for r in a_list) / len(a_list)
        avg_veh_b = sum(r["routes"] for r in b_list) / len(b_list)
        avg_chg_a = sum(r["avg_charges_per_vehicle"] for r in a_list) / len(a_list)
        avg_chg_b = sum(r["avg_charges_per_vehicle"] for r in b_list) / len(b_list)

        print(
            f"{Q:<6} {feas_a:>5}/{len(a_list):<5} {feas_b:>5}/{len(b_list):<5}"
            f" {avg_veh_a:>11.1f} {avg_veh_b:>11.1f} {avg_chg_a:>14.2f} {avg_chg_b:>14.2f}"
        )

    print("=" * 80)


# Entry: run both variants and print the summary table.
def main():
    print("EVRPTW Feasibility Verification")
    print(f"Energy consumption rate h = {ENERGY_RATE}")
    print(f"Battery capacities tested: {Q_VALUES}\n")

    results_a = run_variant("Variant A (original time windows)", VARIANT_A_DIR)
    results_b = run_variant("Variant B (relaxed time windows)", VARIANT_B_DIR)

    print_table(results_a, results_b)


if __name__ == "__main__":
    main()
