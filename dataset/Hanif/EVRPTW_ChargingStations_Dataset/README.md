# Electric Vehicle Routing with Time Windows and Heterogeneous Charging Station Attributes — Dataset

This repository contains the **EVRPTW benchmark extension** of the Homberger **1000-customer VRPTW** instances: **60** base problems (10 per class: R1, R2, C1, C2, RC1, RC2), each augmented with **synthetically generated charging stations** and **heterogeneous station attributes** (charging rate, log-normal parameters for price and waiting time, density score). Two **time-window regimes** are provided: **original** customer windows (Variant A) and **distance-based relaxed** due dates capped at depot closing (Variant B).

The deposited archive is intended as a **reproducible, large-scale** testbed for EVRPTW methods. Full methodological detail, feasibility statistics, baseline solver results, and sensitivity of the generator appear in the associated **Data Descriptor** (*Data*, MDPI).

**License:** Creative Commons Attribution 4.0 International (CC BY 4.0)

---

## Repository layout

| Path | Contents |
|------|----------|
| `homberger_1000_customer_instances/` | Original unmodified Homberger VRPTW `.txt` files (1000 customers each). |
| `variant_A_original_TW/` | Augmented instances + matching `*_stations.csv` (original time windows). |
| `variant_B_relaxed_TW/` | Augmented instances + matching `*_stations.csv` (relaxed customer due dates). |
| `code/` | Python scripts for generation, verification, baselines, and sensitivity analysis. |
| `params.json` | Random seed and **all** generation parameters used for the released dataset. |
| `ev_specs.json` | Reference EV settings: battery capacity **Q** and energy consumption rate **h** (used for feasibility checks and baseline runs). |
| `sensitivity_runs/` | *(Optional)* Sensitivity outputs: **nine** case folders (`num_stations_*`, `R_factor_*`, `attr_ranges_*`) matching the paper, each with **12** base instances under `variant_A_original_TW/` and `variant_B_relaxed_TW/`. Not part of the core 60×2 release. |

**Instance text format:** Standard Homberger-style VRPTW text file: depot (node 0), customers (1–1000), then charging stations (node id **> 1000**).  
**Station CSV columns:** `station_id`, `charging_rate`, `price_mu`, `price_sigma`, `wait_mu`, `wait_sigma`, `density_score` (join `station_id` to the node id in the `.txt` file).

Stochastic **price** and **waiting time** at a station are not stored as single scalars; they are defined by the log-normal parameters above (sample as needed for your solver).

---

## Requirements

- **Python 3.10+** (recommended)
- **`numpy`** — required for `code/evrptw_solvers.py` (feasibility, CON/HC baselines, sensitivity HC runs)

Install example:

```bash
pip install numpy
```

---

## Regenerating the released dataset

Run from the **`code/`** directory so relative paths resolve correctly.

1. **Variant A** (charging stations + attributes, original time windows):

   ```bash
   cd code
   python algorithm1_stations.py
   ```

   Reads `../homberger_1000_customer_instances/` and `../params.json`, writes to `../variant_A_original_TW/`.

2. **Variant B** (relaxed customer due dates; copies non-`.txt` files such as CSVs from A):

   ```bash
   python algorithm2_relax_tw.py
   ```

   Reads `../variant_A_original_TW/`, writes to `../variant_B_relaxed_TW/` using `R_near`, `R_medium`, `R_far` from `params.json`.

To obtain **different** realizations, change `seed` (and optionally other fields) in `params.json` before step 1, then run both steps again.

---

## Verification and experiments (paper-aligned)

All commands below assume **`cd code`**.

| Script | Role |
|--------|------|
| `verify_feasibility.py` | Constructive feasibility check for **Q ∈ {200, 300, 400, 500, 600, 700}** with **h = 1.0**, Variants A and B; prints an aggregate table (same route evaluator as the metaheuristic code). |
| `run_baseline_solvers.py` | Parallel **CON** (construction only) and **HC** (hill climbing) runs for **Q ∈ {500, 600, 700, 800}**, **h** from `ev_specs.json`; writes `baseline_solver_results.csv`. |
| `analyze_solver_difficulty.py` | Reads `baseline_solver_results.csv` and prints summaries (by variant, Q, method, and instance class). |
| `run_sensitivity_analysis.py` | One-at-a-time sweeps of generator settings (`num_stations`, `R_factor`, attribute range width) on a **subset** of instances; regenerates under `../sensitivity_runs/<case_id>/` and writes `sensitivity_analysis_results.csv`. |

Examples:

```bash
cd code
python verify_feasibility.py
python run_baseline_solvers.py
python analyze_solver_difficulty.py
python run_sensitivity_analysis.py
```

Sensitivity CLI options (see `python run_sensitivity_analysis.py --help`): e.g. `--per-class N`, `--q <Q>`.

---

## Design notes

- Station coordinates are sampled in the **customer bounding box** with a **minimum separation**; attributes are tied to a **normalized local customer-density score** (see the Data Descriptor).
- Variant B relaxes **customer due dates** only; depot and station time windows are unchanged from Variant A.
- All charging-station data are **synthetic** and for **algorithmic benchmarking** only; they do not represent real infrastructure.

---

## Acknowledgments

Base VRPTW instances are the **Homberger 1000-customer** benchmark (SINTEF TOP / literature sources cited in the Data Descriptor). Thank you to the maintainers of that resource.

---

## Citation

Please cite the **Data Descriptor** article when using this dataset (use the bibliographic details from the publisher version of the paper once it is available).
