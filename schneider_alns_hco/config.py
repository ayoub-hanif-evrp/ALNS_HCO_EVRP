"""Settings for the Schneider ALNS-HCO vs-literature experiment.

All experiment outputs stay under this folder. The solver itself is the
existing paper method in ``src/alns-hco`` (imported, not copied).
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RESULTS = HERE / "results"
ROUTES = RESULTS / "routes"
INSTANCE_DIR = ROOT / "dataset" / "Schneider" / "evrptw_instances"

# Official Schneider large set: 100 customers, 21 stations.
LARGE_56 = tuple(
    [f"c10{i}_21" for i in range(1, 10)]
    + [f"c20{i}_21" for i in range(1, 9)]
    + [f"r10{i}_21" for i in range(1, 10)]
    + [f"r1{i}_21" for i in range(10, 13)]
    + [f"r20{i}_21" for i in range(1, 10)]
    + [f"r2{i}_21" for i in range(10, 12)]
    + [f"rc10{i}_21" for i in range(1, 9)]
    + [f"rc20{i}_21" for i in range(1, 9)]
)

QUICK_SIX = ("c101_21", "c201_21", "r101_21", "r201_21", "rc101_21", "rc201_21")

SEEDS = (0, 1, 2, 3, 4)
N_RUNS = len(SEEDS)

# 100-customer instances; literature CPU times are often 15–60 min.
# Default is 3 min split equally so the full 56 × 5-seed run is feasible.
TOTAL_BUDGET_S = 180.0
PHASE1_S = 90.0
PHASE2_S = 90.0

RESULT_FIELDS = (
    "instance", "family", "seed",
    "feasible", "served", "unserved", "hard_violations",
    "k", "distance", "duration",
    "charging_visits", "charged_energy", "charging_expense",
    "runtime_s", "phase1_s", "phase2_s",
    "phase1_k", "phase1_distance",
    "k_star", "k_frozen_ok",
)

TIE_TOL = 1e-6
