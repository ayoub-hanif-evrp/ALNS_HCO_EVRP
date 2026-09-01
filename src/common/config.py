"""Paper experiment settings. One source of truth for budgets, seeds, names."""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

METHOD_GI = "GI"
METHOD_VNS = "VNS"
METHOD_SA = "SA"
METHOD_ALNS_FR = "ALNS-FR"
METHOD_ALNS_PR = "ALNS-PR"
METHOD_ALNS_PR_HCO = "ALNS-PR+HCO"
METHOD_ALNS_HCO = "ALNS-HCO"

PAPER_METHODS = (
    METHOD_GI, METHOD_VNS, METHOD_SA,
    METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_HCO,
)
STOCHASTIC_METHODS = (METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_HCO)
ABLATION_METHODS = (METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO, METHOD_ALNS_HCO)

# n=5 independent runs. Same seeds for ALNS-PR and ALNS-HCO.
SEEDS = (0, 1, 2, 3, 4)
N_RUNS = len(SEEDS)

TOTAL_BUDGET_S = 30.0
PHASE1_S = 15.0
PHASE2_S = 15.0
GI_DEADLINE_S = 40.0

FAMILIES = ("C1", "C2", "R1", "R2", "RC1", "RC2")
TW_VARIANTS = ("original", "relaxed")
TIE_TOL = 1e-6

MASTER_FIELDS = (
    "instance", "family", "tw", "method", "seed",
    "feasible", "served", "unserved", "hard_violations",
    "k", "distance", "duration", "customer_waiting",
    "charging_visits", "charged_energy", "station_waiting",
    "charging_time", "charging_expense",
    "runtime_s", "termination_reason",
    "phase1_s", "phase2_s", "phase1_k", "phase1_distance", "phase1_charging_expense",
    "k_star", "k_frozen_ok",
)


def experiment_config() -> dict:
    return {
        "benchmark": "Hanif 1000-customer EVRPTW",
        "n_instances_per_tw": 60,
        "customers_per_instance": 1000,
        "tw_variants": list(TW_VARIANTS),
        "seeds": list(SEEDS),
        "n_runs": N_RUNS,
        "alns_fr_budget_s": TOTAL_BUDGET_S,
        "alns_pr_budget_s": TOTAL_BUDGET_S,
        "alns_hco_total_budget_s": TOTAL_BUDGET_S,
        "alns_hco_phase1_s": PHASE1_S,
        "alns_hco_phase2_s": PHASE2_S,
        "gi_deadline_s": GI_DEADLINE_S,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method_names": list(PAPER_METHODS),
        "tie_tolerance": TIE_TOL,
        "percentage_definition": "paired instance-level 100*(HCO-PR)/PR, then mean of those percentages",
    }
