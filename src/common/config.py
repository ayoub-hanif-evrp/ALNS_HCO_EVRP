"""Clean Hanif revision experiment settings."""
from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone

METHOD_GI = "GI"
METHOD_VNS = "VNS"
METHOD_SA = "SA"
METHOD_ALNS_FR = "ALNS-FR"
METHOD_ALNS_PR = "ALNS-PR"
METHOD_PHASE1_PR15 = "PhaseI-PR-15"
METHOD_FIXED_DETOUR = "FixedRoute-HCO-Detour"
METHOD_FIXED_HET = "FixedRoute-HCO-Heterogeneous"
METHOD_PHASE2_DETOUR = "PhaseII-Detour"
METHOD_PHASE2_HET = "PhaseII-Heterogeneous"
METHOD_PHASE2_PLAIN = "PhaseII-Plain"
METHOD_PHASE2_NOHCO = "PhaseII-NoHCO"

# Legacy aliases kept for old scripts
METHOD_ALNS_PR_HCO = "ALNS-PR+HCO"
METHOD_ALNS_HCO = "ALNS-HCO"

ROUTING_METHODS = (METHOD_GI, METHOD_VNS, METHOD_SA, METHOD_ALNS_FR, METHOD_ALNS_PR)
TRUSTED_BASELINE_METHODS = ROUTING_METHODS + (METHOD_PHASE1_PR15,)
HCO_DEPENDENT_METHODS = (
    METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
    METHOD_PHASE2_NOHCO, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
    METHOD_PHASE2_PLAIN,
)
SHARED_DEPENDENT = HCO_DEPENDENT_METHODS
CLEAN_METHODS = TRUSTED_BASELINE_METHODS + HCO_DEPENDENT_METHODS

PAPER_METHODS = (
    METHOD_GI, METHOD_VNS, METHOD_SA,
    METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_HCO,
)
STOCHASTIC_METHODS = (
    METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15,
    METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
    METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
    METHOD_VNS, METHOD_SA,
)
ABLATION_METHODS = (
    METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15,
    METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
    METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
)

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
    "k_star", "k_frozen_ok", "phase1_checksum", "hco_mode", "eps",
    "error",
    # Round-1 reviewer reporting (post-hoc; not used by optimizers)
    "propulsion_energy",
    "enroute_purchased_energy",
    "residual_battery_at_depot",
    "depot_replenishment_energy",
    "mean_energy_per_charging_visit",
    "initial_battery_energy",
    "energy_balance_ok",
    "energy_balance_residual",
    "alns_iterations",
    "alns_phase1_iterations",
    "alns_phase2_iterations",
    "hco_calls",
    "feasible_hco_calls",
    "hco_eval_time_s",
    "execution_time_s",
    "solver_runtime_s",
    "reporting_overhead_s",
    "end_to_end_runtime_s",
    "phase1_runtime_s",
    "phase2_runtime_s",
)

# Fields computed after the solve; never consulted by ALNS/HCO decisions.
REPORTING_ONLY_FIELDS = (
    "propulsion_energy",
    "enroute_purchased_energy",
    "residual_battery_at_depot",
    "depot_replenishment_energy",
    "mean_energy_per_charging_visit",
    "initial_battery_energy",
    "energy_balance_ok",
    "energy_balance_residual",
    "alns_iterations",
    "alns_phase1_iterations",
    "alns_phase2_iterations",
    "hco_calls",
    "feasible_hco_calls",
    "hco_eval_time_s",
    "execution_time_s",
    "solver_runtime_s",
    "reporting_overhead_s",
    "end_to_end_runtime_s",
    "phase1_runtime_s",
    "phase2_runtime_s",
)


def development_instances() -> list[str]:
    """First instance alphabetically/numerically from each family (deterministic)."""
    return ["C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1"]


def experiment_config() -> dict:
    from src.alns_hco.hyperparameters import HYPERPARAMETERS
    return {
        "benchmark": "Hanif 1000-customer EVRPTW",
        "study": "clean_revision_shared_phase1",
        "n_instances_per_tw": 60,
        "customers_per_instance": 1000,
        "tw_variants": list(TW_VARIANTS),
        "seeds": list(SEEDS),
        "n_runs": N_RUNS,
        "alns_fr_budget_s": TOTAL_BUDGET_S,
        "alns_pr_budget_s": TOTAL_BUDGET_S,
        "shared_phase1_s": PHASE1_S,
        "phase2_s": PHASE2_S,
        "gi_deadline_s": GI_DEADLINE_S,
        "hco_candidates_per_criterion": HYPERPARAMETERS["hco_candidates_per_criterion"],
        "phase2_distance_tolerance": HYPERPARAMETERS["phase2_distance_tolerance"],
        "epsilon_candidates": list(HYPERPARAMETERS["epsilon_candidates"]),
        "development_instances": development_instances(),
        "methods": list(CLEAN_METHODS),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tie_tolerance": TIE_TOL,
        "percentage_definition": "paired instance-level 100*(B-A)/A, then mean of those percentages",
        "no_schneider": True,
    }
