"""ALNS-HCO hyperparameters for the clean Hanif revision study."""

HYPERPARAMETERS = {
    "name": "Adaptive large neighbourhood search with heterogeneous charging oracle",
    "abbreviation": "ALNS-HCO",
    "random_seed": 0,
    "n_runs": 5,
    "seeds": [0, 1, 2, 3, 4],
    "total_time_limit_s": 30.0,
    "phase1_time_limit_s": 15.0,
    "phase2_time_limit_s": 15.0,
    "phase1_engine": "ALNS-PR: GI + ALNS with just-enough partial recharge",
    "phase1_charge_policy": "just-enough partial recharge",
    "phase1_objective": "lexicographic (unserved, K, distance)",
    "phase2_objective": (
        "Within D <= (1+eps)*D_PR: lexicographic "
        "(charging_expense, station_waiting, charging_time, distance) at frozen K=K*"
    ),
    "phase2_distance_tolerance": 0.0,  # provisional until post-fix epsilon selection
    "hco_candidates_per_criterion": 5,
    "phase2_repair_shortlist_size": 5,
    "hco_modes": ["detour", "heterogeneous"],
    "fleet_constraint": "Phase II uses exactly K=K* vehicles; no extra vehicles",
    "fixed_route_objective": (
        "heterogeneous: lex (C_ch, T_wait, T_ch, D), non-worsening; "
        "detour: geometric (D, N_ch, station_id sequence), non-worsening — never uses P/W/r"
    ),
    "hco": (
        "deterministic route-level charging heuristic with explicit (s,e_s) plans; "
        "detour mode ranks and accepts by geometry only (detour/station_id / D,N_ch); "
        "heterogeneous mode unions top-k by detour/price/wait/rate then lex charge score"
    ),
    "destroy_size_min": 4,
    "destroy_size_max": 12,
    "phase1_destroy_operators": ["random", "worst_detour", "related_shaw", "short_route"],
    "phase2_destroy_operators": [
        "random", "worst_detour", "related_shaw", "short_route",
        "expensive_charge", "slow_charge_wait", "undesirable_charge",
    ],
    "repair_phase1": "greedy cheapest feasible insertion with just-enough partial charge",
    "repair_phase2": (
        "shortlist promising insertions, HCO-evaluate each (or NoHCO rank without oracle), "
        "select best by lex (C_ch, T_wait, T_ch, D)"
    ),
    "nearby_routes": 8,
    "insert_positions": 3,
    "nearby_stations": 3,
    "adaptive_reaction_factor_rho": 0.2,
    "segment_length": 20,
    "score_new_best": 33,
    "score_better": 9,
    "score_accepted": 5,
    "sa_start_temperature_factor": 0.05,
    "sa_cooling": 0.9995,
    "accept_infeasible": False,
    "cache_by_customer_sequence": True,
    "epsilon_candidates": [0.0, 0.01, 0.02, 0.05],
    "time_allocations": [[10.0, 20.0], [15.0, 15.0], [20.0, 10.0]],
    "dev_seed": 0,
}
