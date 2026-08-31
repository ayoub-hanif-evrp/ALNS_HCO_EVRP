"""Literature EVRPTW metaheuristics used as comparison methods.

None of these papers report results on this 1000-customer Homberger+stations
set, so comparison means re-running the same algorithmic families here under
matched instance parameters (Q, h, time windows). Implementations here follow
the published neighborhoods and search templates; they are not bitwise clones
of the original codes.
"""
from __future__ import annotations

from typing import Any

LITERATURE_METAHEURISTICS: list[dict[str, Any]] = [
    {
        "id": "VNS_TS",
        "name": "VNS-style",
        "citation": "Schneider, Stenger, Goeke (2014), Transportation Science",
        "title": "The Electric Vehicle-Routing Problem with Time Windows and Recharging Stations",
        "family": "variable neighborhood search (relocate, swap, 2-opt)",
        "original_instances": "Schneider EVRPTW (Solomon-based, up to 100 customers)",
        "reimplemented_here": True,
        "notes": "Matched baseline inspired by Schneider neighbourhoods. Not a faithful VNS+tabu clone (tabu unused).",
    },
    {
        "id": "ALNS",
        "name": "ALNS-style",
        "citation": "Keskin and Catay (2016), Transportation Research Part C",
        "title": "Partial recharge strategies for the electric vehicle routing problem with time windows",
        "family": "adaptive large neighborhood search (destroy/repair)",
        "original_instances": "Schneider EVRPTW with partial recharging",
        "reimplemented_here": True,
        "notes": "Matched destroy/repair baseline. Not a bitwise Keskin & Catay ALNS (no adaptive weights, no continuous recharge).",
    },
    {
        "id": "ALNS_LS",
        "name": "ALNS+LS",
        "citation": "Goeke and Schneider (2015), EJOR; Hiermann, Puchinger, Ropke, Hartl (2016), EJOR",
        "title": "Mixed-fleet / fleet-size-and-mix EVRPTW ALNS with local search",
        "family": "ALNS with embedded local search",
        "original_instances": "Schneider EVRPTW and E-FSMFTW (up to a few hundred customers)",
        "reimplemented_here": False,
        "notes": "Same ALNS family with extra LS intensification. Covered here by the ALNS path.",
    },
    {
        "id": "ILS",
        "name": "ILS",
        "citation": "Montoya, Gueret, Mendoza, Villegas (2017), Transportation Research Part B",
        "title": "The electric vehicle routing problem with nonlinear charging function",
        "family": "iterated local search",
        "original_instances": "EVRP with nonlinear charging (Schneider-scale and extensions)",
        "reimplemented_here": False,
        "notes": "Standard large-scale EVRP local-search family; nonlinear charging is a different physics model.",
    },
    {
        "id": "LNS_GTS",
        "name": "LNS×GTS",
        "citation": "Loffler, Desaulniers, Irnich, Schneider (2020), Networks",
        "title": "Routing electric vehicles with a single recharge per route",
        "family": "large neighborhood search + granular tabu search",
        "original_instances": "EVRPTW with a single recharge (up to 100 customers)",
        "reimplemented_here": False,
        "notes": "Implicit recharge representation. Different recharge cardinality than this benchmark.",
    },
    {
        "id": "HMA",
        "name": "Hybrid memetic algorithm",
        "citation": "Jia et al. (2024), arXiv:2410.19580",
        "title": "Hybrid Memetic Search for Electric Vehicle Routing with Time Windows, Simultaneous Pickup-Delivery, and Partial Recharges",
        "family": "memetic / population search",
        "original_instances": "EVRP-TW-SPD, including new 1000-customer JD Logistics instances",
        "reimplemented_here": False,
        "notes": "Closest published 1000-customer EVRP metaheuristic, but a different problem (pickup-delivery).",
    },
    {
        "id": "KESKIN_2021",
        "name": "Simulation-based ALNS",
        "citation": "Keskin, Catay, Laporte (2021), Computers & Operations Research",
        "title": "A simulation-based heuristic for EVRPTW with stochastic waiting times at recharging stations",
        "family": "ALNS + simulation / recourse",
        "original_instances": "100-customer EVRPTW with stochastic station queues",
        "reimplemented_here": False,
        "notes": "Precedent for expected-wait construction plus out-of-sample simulation. Not run here.",
    },
    {
        "id": "FROGER_2022",
        "name": "ILS + branch-and-cut",
        "citation": "Froger, Mendoza, Jabali, Laporte (2022), Transportation Science",
        "title": "The electric vehicle routing problem with capacitated charging stations",
        "family": "ILS with B&C assembler",
        "original_instances": "EVRP with capacitated stations and multiple technologies",
        "reimplemented_here": False,
        "notes": "Station capacity is not in the Hanif model used here.",
    },
    {
        "id": "NAFSTAD_2025",
        "name": "Branch-price-and-cut",
        "citation": "Nafstad, Hasle, Kloster (2025), Transportation Science",
        "title": "Branch-price-and-cut for EVRP with heterogeneous recharging technologies and nonlinear recharging",
        "family": "exact BP&C",
        "original_instances": "up to 100 customers and 21 charging locations",
        "reimplemented_here": False,
        "notes": "Exact method on a different scale and charging physics. Motivates route-level exact validation of Stage 2, not a 1000-customer competitor.",
    },
    {
        "id": "WANG_2026",
        "name": "MILP + AILS",
        "citation": "Wang, Adulyasak, Cordeau (2026), Computers & Operations Research",
        "title": "Electric vehicle routing with heterogeneous charging stations",
        "family": "AILS with exact fixed-route scheduler",
        "original_instances": "100-customer controlled instances; 1100–1500-customer real cases",
        "reimplemented_here": False,
        "notes": "Closest conceptual neighbour (heterogeneous stations, TOU, time-dependent wait). Different dataset and joint routing+charging search. Not re-run here.",
    },
    {
        "id": "WANG_2026_TWOPHASE",
        "name": "Two-phase lexicographic ALNS",
        "citation": "Wang et al. (2026), Transportation Research Part C",
        "title": "A two-phase lexicographic optimization framework for fleet sizing and routing in electric vehicle routing problems",
        "family": "lexicographic fleet-first ALNS with partial recharge",
        "original_instances": "EVRPTW (not the Hanif 1000-customer station set)",
        "reimplemented_here": False,
        "notes": "Already uses Phase 1 minimise fleet then Phase 2 routing. Blocks claiming two-phase fleet-first as CARE-FI novelty.",
    },
]


def competitor_table_rows() -> list[dict[str, Any]]:
    rows = []
    for item in LITERATURE_METAHEURISTICS:
        rows.append({
            "method_id": item["id"],
            "method": item["name"],
            "citation": item["citation"],
            "algorithm_family": item["family"],
            "original_instances": item["original_instances"],
            "reimplemented_on_this_benchmark": item["reimplemented_here"],
        })
    return rows
