"""Post-hoc Reviewer-2 economic sensitivity (does not change the optimizer)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import aggregate_means, reviewer_root, write_csv, write_json  # noqa: E402
from src.common.config import METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15, METHOD_PHASE2_HET  # noqa: E402
from src.hanif.data.schneider_txt_convert import LARGE_56, parse_schneider_txt, DEFAULT_RAW, schneider_family  # noqa: E402
from src.hanif.loader import list_instances, load_hanif  # noqa: E402

# Hypothetical parameters — clearly labeled, not optimized by ALNS/HCO
DEPOT_TARIFFS = (0.0, 0.10, 0.20, 0.50)  # currency per energy unit
PUBLIC_PRICE_DEFAULT = None  # use observed charging_expense / charged_energy when available
WAGE_PER_TIME = (0.0, 0.25, 0.50, 1.00)  # currency per duration unit


def _f(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_hanif_rows():
    path = reviewer_root() / "hanif_main_seeds012" / "runs.csv"
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def economic_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        if r.get("method") not in (
            METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15, METHOD_PHASE2_HET,
        ):
            continue
        if str(r.get("feasible")).lower() not in {"true", "1"}:
            continue
        e_en = _f(r, "enroute_purchased_energy")
        if e_en is None:
            e_en = _f(r, "charged_energy")
        e_dep = _f(r, "depot_replenishment_energy")
        public_exp = _f(r, "charging_expense")
        # Implied public unit price from observed expense when possible
        public_unit = None
        if public_exp is not None and e_en is not None and e_en > 1e-12:
            public_unit = public_exp / e_en
        duration = _f(r, "duration")
        distance = _f(r, "distance")
        for depot_p in DEPOT_TARIFFS:
            depot_exp = None if e_dep is None else e_dep * depot_p
            total_elec = None
            if depot_exp is not None and public_exp is not None:
                total_elec = depot_exp + public_exp
            elif depot_exp is not None and public_unit is not None and e_en is not None:
                total_elec = depot_exp + public_unit * e_en
            for wage in WAGE_PER_TIME:
                wage_cost = None if duration is None else duration * wage
                total_hyp = None
                if total_elec is not None and wage_cost is not None:
                    total_hyp = total_elec + wage_cost
                out.append({
                    "instance": r["instance"], "family": r["family"], "tw": r["tw"],
                    "method": r["method"], "seed": r["seed"],
                    "k": r.get("k"), "distance": distance, "duration": duration,
                    "charged_energy": e_en, "depot_replenishment_energy": e_dep,
                    "observed_public_charging_expense": public_exp,
                    "implied_public_unit_price": public_unit,
                    "hypothetical_depot_tariff": depot_p,
                    "hypothetical_depot_expense": depot_exp,
                    "hypothetical_total_electricity_expense": total_elec,
                    "hypothetical_wage_per_duration": wage,
                    "hypothetical_driver_wage_cost": wage_cost,
                    "hypothetical_total_elec_plus_wage": total_hyp,
                    "analysis_type": "post_hoc_sensitivity_NOT_optimized",
                })
    return out


def break_even_notes(econ: list[dict]) -> list[dict]:
    """For FR vs PR matched means: depot price where extra public savings offset distance — illustrative only."""
    # Aggregate mean public expense and distance by method at depot_p=0, wage=0
    from collections import defaultdict
    buckets = defaultdict(list)
    for r in econ:
        if r["hypothetical_depot_tariff"] != 0.0 or r["hypothetical_wage_per_duration"] != 0.0:
            continue
        buckets[(r["method"], r["tw"])].append(r)
    notes = []
    for tw in ("original", "relaxed"):
        fr = buckets.get((METHOD_ALNS_FR, tw)) or []
        pr = buckets.get((METHOD_ALNS_PR, tw)) or []
        if not fr or not pr:
            continue
        def mean(xs, k):
            vals = [float(x[k]) for x in xs if x.get(k) not in (None, "")]
            return sum(vals) / len(vals) if vals else None
        d_fr, d_pr = mean(fr, "distance"), mean(pr, "distance")
        e_fr, e_pr = mean(fr, "observed_public_charging_expense"), mean(pr, "observed_public_charging_expense")
        notes.append({
            "tw": tw,
            "mean_distance_FR": d_fr,
            "mean_distance_PR": d_pr,
            "mean_public_expense_FR": e_fr,
            "mean_public_expense_PR": e_pr,
            "delta_distance_PR_minus_FR": None if d_fr is None or d_pr is None else d_pr - d_fr,
            "delta_public_expense_PR_minus_FR": None if e_fr is None or e_pr is None else e_pr - e_fr,
            "note": (
                "Illustrative only. Algorithm does not optimize wage-inclusive or depot-tariff total cost. "
                "If PR reduces public expense but increases distance, a depot/public price scenario may "
                "or may not favour PR depending on fleet-size (lex primary) and stakeholder weights."
            ),
        })
    return notes


def benchmark_tables() -> None:
    out = reviewer_root() / "benchmark_characteristics"
    out.mkdir(parents=True, exist_ok=True)
    hanif_rows = []
    for name in list_instances():
        inst = load_hanif(name, "original")
        hanif_rows.append({
            "benchmark": "Hanif",
            "instance": name,
            "family": name.split("_")[0],
            "num_customers": inst.num_customers,
            "num_stations": len(inst.charging_station_indices),
            "battery_capacity": float(inst.vehicle.battery_capacity),
            "load_capacity": float(inst.vehicle.load_capacity),
            "energy_consumption_rate": float(inst.vehicle.energy_consumption_rate),
            "tw_regimes": "original,relaxed",
            "charging": "heterogeneous station rate/price/wait from Hanif stations.csv",
        })
    write_csv(out / "table_hanif_instances.csv", hanif_rows)
    write_csv(out / "table_hanif_family_summary.csv", aggregate_means(
        [{"family": r["family"], "method": "meta", "feasible": True,
          "num_customers": r["num_customers"], "num_stations": r["num_stations"],
          "battery_capacity": r["battery_capacity"], "load_capacity": r["load_capacity"]}
         for r in hanif_rows],
        ("family",),
        ("num_customers", "num_stations", "battery_capacity", "load_capacity"),
    ))

    sch = []
    for name in LARGE_56:
        p = parse_schneider_txt(Path(DEFAULT_RAW) / f"{name}.txt")
        sch.append({
            "benchmark": "Schneider",
            "instance": name,
            "family": schneider_family(name),
            "num_customers": len(p["customers"]),
            "num_stations": len(p["stations"]),
            "Q": p["Q"], "C": p["C"], "r": p["r"], "g": p["g"], "v": p["v"],
            "charging": "homogeneous rate 1/g; wait=0; neutral price E=1 (schema only)",
            "tw_regime": "native Schneider TWs (no Hanif relaxation)",
            "distance_convention": "euclidean_unrounded",
        })
    write_csv(out / "table_schneider_instances.csv", sch)
    write_json(out / "README.json", {
        "hanif_n": len(hanif_rows),
        "schneider_n": len(sch),
        "economic_analysis": "post-hoc sensitivity only; optimizer unchanged",
    })


def main() -> int:
    rows = load_hanif_rows()
    econ = economic_rows(rows)
    out = reviewer_root() / "economic_sensitivity"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs_hypothetical_sensitivity.csv", econ)
    # Means at selected scenarios
    scen = [r for r in econ if r["hypothetical_depot_tariff"] in (0.0, 0.20) and r["hypothetical_wage_per_duration"] in (0.0, 0.50)]
    write_csv(out / "table_means_selected_scenarios.csv", aggregate_means(
        [{**r, "feasible": True} for r in scen],
        ("method", "tw", "hypothetical_depot_tariff", "hypothetical_wage_per_duration"),
        ("distance", "k", "observed_public_charging_expense", "hypothetical_depot_expense",
         "hypothetical_total_electricity_expense", "hypothetical_driver_wage_cost",
         "hypothetical_total_elec_plus_wage"),
    ))
    write_csv(out / "table_fr_pr_break_even_notes.csv", break_even_notes(econ))
    write_json(out / "ASSUMPTIONS.json", {
        "depot_tariffs_tested": list(DEPOT_TARIFFS),
        "wages_per_duration_tested": list(WAGE_PER_TIME),
        "optimizer_changed": False,
        "claim_restriction": (
            "These totals are hypothetical post-hoc evaluations. "
            "ALNS/HCO do not optimize wage-inclusive or depot-tariff total cost."
        ),
    })
    benchmark_tables()
    print("economic rows", len(econ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
