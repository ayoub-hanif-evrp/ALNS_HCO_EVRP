"""Correct Schneider feasibility investigation using replay_route (no algorithm changes)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.job_store import JobStore  # noqa: E402
from src.alns_hco.reviewer_experiments import aggregate_means, reviewer_root, write_csv, write_json  # noqa: E402
from src.common.paired_stats import paired_instance_effects, summarize_paired  # noqa: E402
from src.hanif.data.instance_loader import load_instance  # noqa: E402
from src.hanif.data.schneider_txt_convert import (  # noqa: E402
    DEFAULT_CONVERTED,
    DEFAULT_RAW,
    LARGE_56,
    convert_one_txt,
    parse_schneider_txt,
)
from src.hanif.replay import replay_route  # noqa: E402


def _probe_customer(inst, cid: int) -> dict:
    Q = float(inst.vehicle.battery_capacity)
    e0c = float(inst.energy_matrix[0, cid])
    ec0 = float(inst.energy_matrix[cid, 0])
    node = inst.nodes[cid]
    # Direct solo route feasibility via replay_route (not full-solution unserved)
    solo = replay_route(inst, [0, cid, 0], [None, None, None])
    station_ok = []
    for s in map(int, inst.charging_station_indices):
        for nodes in ([0, s, cid, 0], [0, cid, s, 0], [0, s, cid, s, 0]):
            rm = replay_route(inst, nodes, [None] * len(nodes))
            if rm.feasible:
                station_ok.append({"nodes": nodes, "distance": rm.distance, "duration": rm.duration})
                break
        if len(station_ok) >= 3:
            break
    classification = "unknown"
    if solo.feasible or station_ok:
        classification = "route_feasible_under_converter_search_or_construction_issue"
    elif e0c > Q + 1e-9:
        classification = "direct_arc_exceeds_battery_under_unrounded_euclidean"
    elif e0c + ec0 > Q + 1e-9 and not station_ok:
        classification = (
            "round_trip_or_station_insert_infeasible_in_small_probe_under_unrounded_euclidean"
            " — not a formal proof of mathematical infeasibility of the official instance"
        )
    else:
        classification = "probe_failed_for_other_TW_or_timing_reasons_not_proven_infeasible"

    return {
        "customer": cid,
        "ready": node.ready_time,
        "due": node.due_time,
        "service": node.service_time,
        "demand": node.demand,
        "energy_depot_to_cust": e0c,
        "energy_cust_to_depot": ec0,
        "Q": Q,
        "solo_replay_route_feasible": bool(solo.feasible),
        "solo_failure": solo.failure,
        "solo_hard_violations": int(solo.hard_violations),
        "n_station_insert_feasible_in_probe": len(station_ok),
        "station_examples": station_ok[:3],
        "classification": classification,
    }


def diagnose_instance(name: str, missing_from_runs: list[int] | None = None) -> dict:
    raw = Path(DEFAULT_RAW) / f"{name}.txt"
    out = Path(DEFAULT_CONVERTED)
    if not (out / name / "evrp_instance.json").is_file():
        convert_one_txt(raw, out)
    parsed = parse_schneider_txt(raw)
    inst = load_instance(out / name)
    # If missing customers known from failed runs, probe them; else probe none
    probes = []
    for cid in missing_from_runs or []:
        probes.append(_probe_customer(inst, int(cid)))
    return {
        "instance": name,
        "family": name.split("_")[0],
        "Q": parsed["Q"],
        "C": parsed["C"],
        "r": parsed["r"],
        "g": parsed["g"],
        "v": parsed["v"],
        "num_customers": len(parsed["customers"]),
        "num_stations": len(parsed["stations"]),
        "distance_convention": "euclidean_unrounded; travel=distance/v; energy=distance*r",
        "probes": probes,
        "disclaimer": (
            "replay_route tests a single route's battery/TW/capacity walk. "
            "It does not mark other customers unserved. "
            "Failure of a small station-insert probe is not a mathematical proof of "
            "instance infeasibility under every distance-rounding convention."
        ),
    }


def rebuild_main30_stats() -> dict:
    store = JobStore(reviewer_root() / "schneider")
    all_rows = store.collect_rows(feasible_only=False)
    rows30 = [r for r in all_rows if abs(float(r.get("budget_s") or 0) - 30.0) < 1e-9]
    # Deduplicate job keys
    uniq = {}
    for r in rows30:
        key = (r.get("method"), r.get("instance"), int(float(r.get("seed"))), 30.0)
        uniq[key] = r
    rows30 = list(uniq.values())
    assert len(rows30) == 336, f"expected 336 primary 30s rows, got {len(rows30)}"
    feas = [r for r in rows30 if str(r.get("feasible")).lower() in {"true", "1"}]
    infeas = [r for r in rows30 if str(r.get("feasible")).lower() not in {"true", "1"}]
    out = reviewer_root() / "schneider"
    metrics = (
        "k", "distance", "duration", "charging_visits", "charged_energy",
        "enroute_purchased_energy", "propulsion_energy", "alns_iterations", "solver_runtime_s",
    )
    write_csv(out / "table_main30_means.csv", aggregate_means(feas, ("method",), metrics))
    write_csv(out / "table_main30_means_by_family.csv", aggregate_means(feas, ("method", "family"), metrics))
    write_csv(out / "table_main30_feasibility.csv", [
        {
            "instance": r.get("instance"), "family": r.get("family"), "method": r.get("method"),
            "seed": r.get("seed"), "feasible": r.get("feasible"), "unserved": r.get("unserved"),
            "k": r.get("k"), "distance": r.get("distance"),
        }
        for r in rows30
    ])
    # Feasibility rates by family/method
    from collections import defaultdict
    rates = defaultdict(lambda: {"n": 0, "feas": 0})
    for r in rows30:
        key = (r.get("family"), r.get("method"))
        rates[key]["n"] += 1
        if str(r.get("feasible")).lower() in {"true", "1"}:
            rates[key]["feas"] += 1
    rate_rows = [
        {
            "family": k[0], "method": k[1], "n": v["n"], "n_feasible": v["feas"],
            "feasibility_rate": v["feas"] / v["n"] if v["n"] else None,
        }
        for k, v in sorted(rates.items())
    ]
    write_csv(out / "table_main30_feasibility_rates.csv", rate_rows)

    paired, summaries = [], []
    for metric in ("distance", "k", "charged_energy", "duration", "solver_runtime_s"):
        eff = paired_instance_effects(
            feas, method_a="ALNS-FR", method_b="ALNS-PR", metric=metric,
            tw="schneider", budget_s=30.0, require_budget=True,
        )
        paired.extend(eff)
        summaries.append({"metric": metric, "protocol": "30s_only_conditional_feasible", **summarize_paired(eff)})
    write_csv(out / "paired_main30_effects.csv", paired)
    write_csv(out / "paired_main30_summaries.csv", summaries)
    # Instances with both methods feasible on all matched seeds
    both = {e["instance"] for e in paired if e["metric"] == "distance"}
    status = {
        "n_planned_30s": 336,
        "n_rows_30s": len(rows30),
        "n_feasible": len(feas),
        "n_infeasible": len(infeas),
        "n_instances_both_methods_feasible_paired": len(both),
        "infeasible_instances": sorted({r.get("instance") for r in infeas}),
        "excluded_from_primary_stats": "non-30s pilot/selected records",
        "complete_protocol": len(rows30) == 336,
    }
    write_json(out / "main30_status.json", status)
    return status, infeas


def main() -> int:
    status, infeas = rebuild_main30_stats()
    # Probe one unserved customer per infeasible instance via a fresh greedy? We don't have
    # which customer — use investigation on known hard names with scan for battery-tight customers.
    hard = sorted({r.get("instance") for r in infeas})
    findings = []
    for name in hard:
        # Identify customers that cannot solo-route under converter
        raw = Path(DEFAULT_RAW) / f"{name}.txt"
        out = Path(DEFAULT_CONVERTED)
        if not (out / name / "evrp_instance.json").is_file():
            convert_one_txt(raw, out)
        inst = load_instance(out / name)
        bad = []
        for cid in map(int, inst.customer_indices):
            rm = replay_route(inst, [0, cid, 0], [None, None, None])
            if not rm.feasible:
                # also check any station insert
                ok = False
                for s in map(int, inst.charging_station_indices):
                    for nodes in ([0, s, cid, 0], [0, cid, s, 0], [0, s, cid, s, 0]):
                        if replay_route(inst, nodes, [None] * len(nodes)).feasible:
                            ok = True
                            break
                    if ok:
                        break
                if not ok:
                    bad.append(cid)
                    if len(bad) >= 5:
                        break
        findings.append(diagnose_instance(name, missing_from_runs=bad[:3]))
        print(name, "n_customers_failing_small_route_probe", len(bad), "examples", bad[:5], flush=True)

    outp = reviewer_root() / "schneider" / "feasibility_investigation_round4.json"
    write_json(outp, {
        "main30_status": status,
        "instance_findings": findings,
        "method_note": (
            "Individual-route feasibility uses src.hanif.replay.replay_route. "
            "Earlier Round-3 try_solo_route used replay_solution on a one-route solution, "
            "which incorrectly counted the other 99 customers as unserved."
        ),
    })
    print(json.dumps(status, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
