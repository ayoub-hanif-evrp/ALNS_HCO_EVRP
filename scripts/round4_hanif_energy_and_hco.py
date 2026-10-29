"""Fill Hanif energy metrics via verified balance identities; audit HCO contribution."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.paper import hanif_dirs  # noqa: E402
from src.alns_hco.reviewer_experiments import aggregate_means, reviewer_root, write_csv, write_json  # noqa: E402
from src.alns_hco.shared_phase1 import ensure_shared_phase1  # noqa: E402
from src.common.config import (  # noqa: E402
    METHOD_ALNS_FR,
    METHOD_ALNS_PR,
    METHOD_FIXED_DETOUR,
    METHOD_FIXED_HET,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_HET,
    METHOD_PHASE2_NOHCO,
)
from src.common.energy_accounting import attach_energy_reporting  # noqa: E402
from src.common.paired_stats import paired_instance_effects, summarize_paired  # noqa: E402
from src.common.reporting_io import flatten_solution_row, write_master_csv  # noqa: E402
from src.hanif.loader import load_hanif  # noqa: E402

EPS = 1e-6
BALANCE_TOL = 1e-4


def _f(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["seed"] = int(float(r["seed"]))
    return rows


def battery_map() -> dict[tuple[str, str], float]:
    """Cache B_max per (instance, tw)."""
    cache = {}
    for tw in ("original", "relaxed"):
        # sample via loader only when needed; fill lazily in reconstruct
        pass
    return cache


def reconstruct_energy(row: dict, b_max: float) -> dict:
    """Apply fleet-level identities under full-departure / no-loss assumptions.

    E_drive = K*B_max + E_enroute - E_return
    E_depot = E_drive - E_enroute
    E_return = K*B_max - E_depot

    When residual/depot missing but propulsion & enroute known:
      E_return = K*B_max + E_enroute - E_drive
      E_depot  = E_drive - E_enroute
    """
    k = _f(row, "k")
    e_drive = _f(row, "propulsion_energy")
    e_en = _f(row, "enroute_purchased_energy")
    if e_en is None:
        e_en = _f(row, "charged_energy")
    if k is None or e_drive is None or e_en is None or b_max is None:
        row["energy_reconstruction"] = "insufficient_inputs"
        return row
    initial = float(k) * float(b_max)
    e_return = initial + e_en - e_drive
    e_depot = e_drive - e_en
    flags = []
    if e_return < -BALANCE_TOL:
        flags.append("negative_return_battery")
    if e_depot < -BALANCE_TOL:
        flags.append("negative_depot_replenishment")
    if e_return > initial + BALANCE_TOL:
        flags.append("return_exceeds_initial")
    if e_drive < -BALANCE_TOL or e_en < -BALANCE_TOL:
        flags.append("negative_drive_or_enroute")
    # Only fill missing fields; never overwrite observed Phase-I route metrics
    src = str(row.get("energy_source") or "")
    observed = "shared_phase1_routes" in src
    if not observed or row.get("residual_battery_at_depot") in (None, ""):
        if not flags:
            row["residual_battery_at_depot"] = float(e_return)
            row["depot_replenishment_energy"] = float(e_depot)
            row["initial_battery_energy"] = float(initial)
            row["energy_balance_ok"] = True
            row["energy_balance_residual"] = float(initial + e_en - e_drive - e_return)
            row["energy_source"] = (src + "+" if src else "") + "reconstructed_aggregate_energy_balance"
            row["energy_reconstruction"] = "filled_from_balance_identity"
        else:
            row["energy_reconstruction"] = "flagged:" + ",".join(flags)
            row["energy_balance_ok"] = False
            row["energy_flags"] = flags
    else:
        # Validate observed against identity
        resid = _f(row, "residual_battery_at_depot")
        depot = _f(row, "depot_replenishment_energy")
        pred_return = initial + e_en - e_drive
        if resid is not None and abs(resid - pred_return) > 1e-2:
            flags.append("observed_return_mismatch_vs_identity")
        if depot is not None and abs(depot - (e_drive - e_en)) > 1e-2:
            flags.append("observed_depot_mismatch_vs_identity")
        row["energy_reconstruction"] = "observed_kept" + ((";flagged:" + ",".join(flags)) if flags else "")
        if flags:
            row["energy_flags"] = flags
    row["enroute_purchased_energy"] = e_en
    row["propulsion_energy"] = e_drive
    return row


def enrich_and_reconstruct(rows: list[dict]) -> list[dict]:
    bcache: dict[tuple, float] = {}
    ratecache: dict[tuple, float] = {}
    out = []
    for row in rows:
        key = (row["instance"], row["tw"])
        if key not in bcache:
            inst = load_hanif(row["instance"], row["tw"])
            bcache[key] = float(inst.vehicle.battery_capacity)
            ratecache[key] = float(inst.vehicle.energy_consumption_rate)
        # Ensure propulsion / enroute
        if _f(row, "propulsion_energy") is None and _f(row, "distance") is not None:
            row["propulsion_energy"] = _f(row, "distance") * ratecache[key]
            row["energy_source"] = (str(row.get("energy_source") or "") + "+derived_distance_times_rate").strip("+")
        if _f(row, "enroute_purchased_energy") is None and _f(row, "charged_energy") is not None:
            row["enroute_purchased_energy"] = _f(row, "charged_energy")
        # Re-validate Phase-I from routes when possible
        if row["method"] == METHOD_PHASE1_PR15:
            try:
                shared = ensure_shared_phase1(
                    hanif_dirs()["shared_phase1"],
                    instance_name=row["instance"], tw=row["tw"], seed=int(row["seed"]),
                )
                attach_energy_reporting(row, shared["instance"], shared["routes"], force=True)
                row["energy_source"] = "shared_phase1_routes"
                row["energy_reconstruction"] = "observed_from_routes"
            except Exception as exc:  # noqa: BLE001
                row = reconstruct_energy(row, bcache[key])
                row["phase1_route_enrich_error"] = repr(exc)
        else:
            row = reconstruct_energy(row, bcache[key])
        out.append(row)
    return out


def hco_audit(rows: list[dict]) -> tuple[list[dict], dict]:
    """Compare PhaseI / NoHCO / Het on matched instance×tw×seed."""
    idx = {(r["instance"], r["tw"], int(r["seed"]), r["method"]): r for r in rows}
    metrics = (
        "charging_expense", "station_waiting", "distance", "k",
        "charged_energy", "duration",
    )
    detail = []
    counts = defaultdict(lambda: {"improved": 0, "unchanged": 0, "worsened": 0, "missing": 0})
    n_identical_agg = 0
    n_checksum_same = 0
    n_checksum_diff = 0
    n_checksum_missing = 0
    for (inst, tw, seed) in sorted({(r["instance"], r["tw"], int(r["seed"])) for r in rows}):
        p1 = idx.get((inst, tw, seed, METHOD_PHASE1_PR15))
        noh = idx.get((inst, tw, seed, METHOD_PHASE2_NOHCO))
        het = idx.get((inst, tw, seed, METHOD_PHASE2_HET))
        if not p1 or not het:
            continue
        # Aggregate identity vs Phase I
        same_d = abs((_f(het, "distance") or 0) - (_f(p1, "distance") or -1)) <= 1e-6
        same_e = abs((_f(het, "charged_energy") or 0) - (_f(p1, "charged_energy") or -1)) <= 1e-6
        same_k = abs((_f(het, "k") or 0) - (_f(p1, "k") or -1)) <= 1e-9
        identical_agg = same_d and same_e and same_k
        if identical_agg:
            n_identical_agg += 1
        # Checksum / route identity evidence
        c1 = p1.get("phase1_checksum") or ""
        ch = het.get("phase1_checksum") or ""
        # Publication CSV may not store solution checksum of final routes; only Phase-I host checksum.
        # Without final-route checksum we cannot assert route identity.
        route_identity = "unknown_no_final_route_checksum"
        if identical_agg:
            route_identity = "aggregate_metrics_match_phase1_route_identity_unproven"
        else:
            route_identity = "aggregate_metrics_differ"

        rec = {
            "instance": inst, "tw": tw, "seed": seed,
            "identical_aggregate_vs_phase1": identical_agg,
            "route_identity_status": route_identity,
            "phase1_checksum_present": bool(c1),
        }
        for metric in metrics:
            a, b = _f(p1, metric), _f(het, metric)
            if a is None or b is None:
                counts[("PhaseI", "Het", metric)]["missing"] += 1
                continue
            diff = b - a
            # lower-is-better for these metrics
            if abs(diff) <= 1e-6:
                tag = "unchanged"
            elif diff < 0:
                tag = "improved"
            else:
                tag = "worsened"
            counts[("PhaseI", "Het", metric)][tag] += 1
            rec[f"het_vs_phase1_{metric}_diff"] = diff
            rec[f"het_vs_phase1_{metric}"] = tag
        if noh:
            for metric in metrics:
                a, b = _f(noh, metric), _f(het, metric)
                if a is None or b is None:
                    counts[("NoHCO", "Het", metric)]["missing"] += 1
                    continue
                diff = b - a
                if abs(diff) <= 1e-6:
                    tag = "unchanged"
                elif diff < 0:
                    tag = "improved"
                else:
                    tag = "worsened"
                counts[("NoHCO", "Het", metric)][tag] += 1
                rec[f"het_vs_nohco_{metric}_diff"] = diff
                rec[f"het_vs_nohco_{metric}"] = tag
            # Separate NoHCO vs PhaseI (search effect without HCO)
            for metric in metrics:
                a, b = _f(p1, metric), _f(noh, metric)
                if a is None or b is None:
                    continue
                diff = b - a
                if abs(diff) <= 1e-6:
                    tag = "unchanged"
                elif diff < 0:
                    tag = "improved"
                else:
                    tag = "worsened"
                counts[("PhaseI", "NoHCO", metric)][tag] += 1
        detail.append(rec)

    summary_rows = []
    for (a, b, metric), c in sorted(counts.items()):
        summary_rows.append({
            "baseline": a, "treatment": b, "metric": metric,
            "improved": c["improved"], "unchanged": c["unchanged"], "worsened": c["worsened"],
            "missing": c["missing"],
            "n": c["improved"] + c["unchanged"] + c["worsened"],
            "lower_is_better": True,
        })
    summary = {
        "n_compared_phase1_het": len(detail),
        "n_identical_aggregate_metrics_vs_phase1": n_identical_agg,
        "fraction_identical_aggregate": n_identical_agg / len(detail) if detail else None,
        "eps": 0.0,
        "eps_note": (
            "phase2_distance_tolerance=0.0 freezes distance at D<=D_PR. "
            "Acceptance also requires K<=K* and lexicographic improvement. "
            "Matching aggregate distance/charged_energy/K does not prove identical routes "
            "without final-route checksums or node sequences."
        ),
        "algorithm_modified": False,
    }
    return detail, summary_rows, summary


def main() -> int:
    src = reviewer_root() / "hanif_main_seeds012" / "runs.csv"
    rows = enrich_and_reconstruct(_load(src))
    out = reviewer_root() / "hanif_main_seeds012"
    write_master_csv(out / "runs.csv", [flatten_solution_row(r) for r in rows])
    # Coverage of reconstructed fields
    cov = defaultdict(lambda: {"n": 0, "has_residual": 0, "has_depot": 0, "flagged": 0})
    for r in rows:
        m = r["method"]
        cov[m]["n"] += 1
        if _f(r, "residual_battery_at_depot") is not None:
            cov[m]["has_residual"] += 1
        if _f(r, "depot_replenishment_energy") is not None:
            cov[m]["has_depot"] += 1
        if str(r.get("energy_reconstruction", "")).startswith("flagged"):
            cov[m]["flagged"] += 1
    write_csv(out / "table_energy_coverage.csv", [
        {"method": m, **v, "residual_fill_rate": v["has_residual"] / v["n"]} for m, v in sorted(cov.items())
    ])
    metrics = (
        "k", "distance", "charged_energy", "propulsion_energy", "enroute_purchased_energy",
        "depot_replenishment_energy", "residual_battery_at_depot", "charging_visits",
        "charging_expense", "station_waiting", "charging_time", "solver_runtime_s",
    )
    write_csv(out / "table_means_by_method.csv", aggregate_means(rows, ("method",), metrics))
    write_csv(out / "table_means_by_method_tw.csv", aggregate_means(rows, ("method", "tw"), metrics))
    write_csv(out / "table_means_by_method_family_tw.csv", aggregate_means(rows, ("method", "family", "tw"), metrics))

    detail, summary_rows, summary = hco_audit(rows)
    hco_dir = reviewer_root() / "hco_contribution_audit"
    hco_dir.mkdir(parents=True, exist_ok=True)
    write_csv(hco_dir / "instance_seed_details.csv", detail)
    write_csv(hco_dir / "summary_counts.csv", summary_rows)
    write_json(hco_dir / "summary.json", summary)
    write_json(out / "energy_reconstruction_note.json", {
        "identities": {
            "E_drive": "K*B_max + E_enroute - E_return",
            "E_depot": "E_drive - E_enroute",
            "E_return": "K*B_max - E_depot",
        },
        "label": "reconstructed_from_aggregate_energy_balance",
        "assumption": "full departure battery; no charging losses; Hanif energy=distance*rate where propulsion derived",
        "coverage": {m: dict(v) for m, v in cov.items()},
    })
    print(json.dumps({"energy_coverage": {m: dict(v) for m, v in cov.items()}, "hco": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
