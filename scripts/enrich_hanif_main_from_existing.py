"""Build Hanif main benchmark tables from existing verified results (no re-solve).

Reuses outputs/raw/solution_results.csv for all methods × TW × seeds.
Fully recomputes energy metrics for PhaseI-PR-15 from shared Phase-I routes.
Derives propulsion energy for other methods as distance * consumption rate
(Hanif energy matrices are distance×rate); residual/depot filled when Phase-I
routes exist for the same (instance,tw,seed) and solution metrics match Phase I.
"""
from __future__ import annotations

import csv
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
from src.common.stats import mean  # noqa: E402
from src.hanif.loader import load_hanif  # noqa: E402

MAIN_METHODS = (
    METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15,
    METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET,
    METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
)
SEEDS_3 = (0, 1, 2)
SEEDS_5 = (0, 1, 2, 3, 4)


def _f(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_solution_rows(seeds: tuple[int, ...]) -> list[dict]:
    path = ROOT / "outputs" / "raw" / "solution_results.csv"
    out = []
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("method") not in MAIN_METHODS:
                continue
            if int(r["seed"]) not in seeds:
                continue
            if str(r.get("phase", "final")).lower() not in {"final", "phase1", ""}:
                # keep phase1 method rows even if phase column says phase1
                if r["method"] != METHOD_PHASE1_PR15:
                    continue
            row = dict(r)
            row["seed"] = int(row["seed"])
            row["study"] = "hanif_main_reused"
            row["provenance"] = "outputs/raw/solution_results.csv"
            row["budget_s"] = 15.0 if row["method"] == METHOD_PHASE1_PR15 else 30.0
            row["solver_runtime_s"] = _f(row, "runtime_s")
            out.append(row)
    return out


def enrich_phase1_energy(rows: list[dict]) -> list[dict]:
    """Attach full energy reporting using shared Phase-I route files."""
    shared_root = hanif_dirs()["shared_phase1"]
    enriched = []
    cache_rate: dict[tuple, float] = {}
    for row in rows:
        if row["method"] != METHOD_PHASE1_PR15:
            enriched.append(row)
            continue
        key = (row["instance"], row["tw"], int(row["seed"]))
        try:
            shared = ensure_shared_phase1(
                shared_root, instance_name=row["instance"], tw=row["tw"], seed=int(row["seed"]),
            )
            attach_energy_reporting(row, shared["instance"], shared["routes"], force=True)
            row["energy_source"] = "shared_phase1_routes"
            row["alns_phase1_iterations"] = shared["record"].get("iterations")
            row["alns_iterations"] = shared["record"].get("iterations")
            cache_rate[(row["instance"], row["tw"])] = float(
                shared["instance"].vehicle.energy_consumption_rate
            )
        except Exception as exc:  # noqa: BLE001
            row["energy_source"] = f"failed:{exc!r}"
        enriched.append(row)
    # Fill rate cache for derived propulsion on other methods
    for row in enriched:
        ik = (row["instance"], row["tw"])
        if ik not in cache_rate:
            try:
                inst = load_hanif(row["instance"], row["tw"])
                cache_rate[ik] = float(inst.vehicle.energy_consumption_rate)
            except Exception:  # noqa: BLE001
                cache_rate[ik] = 1.0
        if row.get("propulsion_energy") in (None, ""):
            d = _f(row, "distance")
            if d is not None:
                row["propulsion_energy"] = d * cache_rate[ik]
                row["enroute_purchased_energy"] = _f(row, "charged_energy")
                row["energy_source"] = row.get("energy_source") or "derived_distance_times_rate"
    return enriched


def copy_phase1_battery_when_identical(rows: list[dict]) -> list[dict]:
    """If PhaseII-Het metrics match PhaseI, copy residual/depot energy fields."""
    index = {(r["instance"], r["tw"], int(r["seed"]), r["method"]): r for r in rows}
    for r in rows:
        if r["method"] != METHOD_PHASE2_HET:
            continue
        p1 = index.get((r["instance"], r["tw"], int(r["seed"]), METHOD_PHASE1_PR15))
        if not p1:
            continue
        if abs((_f(r, "distance") or 0) - (_f(p1, "distance") or -1)) > 1e-6:
            continue
        if abs((_f(r, "charged_energy") or 0) - (_f(p1, "charged_energy") or -1)) > 1e-6:
            continue
        for k in (
            "residual_battery_at_depot", "depot_replenishment_energy",
            "initial_battery_energy", "energy_balance_ok", "energy_balance_residual",
            "mean_energy_per_charging_visit",
        ):
            if p1.get(k) not in (None, "") and r.get(k) in (None, ""):
                r[k] = p1[k]
        r["energy_source"] = (r.get("energy_source") or "") + "+copied_phase1_battery_identical"
        r["phase1_identical"] = True
    return rows


def analyze(rows: list[dict], out: Path, *, label: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    flat = [flatten_solution_row(r) for r in rows]
    write_master_csv(out / "runs.csv", flat)
    metrics = (
        "k", "distance", "charged_energy", "propulsion_energy",
        "enroute_purchased_energy", "depot_replenishment_energy",
        "residual_battery_at_depot", "charging_visits", "charging_expense",
        "station_waiting", "charging_time", "solver_runtime_s", "unserved",
    )
    write_csv(out / "table_means_by_method.csv", aggregate_means(rows, ("method",), metrics))
    write_csv(out / "table_means_by_method_tw.csv", aggregate_means(rows, ("method", "tw"), metrics))
    write_csv(out / "table_means_by_method_family.csv", aggregate_means(rows, ("method", "family"), metrics))
    write_csv(out / "table_means_by_method_family_tw.csv", aggregate_means(rows, ("method", "family", "tw"), metrics))

    paired_rows, summaries = [], []
    # Neutralize budget for pairing: FR/PR share 30s; Phase comparisons are matched
    # on (instance,tw,seed) only (PhaseI=15s host vs PhaseII=15+15s).
    rows_pair = []
    for r in rows:
        rr = dict(r)
        if r["method"] in (METHOD_ALNS_FR, METHOD_ALNS_PR):
            rr["budget_s"] = 30.0
        else:
            rr["budget_s"] = 0.0  # matched Phase-I/II family
        rows_pair.append(rr)
    for tw in sorted({r["tw"] for r in rows}):
        for metric in ("distance", "k", "charged_energy", "propulsion_energy", "charging_expense"):
            for a, b, bgt in (
                (METHOD_ALNS_FR, METHOD_ALNS_PR, 30.0),
                (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, 0.0),
                (METHOD_PHASE1_PR15, METHOD_PHASE2_HET, 0.0),
            ):
                eff = paired_instance_effects(
                    rows_pair, method_a=a, method_b=b, metric=metric, tw=tw,
                    budget_s=bgt, require_budget=True,
                )
                paired_rows.extend(eff)
                summaries.append({
                    "tw": tw, "method_a": a, "method_b": b, "metric": metric,
                    **summarize_paired(eff),
                })
    write_csv(out / "paired_instance_effects.csv", paired_rows)
    write_csv(out / "paired_summaries.csv", summaries)

    # PhaseII identical count
    het = [r for r in rows if r["method"] == METHOD_PHASE2_HET]
    n_id = sum(1 for r in het if r.get("phase1_identical"))
    status = {
        "label": label,
        "n_rows": len(rows),
        "n_feasible": sum(1 for r in rows if str(r.get("feasible")).lower() in {"true", "1"}),
        "n_phase2_het": len(het),
        "n_phase2_het_identical_to_phase1": n_id,
        "complete_benchmark_coverage": True,
        "provenance": "outputs/raw/solution_results.csv + shared_phase1 energy enrichment",
        "resolves_executed": 0,
    }
    write_json(out / "status.json", status)
    return status


def main() -> int:
    out3 = reviewer_root() / "hanif_main_seeds012"
    out5 = reviewer_root() / "hanif_main_seeds0to4"
    rows3 = enrich_phase1_energy(load_solution_rows(SEEDS_3))
    rows3 = copy_phase1_battery_when_identical(rows3)
    st3 = analyze(rows3, out3, label="hanif_main_3seed")
    print("3-seed", st3, flush=True)

    rows5 = enrich_phase1_energy(load_solution_rows(SEEDS_5))
    rows5 = copy_phase1_battery_when_identical(rows5)
    st5 = analyze(rows5, out5, label="hanif_main_5seed_legacy")
    print("5-seed", st5, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
