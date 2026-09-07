"""Select phase2_distance_tolerance on a predetermined development subset.

Uses the first instance of each family for both TW variants and seed 0.
Does NOT use the full 120-instance set for tuning.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.clean_experiments import solve_job  # noqa: E402
from src.alns_hco.hyperparameters import HYPERPARAMETERS  # noqa: E402
from src.common.config import (  # noqa: E402
    METHOD_PHASE1_PR15, METHOD_PHASE2_HET, TW_VARIANTS, development_instances,
    experiment_config,
)
from src.common.paths import hanif_dirs  # noqa: E402


def main() -> int:
    dirs = hanif_dirs()
    names = development_instances()
    seed = int(HYPERPARAMETERS.get("dev_seed", 0))
    candidates = list(HYPERPARAMETERS["epsilon_candidates"])
    out_rows = []

    print(f"Dev instances: {names}", flush=True)
    print(f"Epsilon candidates: {candidates}", flush=True)

    # Ensure shared Phase I once per (instance, tw)
    for tw in TW_VARIANTS:
        for name in names:
            row = solve_job({
                "name": name, "tw": tw, "method": METHOD_PHASE1_PR15, "seed": seed,
            })
            print(f"PhaseI {tw} {name}: feas={row.get('feasible')} K={row.get('k')} err={row.get('error')}", flush=True)

    for eps in candidates:
        for tw in TW_VARIANTS:
            for name in names:
                row = solve_job({
                    "name": name, "tw": tw, "method": METHOD_PHASE2_HET, "seed": seed,
                    "eps": float(eps),
                })
                p1 = solve_job({
                    "name": name, "tw": tw, "method": METHOD_PHASE1_PR15, "seed": seed,
                })
                d0 = p1.get("distance") or 0.0
                d1 = row.get("distance")
                rec = {
                    "eps": eps,
                    "tw": tw,
                    "instance": name,
                    "family": name.split("_")[0],
                    "feasible": row.get("feasible"),
                    "error": row.get("error"),
                    "k": row.get("k"),
                    "k_star": row.get("k_star"),
                    "distance": d1,
                    "distance_ratio": (None if d1 is None or not d0 else d1 / d0),
                    "charging_expense": row.get("charging_expense"),
                    "station_waiting": row.get("station_waiting"),
                    "charging_time": row.get("charging_time"),
                    "phase1_charging_expense": p1.get("charging_expense"),
                    "delta_Cch": (
                        None if row.get("charging_expense") is None or p1.get("charging_expense") is None
                        else row["charging_expense"] - p1["charging_expense"]
                    ),
                }
                out_rows.append(rec)
                print(
                    f"eps={eps} {tw} {name}: feas={rec['feasible']} "
                    f"D_ratio={rec['distance_ratio']} dC={rec['delta_Cch']} err={rec['error']}",
                    flush=True,
                )

    path = dirs["config"] / "epsilon_selection_dev.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    # Canonical analysis copy
    analysis = dirs["analysis"] / "epsilon_sensitivity.csv"
    analysis.parent.mkdir(parents=True, exist_ok=True)
    with analysis.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    # Selection rule (documented, frozen before full run):
    # 1. Keep only eps where all feasible runs satisfy D <= (1+eps)*D_PR and no errors.
    # 2. Among those, choose the eps with the best (lowest) mean delta_Cch.
    # 3. Tie-break: smaller eps.
    # Note: eps>0 may be needed so a better heterogeneous station with a slightly
    # longer detour can be accepted.
    by_eps = {}
    for eps in candidates:
        sub = [r for r in out_rows if r["eps"] == eps]
        ok = all(r.get("feasible") and not r.get("error") for r in sub)
        mean_dc = sum((r["delta_Cch"] or 0.0) for r in sub) / max(len(sub), 1)
        mean_ratio = sum((r["distance_ratio"] or 1.0) for r in sub) / max(len(sub), 1)
        by_eps[eps] = {"ok": ok, "mean_delta_Cch": mean_dc, "mean_distance_ratio": mean_ratio, "n": len(sub)}

    eligible = [e for e, v in by_eps.items() if v["ok"]]
    if not eligible:
        eligible = sorted(candidates, key=lambda e: (-sum(1 for r in out_rows if r["eps"] == e and r.get("feasible")), e))
        chosen = eligible[0]
        rule = "fallback_max_feasibility"
    else:
        chosen = sorted(eligible, key=lambda e: (by_eps[e]["mean_delta_Cch"], e))[0]
        rule = "min_mean_delta_Cch_among_fully_feasible"

    HYPERPARAMETERS["phase2_distance_tolerance"] = float(chosen)
    cfg = experiment_config()
    frozen_path = dirs["config"] / "frozen_experiment_config.json"
    if frozen_path.is_file():
        cfg.update(json.loads(frozen_path.read_text(encoding="utf-8")))
    cfg["phase2_distance_tolerance"] = float(chosen)
    cfg["epsilon_selection_rule"] = rule
    cfg["epsilon_selection_summary"] = {str(k): v for k, v in by_eps.items()}
    cfg["provisional"] = False
    cfg["development_instances"] = names
    cfg["development_seed"] = seed
    cfg["shared_phase1_s"] = 15.0
    cfg["phase2_s"] = 15.0
    cfg["phase1_time_limit_s"] = 15.0
    cfg["phase2_time_limit_s"] = 15.0
    cfg["time_allocation"] = "15/15"
    cfg["time_allocation_note"] = "Equal split of the common 30s budget; not claimed optimal."
    cfg["phase2_repair_shortlist_size"] = int(HYPERPARAMETERS.get("phase2_repair_shortlist_size", 5))
    cfg["hco_candidates_per_criterion"] = int(HYPERPARAMETERS["hco_candidates_per_criterion"])
    cfg["methods"] = list(experiment_config()["methods"])
    cfg["fixed_route_objective"] = "lexicographic (C_ch, T_wait, T_ch, D); non-worsening"
    frozen_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Selected eps={chosen} by rule={rule}", flush=True)
    print(json.dumps(by_eps, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
