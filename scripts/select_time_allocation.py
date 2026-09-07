"""Select Phase-I / Phase-II time allocation on the predetermined development subset.

Candidates: (10,20), (15,15), (20,10) seconds.
Uses seed 0 and both TW variants for the first instance of each family.
Does NOT use the full 120-instance set.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.checksum import customer_checksum  # noqa: E402
from src.alns_hco.hyperparameters import HYPERPARAMETERS  # noqa: E402
from src.alns_hco.phase1 import solve_phase1  # noqa: E402
from src.alns_hco.solver import solve_alns_hco  # noqa: E402
from src.common.config import TW_VARIANTS, development_instances, experiment_config  # noqa: E402
from src.common.paths import hanif_dirs  # noqa: E402
from src.hanif.loader import load_hanif  # noqa: E402


def main() -> int:
    dirs = hanif_dirs()
    names = development_instances()
    seed = int(HYPERPARAMETERS.get("dev_seed", 0))
    allocations = [tuple(x) for x in HYPERPARAMETERS.get("time_allocations", [[10, 20], [15, 15], [20, 10]])]
    # Distance tolerance already frozen (or provisional).
    frozen_path = dirs["config"] / "frozen_experiment_config.json"
    eps = float(HYPERPARAMETERS["phase2_distance_tolerance"])
    if frozen_path.is_file():
        frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
        eps = float(frozen.get("phase2_distance_tolerance", eps))

    rows = []
    print(f"Dev instances: {names}", flush=True)
    print(f"Allocations: {allocations}", flush=True)
    print(f"eps={eps}, seed={seed}", flush=True)

    for p1_s, p2_s in allocations:
        for tw in TW_VARIANTS:
            for name in names:
                inst = load_hanif(name, tw)
                phase1 = solve_phase1(inst, time_limit_s=float(p1_s), seed=seed)
                routes = phase1["routes"]
                m1 = phase1["metrics"]
                chk = customer_checksum(inst, routes)
                k_star = int(phase1["k_star"])
                d_pr = float(m1.distance)
                result = solve_alns_hco(
                    inst, mode="phase2", phase1_s=0.0, phase2_s=float(p2_s), seed=seed,
                    start_routes=routes, hco_mode="heterogeneous",
                    k_star=k_star, d_pr=d_pr, phase1_checksum=chk,
                    distance_tolerance=eps,
                )
                m2 = result.metrics
                rec = {
                    "phase1_s": p1_s,
                    "phase2_s": p2_s,
                    "allocation": f"{int(p1_s)}/{int(p2_s)}",
                    "tw": tw,
                    "instance": name,
                    "family": name.split("_")[0],
                    "seed": seed,
                    "eps": eps,
                    "feasible_phase1": bool(m1.feasible),
                    "feasible_phase2": bool(m2.feasible),
                    "error": None,
                    "k_star": k_star,
                    "k_final": int(m2.used_k),
                    "distance_phase1": float(m1.distance),
                    "distance_phase2": float(m2.distance),
                    "distance_ratio": float(m2.distance) / max(float(m1.distance), 1e-12),
                    "Cch_phase1": float(m1.charging_expense),
                    "Cch_phase2": float(m2.charging_expense),
                    "delta_Cch": float(m2.charging_expense) - float(m1.charging_expense),
                    "Twait_phase1": float(m1.station_waiting),
                    "Twait_phase2": float(m2.station_waiting),
                    "Tch_phase1": float(m1.charging_time),
                    "Tch_phase2": float(m2.charging_time),
                    "phase2_runtime_s": (result.stages.get("phase2") or {}).get("runtime_s"),
                }
                rows.append(rec)
                print(
                    f"{rec['allocation']} {tw} {name}: feas={rec['feasible_phase2']} "
                    f"D_ratio={rec['distance_ratio']:.4f} dC={rec['delta_Cch']:.2f}",
                    flush=True,
                )

    out = dirs["analysis"] / "time_allocation_sensitivity.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Selection rule (documented and frozen before the full experiment):
    # 1. Keep allocations where every Phase-II run is feasible and K is preserved.
    # 2. Among those, choose the allocation with the lowest mean delta_Cch
    #    (PhaseII - PhaseI). Negative is better.
    # 3. Tie-break: prefer 15/15, then smaller |phase1-phase2| imbalance.
    by_alloc = {}
    for p1_s, p2_s in allocations:
        key = f"{int(p1_s)}/{int(p2_s)}"
        sub = [r for r in rows if r["allocation"] == key]
        ok = all(r["feasible_phase1"] and r["feasible_phase2"] and r["k_final"] == r["k_star"] for r in sub)
        mean_dc = sum(r["delta_Cch"] for r in sub) / max(len(sub), 1)
        mean_ratio = sum(r["distance_ratio"] for r in sub) / max(len(sub), 1)
        imbalance = abs(p1_s - p2_s)
        by_alloc[key] = {
            "ok": ok,
            "mean_delta_Cch": mean_dc,
            "mean_distance_ratio": mean_ratio,
            "imbalance": imbalance,
            "n": len(sub),
            "phase1_s": p1_s,
            "phase2_s": p2_s,
        }

    eligible = [k for k, v in by_alloc.items() if v["ok"]]
    if not eligible:
        eligible = list(by_alloc.keys())
        rule = "fallback_all_allocations"
    else:
        rule = "min_mean_delta_Cch_among_fully_feasible_then_prefer_15_15"

    def sort_key(k: str):
        v = by_alloc[k]
        prefer_1515 = 0 if k == "15/15" else 1
        return (v["mean_delta_Cch"], prefer_1515, v["imbalance"])

    chosen_key = sorted(eligible, key=sort_key)[0]
    chosen = by_alloc[chosen_key]

    HYPERPARAMETERS["phase1_time_limit_s"] = float(chosen["phase1_s"])
    HYPERPARAMETERS["phase2_time_limit_s"] = float(chosen["phase2_s"])

    cfg = experiment_config()
    if frozen_path.is_file():
        cfg.update(json.loads(frozen_path.read_text(encoding="utf-8")))
    cfg["shared_phase1_s"] = float(chosen["phase1_s"])
    cfg["phase2_s"] = float(chosen["phase2_s"])
    cfg["phase1_time_limit_s"] = float(chosen["phase1_s"])
    cfg["phase2_time_limit_s"] = float(chosen["phase2_s"])
    cfg["time_allocation"] = chosen_key
    cfg["time_allocation_rule"] = rule
    cfg["time_allocation_summary"] = by_alloc
    cfg["provisional"] = False
    frozen_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # Keep common.config constants in sync for this process; full runs read frozen JSON.
    import src.common.config as cfgmod
    cfgmod.PHASE1_S = float(chosen["phase1_s"])
    cfgmod.PHASE2_S = float(chosen["phase2_s"])

    print(f"Selected allocation {chosen_key} by rule={rule}", flush=True)
    print(json.dumps(by_alloc, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
