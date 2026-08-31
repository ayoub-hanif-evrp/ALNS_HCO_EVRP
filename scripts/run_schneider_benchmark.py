"""Run CARE-FI Stage 1 on the 56 Schneider EVRPTW instances (FR and/or PR).

Compares lexicographic (K, D) against published BKS values. Does not reimplement
Schneider VNS-TS or Keskin ALNS.

  python scripts/run_schneider_benchmark.py --mode both --budget 180 --workers 8 --out outputs/_tmp/schneider
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schneider.data.instance_class import instance_class  # noqa: E402
from src.schneider.data.instance_loader import apply_battery_capacity, load_instance  # noqa: E402
from src.schneider.data.schneider_txt_convert import (  # noqa: E402
    DEFAULT_CONVERTED,
    LARGE_56,
    ensure_schneider_converted,
    schneider_family,
)
from src.schneider.env.cost import CostWeights  # noqa: E402
from src.schneider.evaluation.objective_reference import attach_reference  # noqa: E402
from src.schneider.evaluation.output_paths import FROZEN_RESULT_DIRS, OUTPUTS, SCHNEIDER_OUT  # noqa: E402
from src.schneider.paths import TMP_OUT  # noqa: E402
from src.schneider.evaluation.schneider_bks import BKS_CITATION, compare_row, summarize  # noqa: E402
from src.schneider.evaluation.schneider_incumbent import load_best_incumbent  # noqa: E402
from src.schneider.search.fleet_search import capacity_lower_bound, compute_fleet_bounds  # noqa: E402
from src.schneider.solver import solve_schneider  # noqa: E402
from src.schneider.utils.config import load_config  # noqa: E402

PORTIONS = {
    "fr": [1.0],
    "pr": [0.10, 0.20, 0.35, 0.50, 0.75, 1.0],
}


def _serialize_routes(routes: list) -> list[dict]:
    out = []
    for nodes, labels in routes:
        out.append({
            "nodes": [int(n) for n in nodes],
            "labels": [None if x is None else float(x) for x in (labels or [None] * len(nodes))],
        })
    return out


def _solve_one(
    *,
    name: str,
    root: Path,
    cfg: dict,
    weights: CostWeights,
    budget: float,
    warm_routes: list | None,
    routes_dir: Path | None,
) -> dict:
    inst = load_instance(root / name, load_capacity=None)
    q = (cfg.get("problem") or {}).get("battery_capacity")
    if q is not None:
        inst = apply_battery_capacity(inst, float(q))
    cfg = attach_reference(
        inst, cfg, weights,
        reference_dir=Path(ref) if (ref := (cfg.get("experiment") or {}).get("objective_reference_dir")) else None,
        variant=str((cfg.get("dataset") or {}).get("tw_variant") or ""),
    )
    lo, hi = compute_fleet_bounds(inst, cfg)
    t0 = time.perf_counter()
    result = solve_schneider(
        inst,
        config=cfg,
        weights=weights,
        budget_seconds=float(budget),
        seed=int(cfg.get("seed") or 0),
        warm_start_routes=warm_routes,
    )
    served = int(inst.num_customers) - int(result.score.unserved_customer_count)
    exact = result.exact_check or {}
    method_name = str((cfg.get("experiment") or {}).get("method_name") or "CARE-FI")
    row = {
        "instance": name,
        "class": instance_class(name),
        "family": name.split("_")[0],
        "n_customers": int(inst.num_customers),
        "capacity_lb": int(capacity_lower_bound(inst)),
        "search_lo": int(lo),
        "search_hi": int(hi),
        "feasible": bool(result.feasible),
        "served": served,
        "used_k": int(result.m_star) if result.feasible else int(result.score.used_evs),
        "distance": float(result.score.total_distance),
        "hard_violations": int(exact.get("hard_violations") or result.score.hard_violation_count),
        "served_ratio": served / max(int(inst.num_customers), 1),
        "runtime_s": time.perf_counter() - t0,
        "method": method_name,
        "warm_start": bool(((result.stages or {}).get("warm_start") or {}).get("accepted")),
    }
    if routes_dir is not None:
        routes_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "instance": name,
            "feasible": bool(result.feasible),
            "used_k": row["used_k"],
            "exact_check": exact,
            "routes": _serialize_routes(result.routes),
            "method": method_name,
        }
        (routes_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return row


def _refuse(out: Path) -> None:
    resolved = out.resolve()
    if resolved == OUTPUTS.resolve():
        raise SystemExit("Refusing to overwrite the frozen outputs/ archive")
    frozen_pr = OUTPUTS / "schneider" / "pr"
    frozen_fr = OUTPUTS / "schneider" / "fr"
    if resolved == frozen_pr.resolve() or frozen_pr.resolve() in resolved.parents:
        raise SystemExit("Refusing to overwrite frozen outputs/schneider/pr")
    if resolved == frozen_fr.resolve() or frozen_fr.resolve() in resolved.parents:
        raise SystemExit("Refusing to overwrite frozen outputs/schneider/fr")
    if resolved == SCHNEIDER_OUT.resolve() or resolved == (OUTPUTS / "hanif").resolve():
        raise SystemExit("Write new Schneider runs under outputs/_tmp/schneider, not the frozen archive")
    for p in FROZEN_RESULT_DIRS:
        fr = p.resolve()
        if resolved == fr or fr in resolved.parents:
            raise SystemExit(f"Refusing to overwrite frozen results: {p.name}")


def _worker(job: dict) -> dict:
    cfg = job["cfg"]
    warm, _meta = load_best_incumbent(job["name"], job["mode"])
    row = _solve_one(
        name=job["name"],
        root=Path(job["root"]),
        cfg=cfg,
        weights=CostWeights.from_eval_config(cfg),
        budget=float(job["budget"]),
        warm_routes=warm,
        routes_dir=Path(job["routes_dir"]),
    )
    row["family"] = schneider_family(job["name"])
    row["class"] = instance_class(job["name"])
    row["mode"] = job["mode"]
    row["charging_policy"] = "full_recharge" if job["mode"] == "fr" else "partial_recharge_portions"
    cmp_ = compare_row(job["mode"], job["name"], int(row["used_k"]), float(row["distance"]))
    row.update({
        "bks_source": cmp_["bks_source_name"],
        "bks_k": cmp_["bks_k"],
        "bks_d": cmp_["bks_d"],
        "delta_k": cmp_["delta_k"],
        "distance_gap_%": cmp_["distance_gap_%"],
        "verdict": cmp_["verdict"],
        "literature_source": cmp_["literature_source_name"],
        "literature_k": cmp_["literature_k"],
        "literature_d": cmp_["literature_d"],
    })
    print(
        f"  [{job['mode']}] {job['name']} feas={row['feasible']} "
        f"K={row['used_k']} (BKS {cmp_['bks_k']}, dK={cmp_['delta_k']:+d}) "
        f"D={row['distance']:.2f} t={row['runtime_s']:.1f}s",
        flush=True,
    )
    return row


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _run_mode(mode: str, *, names: list[str], cfg: dict, budget: float, workers: int, out: Path) -> list[dict]:
    portions = PORTIONS[mode]
    cfg = dict(cfg)
    problem = dict(cfg.get("problem") or {})
    problem["charging_portions"] = list(portions)
    problem["allow_partial_charging"] = mode == "pr"
    # Full-recharge BKS assume a complete refill. Just-enough charge is PR-only.
    problem["greedy_min_charge"] = bool(mode == "pr" and problem.get("greedy_min_charge", True))
    cfg["problem"] = problem
    cfg.setdefault("experiment", {})["method_name"] = f"CARE-FI-{mode.upper()}"
    cfg.setdefault("dataset", {})["path"] = str(DEFAULT_CONVERTED)
    root = ROOT / DEFAULT_CONVERTED
    routes_dir = out / mode / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        {
            "name": name,
            "root": str(root),
            "cfg": cfg,
            "budget": budget,
            "routes_dir": str(routes_dir),
            "mode": mode,
        }
        for name in names
    ]
    rows: list[dict] = []
    t0 = time.perf_counter()
    if workers <= 1:
        for job in jobs:
            rows.append(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())
    wall = time.perf_counter() - t0
    rows = sorted(rows, key=lambda r: r["instance"])
    summary = summarize(rows)
    summary.update({
        "mode": mode,
        "charging_portions": portions,
        "budget_s": budget,
        "wall_s": wall,
        "mean_k": sum(float(r["used_k"]) for r in rows) / len(rows) if rows else None,
        "mean_distance": sum(float(r["distance"]) for r in rows) / len(rows) if rows else None,
        "mean_runtime_s": sum(float(r["runtime_s"]) for r in rows) / len(rows) if rows else None,
    })
    mode_dir = out / mode
    _write_csv(mode_dir / "results.csv", rows)
    (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "citation"}, indent=2), flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/schneider_fps.yaml")
    ap.add_argument("--mode", choices=["fr", "pr", "both"], default="both")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(TMP_OUT / "schneider"))
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--pilot", action="store_true", help="Six family representatives")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    _refuse(out)
    out.mkdir(parents=True, exist_ok=True)

    names = list(args.instances) if args.instances else list(LARGE_56)
    if args.pilot:
        names = ["c101_21", "c201_21", "r101_21", "r201_21", "rc101_21", "rc201_21"]
    names = [n if n.endswith("_21") or n.endswith(".txt") else f"{n}_21" for n in names]
    names = [Path(n).stem for n in names]

    ensure_schneider_converted(names=names)
    cfg = dict(load_config(str(ROOT / args.config)))
    cfg.setdefault("hybrid_search", {})["neural_guidance"] = {"enabled": False}
    cfg.setdefault("paper_objective", {})["mode"] = "classical_distance"

    modes = ["fr", "pr"] if args.mode == "both" else [args.mode]
    combined = []
    for mode in modes:
        print(f"=== Schneider {mode.upper()} ({len(names)} instances, budget={args.budget}s) ===", flush=True)
        combined.extend(_run_mode(
            mode, names=names, cfg=cfg, budget=args.budget, workers=args.workers, out=out,
        ))
    if len(modes) == 2:
        _write_csv(out / "results_both.csv", combined)
        (out / "README.txt").write_text(
            "CARE-FI vs published Schneider EVRPTW BKS.\n"
            f"{BKS_CITATION}\n"
            "Lexicographic: min K then min D. Distance gap only when Delta_K=0.\n"
            "FR = full recharge (portion=1). PR = discrete portions 0.25/0.50/0.75/1.0, not continuous Keskin recharge.\n"
            "Internal ALNS-style/VNS-style Hanif baselines are not used here.\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
