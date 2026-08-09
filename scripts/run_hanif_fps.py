"""Hanif Fleet Intensification runner (A + B with A->B warm starts).

Frozen archives (refused as --out):
  - outputs/hanif_fps/fast_feasible_baseline
  - outputs/hanif_fps_intensify
  - outputs/CurrentBestDeterministicHanif

Write new runs to a fresh folder, e.g. outputs/hanif_fps_intensify_v2.

Paper figures:
  python scripts/generate_hanif_paper_figures.py

Usage:
  python scripts/run_hanif_fps.py --all --variant both --budget 180 --out outputs/hanif_fps_run_YYYYmmdd
  python scripts/run_hanif_fps.py --pilot --variant both --budget 180
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hanif_convert import VARIANT_SPECS, ensure_hanif_variant_converted
from src.data.hanif_dataset import discover_hanif_instances
from src.data.instance_class import instance_class
from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.search.deterministic_fps import solve_deterministic_fps
from src.search.fleet_search import capacity_lower_bound, compute_fleet_bounds
from src.utils.config import load_config

PILOT = ["C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1"]


def _normalize_variant(v: str) -> str:
    key = str(v).strip().upper()
    if key in ("ORIGINAL", "ORIGINAL_TW", "VARIANT_A", "A"):
        return "A"
    if key in ("RELAXED", "RELAXED_TW", "VARIANT_B", "B"):
        return "B"
    if key == "BOTH":
        return "BOTH"
    raise SystemExit(f"Unknown --variant {v!r}; use A, B, or both")


def _serialize_routes(routes: list) -> list[dict]:
    out = []
    for nodes, labels in routes:
        out.append({
            "nodes": [int(n) for n in nodes],
            "labels": [None if x is None else float(x) for x in (labels or [None] * len(nodes))],
        })
    return out


def _deserialize_routes(payload: list[dict]) -> list:
    return [
        (list(r["nodes"]), list(r.get("labels") or [None] * len(r["nodes"])))
        for r in payload
    ]


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
    lo, hi = compute_fleet_bounds(inst, cfg)
    t0 = time.perf_counter()
    result = solve_deterministic_fps(
        inst,
        config=cfg,
        weights=weights,
        budget_seconds=float(budget),
        warm_start_routes=warm_routes,
    )
    served = int(inst.num_customers) - int(result.score.unserved_customer_count)
    intens = (result.stages or {}).get("hanif_fleet_intensify") or {}
    exact = result.exact_check or {}
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
        "paper_j": float(result.score.paper_objective),
        "charging_visits": int(result.charging_visits or exact.get("charging_visits") or 0),
        "waiting_time": float(getattr(result.score, "total_waiting_time", 0.0) or getattr(result.score, "waiting_time", 0.0) or 0.0),
        "intensify_initial_k": intens.get("initial_k"),
        "intensify_final_k": intens.get("final_k"),
        "intensify_eliminations": intens.get("eliminations"),
        "warm_start": bool(((result.stages or {}).get("warm_start") or {}).get("accepted")),
        "runtime_s": time.perf_counter() - t0,
    }
    if routes_dir is not None:
        routes_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "instance": name,
            "feasible": bool(result.feasible),
            "used_k": row["used_k"],
            "exact_check": exact,
            "stages": {
                k: result.stages.get(k)
                for k in ("warm_start", "hanif_fleet_intensify", "fleet_search")
                if result.stages and k in result.stages
            },
            "routes": _serialize_routes(result.routes),
            "score": {
                "total_distance": float(result.score.total_distance),
                "used_evs": int(result.score.used_evs),
                "unserved": int(result.score.unserved_customer_count),
                "paper_j": float(result.score.paper_objective),
            },
        }
        (routes_dir / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    return row


def _run_variant(
    *,
    variant: str,
    cfg_base: dict,
    names: list[str],
    budget: float,
    out_dir: Path,
    warm_routes_by_name: dict[str, list] | None = None,
) -> tuple[dict, dict[str, list]]:
    spec = VARIANT_SPECS[variant]
    cfg = dict(cfg_base)
    ds = dict(cfg.get("dataset") or {})
    ds["name"] = "hanif"
    ds["tw_variant"] = variant
    ds["path"] = str(spec["output"])
    ds["source_dir"] = str(spec["source"])
    ds["use_only_this_dataset"] = True
    cfg["dataset"] = ds
    cfg.setdefault("hanif_fleet_intensify", {})["enabled"] = True

    root = ensure_hanif_variant_converted(variant, output_dir=spec["output"], source_dir=spec["source"])
    out_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = out_dir / "routes"
    weights = CostWeights.from_eval_config(cfg)

    print(
        f"\n=== Hanif {spec['label']} ({spec['key']}) intensify  n={len(names)}  "
        f"budget={budget}s ===",
        flush=True,
    )
    rows = []
    saved_routes: dict[str, list] = {}
    t_all = time.perf_counter()
    for name in names:
        warm = (warm_routes_by_name or {}).get(name)
        print(f"\n>>> [{variant}] {name}" + (" (warm A->B)" if warm is not None else ""), flush=True)
        row = _solve_one(
            name=name, root=root, cfg=cfg, weights=weights, budget=budget,
            warm_routes=warm, routes_dir=routes_dir,
        )
        row["variant"] = spec["key"]
        row["tw_variant"] = variant
        rows.append(row)
        # Keep in-memory routes for A→B
        route_file = routes_dir / f"{name}.json"
        if route_file.is_file():
            saved_routes[name] = _deserialize_routes(
                json.loads(route_file.read_text(encoding="utf-8"))["routes"]
            )
        print(
            f"  feas={row['feasible']} served={row['served']}/{row['n_customers']} "
            f"K={row['used_k']} (elim {row['intensify_initial_k']}->{row['intensify_final_k']}, "
            f"n_elim={row['intensify_eliminations']}) "
            f"dist={row['distance']:.1f} t={row['runtime_s']:.1f}s",
            flush=True,
        )

    with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    summary = {
        "variant": spec["key"],
        "tw_variant": variant,
        "label": spec["label"],
        "method": "HanifFleetIntensification",
        "n": len(rows),
        "budget_s": budget,
        "wall_s": time.perf_counter() - t_all,
        "feasibility_rate": sum(1 for r in rows if r["feasible"]) / max(len(rows), 1),
        "mean_served_ratio": sum(r["served"] / max(r["n_customers"], 1) for r in rows) / max(len(rows), 1),
        "mean_k": sum(r["used_k"] for r in rows) / max(len(rows), 1),
        "mean_runtime_s": sum(r["runtime_s"] for r in rows) / max(len(rows), 1),
        "mean_eliminations": sum(int(r["intensify_eliminations"] or 0) for r in rows) / max(len(rows), 1),
        "hanif_reference_mean_k": 74.2 if variant == "A" else 63.7,
        "attributes_used": {
            "battery_Q": 600,
            "energy_h": 1.0,
            "charging_rate": True,
            "wait_mu_deterministic": True,
            "wait_sigma": False,
            "charging_price_in_objective": False,
            "density_score": False,
        },
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2), flush=True)
    return summary, saved_routes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hanif_fps.yaml")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--variant", default="both", help="A | B | both")
    ap.add_argument("--out", default="outputs/hanif_fps_intensify")
    args = ap.parse_args()

    out_root = ROOT / args.out
    protected = {
        (ROOT / "outputs/hanif_fps/fast_feasible_baseline").resolve(): "FastFeasibleBaseline",
        (ROOT / "outputs/CurrentBestDeterministicHanif").resolve(): "CurrentBestDeterministicHanif",
        (ROOT / "outputs/hanif_fps_intensify").resolve(): "hanif_fps_intensify (current best archive)",
    }
    if out_root.resolve() in protected:
        raise SystemExit(f"Refusing to overwrite frozen results: {protected[out_root.resolve()]}")

    cfg = dict(load_config(str(ROOT / args.config)))
    cfg.setdefault("hybrid_search", {})["neural_guidance"] = {"enabled": False}
    fs = dict(cfg.get("fleet_search") or {})
    fs["seeds"] = [0]
    fs["fast_mode"] = True
    fs["construction_backend"] = "dense"
    fs["prefer_expert_construct"] = False
    cfg["fleet_search"] = fs

    variant = _normalize_variant(args.variant)
    # Discover names from A (same instance ids for B).
    root_a = ensure_hanif_variant_converted("A")
    all_names = discover_hanif_instances(root_a)
    if args.instances:
        names = list(args.instances)
    elif args.all or (not args.pilot and not args.instances):
        names = all_names
    else:
        names = [n for n in PILOT if n in set(all_names)] or all_names[:6]

    summaries = []
    routes_a: dict[str, list] = {}

    if variant in ("A", "BOTH"):
        sum_a, routes_a = _run_variant(
            variant="A", cfg_base=cfg, names=names, budget=float(args.budget),
            out_dir=out_root / "original_tw",
        )
        summaries.append(sum_a)

    if variant in ("B", "BOTH"):
        warm = routes_a if variant == "BOTH" else None
        if warm is None:
            # Optional: load A routes from sibling folder if present
            a_routes_dir = out_root / "original_tw" / "routes"
            if a_routes_dir.is_dir():
                warm = {}
                for p in a_routes_dir.glob("*.json"):
                    warm[p.stem] = _deserialize_routes(
                        json.loads(p.read_text(encoding="utf-8"))["routes"]
                    )
        sum_b, _ = _run_variant(
            variant="B", cfg_base=cfg, names=names, budget=float(args.budget),
            out_dir=out_root / "relaxed_tw",
            warm_routes_by_name=warm,
        )
        summaries.append(sum_b)

    if len(summaries) > 1:
        rows_a = {r["instance"]: r for r in summaries[0]["rows"]}
        rows_b = {r["instance"]: r for r in summaries[1]["rows"]}
        regressions = []
        for name, ra in rows_a.items():
            rb = rows_b.get(name)
            if rb and ra["feasible"] and rb["feasible"] and int(rb["used_k"]) > int(ra["used_k"]):
                regressions.append({"instance": name, "k_a": ra["used_k"], "k_b": rb["used_k"]})
        combined = {
            "n_variants": 2,
            "budget_s": args.budget,
            "method": "HanifFleetIntensification",
            "milestone_targets": {"A_mean_k_max": 80, "B_mean_k_max": 70, "B_le_A": True},
            "b_worse_than_a": regressions,
            "b_worse_than_a_count": len(regressions),
            "variants": {
                s["tw_variant"]: {k: s[k] for k in s if k != "rows"} for s in summaries
            },
        }
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "summary_both.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
        all_rows = []
        for s in summaries:
            all_rows.extend(s["rows"])
        with (out_root / "results_both.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(r)
        print(json.dumps(combined, indent=2), flush=True)
        print(
            f"Milestone check: A meanK={summaries[0]['mean_k']:.2f} (<=80?)  "
            f"B meanK={summaries[1]['mean_k']:.2f} (<=70?)  "
            f"B>A count={len(regressions)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
