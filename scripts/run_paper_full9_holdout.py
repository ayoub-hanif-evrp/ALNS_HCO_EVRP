"""Paper full-9 holdout evaluation: Deterministic-FPS ± P11 search-move guide.

Equal-budget fleet-minimized protocol (matches P10.5 PASS): 900s / instance.
Holdouts only — never touches the validation carve-out.

Usage:
  python scripts/run_paper_full9_holdout.py --budget 900 --seed 0 --methods both
  python scripts/run_paper_full9_holdout.py --budget 900 --seed 1 --methods both
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.instance_class import instance_class
from src.data.instance_loader import load_capacity_from_config, load_instance
from src.data.mixed_class_split import _is_type1_instance, _is_type2_instance
from src.env.cost import CostWeights
from src.search.deterministic_fps import solve_deterministic_fps
from src.search.fleet_search import capacity_lower_bound, compute_fleet_bounds
from src.utils.config import load_config

FULL9 = [
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
]
VALIDATION6 = ["c103_21", "c201_21", "r103_21", "r201_21", "rc103_21", "rc201_21"]

# Immutable frozen numbers for the paper table (do not recompute).
FROZEN_AGG = [
    {"method": "Expert", "served": 0.984, "feas": 0.667, "mean_J": 96950.5, "mean_used": None, "source": "full9_20260731"},
    {"method": "Static BC", "served": 0.920, "feas": 0.667, "mean_J": 161077.6, "mean_used": None, "source": "full9_20260731"},
    {"method": "Old RL-best", "served": 0.903, "feas": 0.667, "mean_J": 177774.5, "mean_used": None, "source": "full9_20260731"},
    {
        "method": "FixedFleet-Expert+Repair+ALNS (m=40)",
        "served": 0.998, "feas": 0.889, "mean_J": 62371.0, "mean_used": 39.78,
        "source": "expert_repair_alns_20260801 / fixed_fleet_expert_repair_alns",
        "note": "Fixed fleet m=40 — not fleet-minimized",
    },
]


def _tw_family(name: str) -> str:
    if _is_type1_instance(name):
        return "type1"
    if _is_type2_instance(name):
        return "type2"
    return "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()
    except Exception:
        return "unknown"


def _serialize_routes(routes) -> list:
    out = []
    for nodes, labels in routes:
        out.append({
            "nodes": [int(n) for n in nodes],
            "charge_labels": [None if x is None else float(x) for x in (labels or [])],
        })
    return out


def _prepare_config(base: dict, *, seed: int, enable_guide: bool, guide_ckpt: str) -> dict:
    config = dict(base)
    hs = dict(config.get("hybrid_search") or {})
    alns = dict(hs.get("alns") or {})
    alns["iterations"] = int(alns.get("iterations", 40))
    alns["time_limit_seconds"] = min(float(alns.get("time_limit_seconds", 25)), 40.0)
    hs["alns"] = alns
    cons = dict(hs.get("construction") or {})
    cons["expert_variants_list"] = ["edf", "regret", "energy", "fleet", "hybrid"]
    hs["construction"] = cons
    repair = dict(hs.get("repair") or {})
    repair["station_candidates"] = 2
    repair["max_rounds"] = 8
    hs["repair"] = repair

    ng = dict(hs.get("neural_guidance") or {})
    if enable_guide:
        ng["enabled"] = True
        ng["deprecated_bc_guide"] = False
        ng["checkpoint"] = guide_ckpt
    else:
        ng["enabled"] = False
    hs["neural_guidance"] = ng
    config["hybrid_search"] = hs

    fleet_cfg = dict(config.get("fleet_search") or {})
    base_seeds = [0, 1, 2, 3]
    fleet_cfg["seeds"] = sorted({0, *[s + int(seed) for s in base_seeds]})
    fleet_cfg["upper_bound_retries"] = 3
    fleet_cfg["retries_per_fleet"] = 4
    fleet_cfg["failures_before_stop"] = 2
    fleet_cfg["stop_upper_on_first_feasible"] = True
    fleet_cfg["cold_retry_after_warm_failure"] = True
    fleet_cfg["route_elimination"] = True
    fleet_cfg["set_partitioning_during_descent"] = True
    fleet_cfg["warm_start_from_larger_fleet"] = True
    fleet_cfg["prefer_expert_construct"] = True
    fleet_cfg["upper_expand_step"] = 6
    fleet_cfg["upper_expand_max"] = 48
    config["fleet_search"] = fleet_cfg
    return config


def _solve_one(inst, config, weights, budget: float) -> dict:
    lo, hi = compute_fleet_bounds(inst, config)
    math_lo = capacity_lower_bound(inst)
    result = solve_deterministic_fps(
        inst, config=config, weights=weights, budget_seconds=float(budget),
    )
    n_cust = int(inst.num_customers)
    served = n_cust - int(result.score.unserved_customer_count)
    return {
        "mathematical_lower_bound": int(math_lo),
        "heuristic_lower_estimate": int(result.lower_bound_m or lo),
        "upper_probe": int(result.upper_bound_m or hi),
        "m_star": int(result.m_star) if result.feasible else None,
        "complete_feasibility": bool(result.feasible),
        "served_customers": int(served),
        "total_customers": n_cust,
        "served_ratio": float(served) / max(n_cust, 1),
        "hard_violations": int(result.score.hard_violation_count),
        "used_vehicles": int(result.score.used_evs),
        "total_distance": float(result.score.total_distance),
        "charging_visits": int(result.charging_visits),
        "total_charging_time": float(result.score.total_charging_time),
        "total_duration": float(result.score.total_duration),
        "paper_j": float(result.score.paper_objective),
        "runtime": float(result.runtime_seconds),
        "within_budget_5pct": bool((result.budget_adherence or {}).get("within_5pct", False)),
        "termination": result.termination_reason,
        "fleet_minimization_performed": True,
        "fixed_fleet_used": False,
        "exact_check_ok": bool((result.exact_check or {}).get("ok", False)),
        "stages": result.stages,
        "fleet_trace": result.fleet_trace,
        "routes": result.routes,
        "exact_check": result.exact_check,
        "alns_telemetry": ((result.stages or {}).get("alns") or {}).get("telemetry"),
    }


def _agg(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "feasibility_rate": sum(1.0 if r["complete_feasibility"] else 0.0 for r in rows) / n,
        "mean_served": sum(r["served_ratio"] for r in rows) / n,
        "mean_used_vehicles": sum(r["used_vehicles"] for r in rows) / n,
        "mean_distance": sum(r["total_distance"] for r in rows) / n,
        "mean_J": sum(r["paper_j"] for r in rows) / n,
        "mean_runtime": sum(r["runtime"] for r in rows) / n,
        "all_within_budget_5pct": all(r["within_budget_5pct"] for r in rows),
        "exact_ok_rate": sum(1.0 if r["exact_check_ok"] else 0.0 for r in rows) / n,
    }


def _write_paper_table(out: Path, method_rows: dict[str, list[dict]], seed: int, budget: float) -> None:
    live_agg = []
    for method, rows in method_rows.items():
        a = _agg(rows)
        live_agg.append({
            "method": method,
            "served": a["mean_served"],
            "feas": a["feasibility_rate"],
            "mean_J": a["mean_J"],
            "mean_used": a["mean_used_vehicles"],
            "mean_runtime": a["mean_runtime"],
            "source": "live_full9_holdout",
        })

    lines = [
        "# Final method comparison (full-9 holdout)",
        "",
        "Selected after P10.5 + P11 validation gates: **Deterministic-FPS** "
        "(fleet-minimized) and optional **Det-FPS + Transformer search-move guide**.",
        "",
        f"Budget: {budget}s/instance | Seed: {seed} | Holdouts: {', '.join(FULL9)}",
        "",
        "## Frozen baselines (immutable)",
        "",
        "| Method | Served | Feas | Mean J | Mean used | Source |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in FROZEN_AGG:
        used = "—" if r["mean_used"] is None else f"{r['mean_used']:.2f}"
        lines.append(
            f"| {r['method']} | {r['served']:.3f} | {r['feas']:.3f} | {r['mean_J']:.1f} | {used} | {r['source']} |"
        )

    lines += [
        "",
        "## Live fleet-minimized solvers (this freeze)",
        "",
        "| Method | Served | Feas | Mean J | Mean used | Mean runtime | Source |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in live_agg:
        lines.append(
            f"| {r['method']} | {r['served']:.3f} | {r['feas']:.3f} | {r['mean_J']:.1f} | "
            f"{r['mean_used']:.2f} | {r['mean_runtime']:.1f}s | {r['source']} |"
        )

    for method, rows in method_rows.items():
        lines += [
            "",
            f"### Per-instance — {method}",
            "",
            "| Instance | Class | Feas | Served | m* | Used | Dist | J | Runtime | Within5% | Term |",
            "|---|---|:---:|---:|---:|---:|---:|---:|---:|:---:|---|",
        ]
        for r in rows:
            lines.append(
                f"| {r['instance']} | {r['customer_class']}/{r['tw_family']} | "
                f"{r['complete_feasibility']} | {r['served_customers']}/{r['total_customers']} | "
                f"{r['m_star']} | {r['used_vehicles']} | {r['total_distance']:.1f} | "
                f"{r['paper_j']:.1f} | {r['runtime']:.1f}s | {r['within_budget_5pct']} | "
                f"{r['termination']} |"
            )

    lines += [
        "",
        "## Honesty notes",
        "",
        "- FixedFleet-Expert+Repair+ALNS uses **fixed m=40** and is an ablation, not minimized fleet.",
        "- Deterministic-FPS / Det-FPS+Guide perform genuine descending fleet minimization.",
        "- Validation carve-out (6 instances) was used for method selection; this table is holdout-only.",
        "- P11 guide checkpoint: `outputs/tgfps_p11/transformer_search_guide.pt` (not BC).",
        "",
    ]
    (out / "method_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "method_comparison.json").write_text(
        json.dumps({"frozen": FROZEN_AGG, "live": live_agg, "rows": method_rows}, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/train_tgfps.yaml")
    ap.add_argument("--out", default="outputs/baselines/paper_full9_holdout")
    ap.add_argument("--methods", default="both", choices=["det", "guided", "both"])
    ap.add_argument(
        "--guide-checkpoint",
        default="outputs/tgfps_p11/transformer_search_guide.pt",
    )
    ap.add_argument("--instances", nargs="*", default=None, help="Optional subset of FULL9")
    args = ap.parse_args()

    names = list(args.instances) if args.instances else list(FULL9)
    assert not (set(names) & set(VALIDATION6)), "validation carve-out leaked into holdout eval"
    assert set(names).issubset(set(FULL9)), f"non-holdout instances: {set(names) - set(FULL9)}"

    methods = []
    if args.methods in ("det", "both"):
        methods.append(("Deterministic-FPS", False))
    if args.methods in ("guided", "both"):
        methods.append(("Det-FPS+SearchMoveGuide", True))

    guide_ckpt = str((ROOT / args.guide_checkpoint).resolve())
    if any(g for _, g in methods):
        if not Path(guide_ckpt).is_file():
            print(f"ERROR: guide checkpoint missing: {guide_ckpt}", flush=True)
            return 1

    base_cfg = dict(load_config(str(ROOT / args.config)))
    out_root = ROOT / args.out / f"seed_{args.seed}"
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"PAPER full-9 holdout eval seed={args.seed} budget={args.budget}", flush=True)
    print("instances:", names, flush=True)
    print("methods:", [m for m, _ in methods], flush=True)
    print("=" * 72, flush=True)

    method_rows: dict[str, list[dict]] = {m: [] for m, _ in methods}
    t_all = time.perf_counter()
    cap = load_capacity_from_config(base_cfg)
    root = Path(base_cfg["dataset"]["path"])

    for method_name, enable_guide in methods:
        cfg = _prepare_config(base_cfg, seed=args.seed, enable_guide=enable_guide, guide_ckpt=guide_ckpt)
        weights = CostWeights.from_eval_config(cfg)
        method_dir = out_root / ("det_fps" if not enable_guide else "det_fps_guided")
        for sub in ("routes", "fleet_search_traces", "replay_reports"):
            (method_dir / sub).mkdir(parents=True, exist_ok=True)
        (method_dir / "config.yaml").write_text(
            __import__("yaml").safe_dump(cfg, sort_keys=False), encoding="utf-8",
        )
        (method_dir / "source_hash.txt").write_text(
            f"git={_git_hash()}\nconfig_sha256={_sha256_file(method_dir / 'config.yaml')}\n"
            f"budget={args.budget}\nseed={args.seed}\nguide={enable_guide}\n"
            f"guide_ckpt={guide_ckpt if enable_guide else ''}\n",
            encoding="utf-8",
        )

        print(f"\n######## {method_name} ########", flush=True)
        rows = []
        for name in names:
            print(f"\n>>> {method_name} | {name}", flush=True)
            inst = load_instance(root / name, load_capacity=cap)
            got = _solve_one(inst, cfg, weights, args.budget)
            row = {
                "instance": name,
                "customer_class": instance_class(name),
                "tw_family": _tw_family(name),
                "method": method_name,
                **{k: v for k, v in got.items() if k not in ("routes", "fleet_trace", "exact_check", "stages")},
            }
            rows.append(row)
            method_rows[method_name].append(row)

            (method_dir / "fleet_search_traces" / f"{name}.json").write_text(
                json.dumps(got["fleet_trace"], indent=2, default=str), encoding="utf-8",
            )
            (method_dir / "routes" / f"{name}.json").write_text(
                json.dumps({
                    "instance": name, "m_star": got["m_star"], "feasible": got["complete_feasibility"],
                    "routes": _serialize_routes(got["routes"]),
                }, indent=2),
                encoding="utf-8",
            )
            (method_dir / "replay_reports" / f"{name}.json").write_text(
                json.dumps(got["exact_check"], indent=2, default=str), encoding="utf-8",
            )
            print(
                f"  feas={row['complete_feasibility']} served={row['served_customers']}/{row['total_customers']} "
                f"m*={row['m_star']} used={row['used_vehicles']} J={row['paper_j']:.1f} "
                f"t={row['runtime']:.1f}s within5={row['within_budget_5pct']} term={row['termination']}",
                flush=True,
            )

            # Incremental save so a crash still leaves partial paper rows.
            (method_dir / "summary_partial.json").write_text(
                json.dumps({"agg": _agg(rows), "rows": rows}, indent=2, default=str), encoding="utf-8",
            )
            _write_paper_table(out_root, method_rows, args.seed, args.budget)

        summary = {
            "method": method_name,
            "seed": args.seed,
            "budget_seconds": args.budget,
            "holdouts": names,
            "guide_enabled": enable_guide,
            "agg": _agg(rows),
            "rows": rows,
        }
        (method_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        with (method_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            fields = [
                "instance", "customer_class", "tw_family", "complete_feasibility",
                "served_customers", "total_customers", "m_star", "used_vehicles",
                "total_distance", "paper_j", "runtime", "within_budget_5pct", "termination",
            ]
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k) for k in fields})

    total = time.perf_counter() - t_all
    _write_paper_table(out_root, method_rows, args.seed, args.budget)
    manifest = {
        "baseline_id": f"paper_full9_holdout_seed{args.seed}",
        "immutable": False,
        "seed": args.seed,
        "budget_seconds": args.budget,
        "holdouts": names,
        "methods": [m for m, _ in methods],
        "total_runtime_s": total,
        "git": _git_hash(),
        "created_unix": time.time(),
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nDone in {total/3600:.2f} h. Artifacts: {out_root}", flush=True)
    print(f"Paper table: {out_root / 'method_comparison.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
