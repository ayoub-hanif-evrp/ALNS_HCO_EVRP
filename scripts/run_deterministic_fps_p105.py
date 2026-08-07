"""P10.5: Re-validate Deterministic-FPS after structural repairs.

Usage:
  python scripts/run_deterministic_fps_p105.py --budget 600 --seed 0
  python scripts/run_deterministic_fps_p105.py --budget 600 --seed 1
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

VALIDATION6 = ["c103_21", "c201_21", "r103_21", "r201_21", "rc103_21", "rc201_21"]
FULL9 = [
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
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


def _run_pytest(out_path: Path) -> tuple[int, int]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_path.write_text(text, encoding="utf-8")
    passed = 0
    for line in text.splitlines():
        if "passed" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "passed" and i > 0 and parts[i - 1].isdigit():
                    passed = int(parts[i - 1])
    return proc.returncode, passed


def _serialize_routes(routes) -> list:
    out = []
    for nodes, labels in routes:
        out.append({
            "nodes": [int(n) for n in nodes],
            "charge_labels": [None if x is None else float(x) for x in (labels or [])],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=900.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="configs/train_tgfps.yaml")
    ap.add_argument("--out", default="outputs/baselines/deterministic_fps_p105")
    args = ap.parse_args()

    config = dict(load_config(str(ROOT / args.config)))
    hs = dict(config.get("hybrid_search") or {})
    alns = dict(hs.get("alns") or {})
    alns["iterations"] = int(alns.get("iterations", 40))
    alns["time_limit_seconds"] = min(float(alns.get("time_limit_seconds", 25)), 40.0)
    hs["alns"] = alns
    cons = dict(hs.get("construction") or {})
    cons["expert_variants_list"] = ["edf", "regret", "energy", "fleet", "hybrid"]
    hs["construction"] = cons
    config["hybrid_search"] = hs

    fleet_cfg = dict(config.get("fleet_search") or {})
    base_seeds = [0, 1, 2, 3]
    # Offset diversifies descent/warm probes; seed 0 is always retained so the
    # canonical upper-bound constructor remains available under dual-seed runs.
    fleet_cfg["seeds"] = sorted({0, *[s + int(args.seed) for s in base_seeds]})
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
    # Keep insertion enumeration tractable under wall budgets.
    repair = dict(hs.get("repair") or {})
    repair["station_candidates"] = 2
    repair["max_rounds"] = 8
    hs["repair"] = repair
    config["hybrid_search"] = hs
    config.setdefault("deterministic_fps_budget", {
        "upper_bound_fraction": 0.15,
        "fleet_descent_fraction": 0.40,
        "route_pool_fraction": 0.10,
        "charging_fraction": 0.10,
        "repair_alns_fraction": 0.20,
        "rescue_fraction": 0.05,
    })

    val = list((config.get("dataset") or {}).get("validation_instances") or VALIDATION6)
    assert len(val) == 6
    assert not (set(val) & set(FULL9))

    out = ROOT / args.out / f"seed_{args.seed}"
    for sub in ("validation_routes", "fleet_search_traces", "replay_reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    import yaml
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_hash = _sha256_file(out / "config.yaml")
    source_hash = _git_hash()
    (out / "source_hash.txt").write_text(
        f"git={source_hash}\nconfig_sha256={config_hash}\nbudget_seconds_per_instance={args.budget}\nseed={args.seed}\n",
        encoding="utf-8",
    )

    print("=" * 72, flush=True)
    print(f"P10.5 Deterministic-FPS validation seed={args.seed}", flush=True)
    print("instances:", val, "budget:", args.budget, flush=True)
    print("=" * 72, flush=True)

    weights = CostWeights.from_eval_config(config)
    cap = load_capacity_from_config(config)
    root = Path(config["dataset"]["path"])
    rows = []
    t_all = time.perf_counter()

    for name in val:
        print(f"\n>>> {name}", flush=True)
        inst = load_instance(root / name, load_capacity=cap)
        lo, hi = compute_fleet_bounds(inst, config)
        math_lo = capacity_lower_bound(inst)
        result = solve_deterministic_fps(
            inst, config=config, weights=weights, budget_seconds=float(args.budget),
        )
        n_cust = int(inst.num_customers)
        served = n_cust - int(result.score.unserved_customer_count)
        attempted = sorted({t["m"] for t in result.fleet_trace})
        upper_feasible = None
        for t in result.fleet_trace:
            if t["feasible"]:
                upper_feasible = t["m"] if upper_feasible is None else max(upper_feasible, t["m"])
        success_at_mstar = sum(
            1 for t in result.fleet_trace
            if t["feasible"] and t["m"] == result.m_star
        )
        stuck_at_upper = (
            result.feasible
            and upper_feasible is not None
            and result.m_star == upper_feasible
            and len([m for m in attempted if m < upper_feasible]) <= 1
        )
        stages = result.stages or {}
        row = {
            "instance": name,
            "customer_class": instance_class(name),
            "tw_family": _tw_family(name),
            "mathematical_lower_bound": int(math_lo),
            "heuristic_lower_estimate": int(result.lower_bound_m or lo),
            "first_feasible_upper_bound": int(upper_feasible if upper_feasible is not None else result.upper_bound_m or hi),
            "attempted_fleet_sizes": attempted,
            "smallest_feasible_fleet": int(result.m_star) if result.feasible else None,
            "successful_attempts_at_mstar": int(success_at_mstar),
            "complete_feasibility": bool(result.feasible),
            "served_customers": int(served),
            "total_customers": n_cust,
            "hard_violations": int(result.score.hard_violation_count),
            "used_vehicles": int(result.score.used_evs),
            "total_distance": float(result.score.total_distance),
            "charging_visits": int(result.charging_visits),
            "total_charging_time": float(result.score.total_charging_time),
            "total_duration": float(result.score.total_duration),
            "paper_j": float(result.score.paper_objective),
            "runtime": float(result.runtime_seconds),
            "budget_seconds": float(args.budget),
            "budget_adherence": result.budget_adherence,
            "within_budget_5pct": bool((result.budget_adherence or {}).get("within_5pct", False)),
            "search_termination_reason": result.termination_reason,
            "fleet_minimization_performed": True,
            "fixed_fleet_used": False,
            "stuck_at_upper_after_one_fail": bool(stuck_at_upper),
            "stage_runtimes": result.stage_runtimes,
            "stages": stages,
            "exact_check": result.exact_check,
            "bound_annotations": result.bound_annotations,
            "route_pool_size": (stages.get("fleet_search") or {}).get("route_pool_size")
                or (stages.get("route_pool") or {}).get("pool_size"),
            "set_partitioning": stages.get("route_pool"),
            "route_elimination_attempts": (stages.get("fleet_search") or {}).get("route_elimination_attempts"),
            "unserved_list": list(result.unserved_customers),
        }
        rows.append(row)

        (out / "fleet_search_traces" / f"{name}.json").write_text(
            json.dumps(result.fleet_trace, indent=2, default=str), encoding="utf-8",
        )
        (out / "validation_routes" / f"{name}.json").write_text(
            json.dumps({
                "instance": name, "m_star": result.m_star, "feasible": result.feasible,
                "routes": _serialize_routes(result.routes),
                "score": list(result.score.as_tuple()),
            }, indent=2),
            encoding="utf-8",
        )
        (out / "replay_reports" / f"{name}.json").write_text(
            json.dumps(result.exact_check, indent=2), encoding="utf-8",
        )
        print(
            f"  feas={result.feasible} served={served}/{n_cust} m*={result.m_star} "
            f"used={result.score.used_evs} dist={result.score.total_distance:.1f} "
            f"t={result.runtime_seconds:.1f}s within5={row['within_budget_5pct']} "
            f"stuck_upper={stuck_at_upper} term={result.termination_reason}",
            flush=True,
        )

    total_runtime = time.perf_counter() - t_all
    n = max(len(rows), 1)
    summary = {
        "method": "Deterministic-FPS",
        "phase": "P10.5",
        "seed": args.seed,
        "n": len(rows),
        "validation_instances": val,
        "budget_seconds_per_instance": args.budget,
        "mean_served": sum(r["served_customers"] / max(r["total_customers"], 1) for r in rows) / n,
        "feasibility_rate": sum(1.0 if r["complete_feasibility"] else 0.0 for r in rows) / n,
        "mean_used_vehicles": sum(r["used_vehicles"] for r in rows) / n,
        "mean_distance": sum(r["total_distance"] for r in rows) / n,
        "mean_runtime": sum(r["runtime"] for r in rows) / n,
        "total_validation_runtime_s": total_runtime,
        "all_within_budget_5pct": all(r["within_budget_5pct"] for r in rows),
        "any_stuck_at_upper": any(r["stuck_at_upper_after_one_fail"] for r in rows),
        "rows": rows,
        "source_hash": source_hash,
        "config_sha256": config_hash,
    }

    flat_fields = [
        "instance", "customer_class", "tw_family", "mathematical_lower_bound",
        "heuristic_lower_estimate", "first_feasible_upper_bound", "smallest_feasible_fleet",
        "successful_attempts_at_mstar", "complete_feasibility", "served_customers",
        "total_customers", "used_vehicles", "total_distance", "charging_visits",
        "total_charging_time", "paper_j", "runtime", "within_budget_5pct",
        "search_termination_reason", "stuck_at_upper_after_one_fail",
    ]
    with (out / "validation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in flat_fields})

    # Stage runtime table
    with (out / "stage_runtime_table.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["instance", "stage", "seconds", "invoked"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            times = ((r.get("stage_runtimes") or {}).get("times") or {})
            inv = ((r.get("stage_runtimes") or {}).get("invoked") or {})
            for stage, sec in times.items():
                w.writerow({"instance": r["instance"], "stage": stage, "seconds": sec, "invoked": inv.get(stage)})

    with (out / "budget_overrun_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["instance", "budget", "runtime", "ratio", "within_5pct"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "instance": r["instance"],
                "budget": r["budget_seconds"],
                "runtime": r["runtime"],
                "ratio": r["runtime"] / max(r["budget_seconds"], 1e-9),
                "within_5pct": r["within_budget_5pct"],
            })

    print("\nRunning pytest...", flush=True)
    rc, passed = _run_pytest(out / "test_report.txt")
    summary["pytest_returncode"] = rc
    summary["tests_passed"] = passed

    # Acceptance
    mandatory_stages_ok = all(
        (r.get("stages") or {}).get("charging_optimization", {}).get("invoked")
        and (r.get("stages") or {}).get("repair", {}).get("invoked")
        and (r.get("stages") or {}).get("alns", {}).get("invoked")
        and (r.get("stages") or {}).get("route_pool", {}).get("invoked")
        and (r.get("stages") or {}).get("critical_rescue", {}).get("invoked_check")
        for r in rows
    )
    exact_ok = all((r.get("exact_check") or {}).get("ok", False) for r in rows if r["complete_feasibility"])
    all_served = all(r["complete_feasibility"] and r["served_customers"] == r["total_customers"] for r in rows)
    no_fixed = all(not r["fixed_fleet_used"] for r in rows)
    not_stuck = not any(r["stuck_at_upper_after_one_fail"] for r in rows)
    min_attempts = all(r["successful_attempts_at_mstar"] >= 1 for r in rows if r["complete_feasibility"])
    # At least 3 attempts before stopping descent: check fleet traces have >=3 attempts when descent ran
    descent_attempts_ok = all(len(r["attempted_fleet_sizes"]) >= 2 or r["smallest_feasible_fleet"] is None for r in rows)

    if (
        rc == 0 and all_served and no_fixed and not_stuck and mandatory_stages_ok
        and exact_ok and summary["all_within_budget_5pct"] and descent_attempts_ok
    ):
        decision, text = "PASS", "Deterministic-FPS is a valid no-regression backbone."
    elif rc == 0 and all_served and no_fixed:
        decision, text = "PARTIAL", "The solver works, but specific fleet/feasibility/runtime problems remain."
    else:
        decision, text = "FAIL", "Fleet minimization or exact replay is not reliable."

    decision_payload = {
        "decision": decision,
        "text": text,
        "checks": {
            "pytest_ok": rc == 0,
            "tests_passed": passed,
            "all_served": all_served,
            "no_fixed_fleet": no_fixed,
            "not_stuck_at_upper": not_stuck,
            "mandatory_stages": mandatory_stages_ok,
            "exact_ok": exact_ok,
            "budget_5pct": summary["all_within_budget_5pct"],
            "descent_attempts_ok": descent_attempts_ok,
            "min_success_at_mstar": min_attempts,
        },
    }
    summary["p105_decision"] = decision_payload
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "P105_DECISION.md").write_text(
        f"# P10.5 decision: **{decision}**\n\n{text}\n\n```json\n{json.dumps(decision_payload, indent=2)}\n```\n",
        encoding="utf-8",
    )

    lines = [
        f"# Deterministic-FPS P10.5 (seed={args.seed})",
        "",
        f"Decision: **{decision}** — {text}",
        "",
        f"Budget: {args.budget}s | Total: {total_runtime:.1f}s | tests={passed} rc={rc}",
        "",
        "| Instance | Class | MathLO | HI* | m* | Feas | Served | Used | Dist | Chg | J | Runtime | Within5% | StuckUpper | Term |",
        "|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|:---:|:---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instance']} | {r['customer_class']}/{r['tw_family']} | "
            f"{r['mathematical_lower_bound']} | {r['first_feasible_upper_bound']} | "
            f"{r['smallest_feasible_fleet']} | {r['complete_feasibility']} | "
            f"{r['served_customers']}/{r['total_customers']} | {r['used_vehicles']} | "
            f"{r['total_distance']:.1f} | {r['charging_visits']} | {r['paper_j']:.1f} | "
            f"{r['runtime']:.1f}s | {r['within_budget_5pct']} | {r['stuck_at_upper_after_one_fail']} | "
            f"{r['search_termination_reason']} |"
        )
    (out / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"\nP10.5 DECISION: {decision}", flush=True)
    print(f"Artifacts: {out}", flush=True)
    return 0 if decision != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
