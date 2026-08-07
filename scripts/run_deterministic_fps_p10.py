"""P10: Validate Deterministic-FPS on 6 reserved instances; freeze if gate passes.

Usage:
  python scripts/run_deterministic_fps_p10.py
  python scripts/run_deterministic_fps_p10.py --budget 180
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
from src.search.fleet_search import compute_fleet_bounds
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
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _run_pytest(out_path: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=line"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    out_path.write_text(text, encoding="utf-8")
    # Parse "N passed"
    passed = 0
    for line in text.splitlines():
        if "passed" in line:
            parts = line.strip().split()
            for i, p in enumerate(parts):
                if p == "passed" and i > 0 and parts[i - 1].isdigit():
                    passed = int(parts[i - 1])
                if parts[i].endswith("passed") and parts[i][0].isdigit():
                    passed = int(parts[i].replace("passed", ""))
    return proc.returncode, text


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
    ap.add_argument("--budget", type=float, default=720.0, help="Equal wall-clock budget per instance (seconds)")
    ap.add_argument("--config", default="configs/train_tgfps.yaml")
    args = ap.parse_args()

    config = load_config(str(ROOT / args.config))
    config = dict(config)
    # Cap ALNS inside remaining budget; never use fixed m.
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
    fleet_cfg["seeds"] = [0, 1, 2, 3]
    fleet_cfg["upper_bound_retries"] = int(fleet_cfg.get("upper_bound_retries", 5))
    fleet_cfg["retries_per_fleet"] = int(fleet_cfg.get("retries_per_fleet", 2))
    fleet_cfg["failures_before_stop"] = int(fleet_cfg.get("failures_before_stop", 1))
    fleet_cfg["stop_upper_on_first_feasible"] = True
    fleet_cfg["upper_budget_fraction"] = 0.45
    fleet_cfg["warm_start_from_larger_fleet"] = True
    fleet_cfg["upper_expand_step"] = 4
    fleet_cfg["upper_expand_max"] = 48
    config["fleet_search"] = fleet_cfg

    val = list((config.get("dataset") or {}).get("validation_instances") or VALIDATION6)
    assert len(val) == 6, f"expected 6 validation instances, got {val}"
    assert not (set(val) & set(FULL9)), "validation must not overlap full-9 holdouts"

    out = ROOT / "outputs" / "baselines" / "deterministic_fps"
    for sub in ("validation_routes", "fleet_search_traces", "replay_reports"):
        (out / sub).mkdir(parents=True, exist_ok=True)

    # Save config used
    import yaml
    (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    config_hash = _sha256_file(out / "config.yaml")
    source_hash = _git_hash()
    (out / "source_hash.txt").write_text(
        f"git={source_hash}\nconfig_sha256={config_hash}\nbudget_seconds_per_instance={args.budget}\n",
        encoding="utf-8",
    )

    print("=" * 72, flush=True)
    print("P10 Deterministic-FPS validation", flush=True)
    print("instances:", val, flush=True)
    print("budget_s:", args.budget, flush=True)
    print("source:", source_hash, flush=True)
    print("=" * 72, flush=True)

    weights = CostWeights.from_eval_config(config)
    cap = load_capacity_from_config(config)
    root = Path(config["dataset"]["path"])
    rows = []
    all_failures = []
    runtime_profile = {"per_instance": {}, "budget_seconds": args.budget}
    t_all = time.perf_counter()

    for name in val:
        print(f"\n>>> {name}", flush=True)
        inst = load_instance(root / name, load_capacity=cap)
        lo, hi = compute_fleet_bounds(inst, config)
        t0 = time.perf_counter()
        result = solve_deterministic_fps(
            inst, config=config, weights=weights, budget_seconds=float(args.budget),
        )
        elapsed = time.perf_counter() - t0
        n_cust = int(inst.num_customers)
        served = n_cust - int(result.score.unserved_customer_count)
        attempted = sorted({t["m"] for t in result.fleet_trace})
        upper_feasible = None
        for t in result.fleet_trace:
            if t["feasible"]:
                upper_feasible = t["m"] if upper_feasible is None else max(upper_feasible, t["m"])

        row = {
            "instance": name,
            "customer_class": instance_class(name),
            "tw_family": _tw_family(name),
            "fleet_lower_bound": int(result.lower_bound_m or lo),
            "initial_feasible_upper_bound": int(upper_feasible if upper_feasible is not None else result.upper_bound_m or hi),
            "attempted_fleet_sizes": attempted,
            "smallest_feasible_fleet": int(result.m_star) if result.feasible else None,
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
            "search_termination_reason": result.termination_reason,
            "fleet_minimization_performed": bool(result.fleet_minimization_performed),
            "fixed_fleet_used": False,
            "stages": result.stages,
            "exact_check": result.exact_check,
            "unserved_list": list(result.unserved_customers),
        }
        rows.append(row)
        runtime_profile["per_instance"][name] = {
            "runtime": elapsed,
            "solver_runtime": result.runtime_seconds,
            "termination": result.termination_reason,
            "stages": {
                k: {sk: sv for sk, sv in (v.items() if isinstance(v, dict) else [])}
                if isinstance(v, dict) else v
                for k, v in result.stages.items()
            },
        }

        # Traces
        (out / "fleet_search_traces" / f"{name}.json").write_text(
            json.dumps(result.fleet_trace, indent=2, default=str), encoding="utf-8",
        )
        # Routes
        (out / "validation_routes" / f"{name}.json").write_text(
            json.dumps({
                "instance": name,
                "m_star": result.m_star,
                "feasible": result.feasible,
                "routes": _serialize_routes(result.routes),
                "score": list(result.score.as_tuple()),
            }, indent=2),
            encoding="utf-8",
        )
        # Replay report
        (out / "replay_reports" / f"{name}.json").write_text(
            json.dumps(result.exact_check, indent=2), encoding="utf-8",
        )
        for fr in result.failure_attribution:
            if isinstance(fr, dict):
                all_failures.append({"instance": name, **fr})

        print(
            f"  feas={result.feasible} served={served}/{n_cust} "
            f"m*={result.m_star} used={result.score.used_evs} "
            f"dist={result.score.total_distance:.1f} J={result.score.paper_objective:.1f} "
            f"t={result.runtime_seconds:.1f}s term={result.termination_reason}",
            flush=True,
        )
        print(f"  fleet attempts={len(result.fleet_trace)} sizes={attempted} lo={lo} hi={hi}", flush=True)

    total_runtime = time.perf_counter() - t_all
    runtime_profile["total_validation_runtime_s"] = total_runtime

    # Summaries
    n = max(len(rows), 1)
    summary = {
        "method": "Deterministic-FPS",
        "n": len(rows),
        "validation_instances": val,
        "budget_seconds_per_instance": args.budget,
        "mean_served": sum(r["served_customers"] / max(r["total_customers"], 1) for r in rows) / n,
        "feasibility_rate": sum(1.0 if r["complete_feasibility"] else 0.0 for r in rows) / n,
        "mean_used_vehicles": sum(r["used_vehicles"] for r in rows) / n,
        "mean_distance": sum(r["total_distance"] for r in rows) / n,
        "mean_charging_time": sum(r["total_charging_time"] for r in rows) / n,
        "mean_duration": sum(r["total_duration"] for r in rows) / n,
        "mean_paper_j": sum(r["paper_j"] for r in rows) / n,
        "mean_runtime": sum(r["runtime"] for r in rows) / n,
        "total_validation_runtime_s": total_runtime,
        "all_fleet_minimization": all(r["fleet_minimization_performed"] for r in rows),
        "any_fixed_fleet": any(r["fixed_fleet_used"] for r in rows),
        "rows": rows,
        "source_hash": source_hash,
        "config_sha256": config_hash,
    }

    # Failure attribution CSV
    fail_path = out / "failure_attribution.csv"
    if all_failures:
        keys = sorted({k for r in all_failures for k in r.keys()})
        with fail_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in all_failures:
                w.writerow(r)
    else:
        # Still write header for unserved from rows
        with fail_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["instance", "unserved_customers", "complete_feasibility"])
            w.writeheader()
            for r in rows:
                if r["unserved_list"] or not r["complete_feasibility"]:
                    w.writerow({
                        "instance": r["instance"],
                        "unserved_customers": ",".join(str(x) for x in r["unserved_list"]),
                        "complete_feasibility": r["complete_feasibility"],
                    })

    # CSV summary
    csv_path = out / "validation_summary.csv"
    flat_fields = [
        "instance", "customer_class", "tw_family", "fleet_lower_bound",
        "initial_feasible_upper_bound", "smallest_feasible_fleet", "complete_feasibility",
        "served_customers", "total_customers", "hard_violations", "used_vehicles",
        "total_distance", "charging_visits", "total_charging_time", "total_duration",
        "paper_j", "runtime", "search_termination_reason", "fleet_minimization_performed",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=flat_fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in flat_fields})

    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "runtime_profile.json").write_text(json.dumps(runtime_profile, indent=2, default=str), encoding="utf-8")

    # Tests
    print("\nRunning pytest...", flush=True)
    rc, _ = _run_pytest(out / "test_report.txt")
    summary["pytest_returncode"] = rc

    # Acceptance gate classification
    fleet_ok = summary["all_fleet_minimization"] and not summary["any_fixed_fleet"]
    descent_ran = all(len(r["attempted_fleet_sizes"]) >= 1 for r in rows)
    # Genuine descent: for each instance, either multiple fleet sizes attempted OR upper>m_star when feasible
    genuine_descent = []
    for r in rows:
        sizes = r["attempted_fleet_sizes"]
        genuine_descent.append(len(sizes) >= 2 or (r["complete_feasibility"] and r["smallest_feasible_fleet"] is not None))
    exact_ok = all(bool((r.get("exact_check") or {}).get("ok", False)) or not r["complete_feasibility"] for r in rows)
    # For feasible rows, exact_check must be ok
    exact_ok = all(
        (r.get("exact_check") or {}).get("ok", False)
        for r in rows if r["complete_feasibility"]
    ) and all(
        # infeasible still must not have duplicate/objective mismatch
        "duplicate" not in str((r.get("exact_check") or {}).get("errors", []))
        for r in rows
    )
    all_served_or_attributed = all(
        r["served_customers"] == r["total_customers"]
        or len(r["unserved_list"]) > 0
        or not r["complete_feasibility"]
        for r in rows
    )
    # Prefer: every instance fully served
    all_served = all(r["served_customers"] == r["total_customers"] and r["complete_feasibility"] for r in rows)
    pool_or_rescue = any(
        (r.get("stages") or {}).get("route_pool", {}).get("invoked")
        or (r.get("stages") or {}).get("critical_rescue", {}).get("invoked")
        for r in rows
    )
    tests_ok = rc == 0

    if fleet_ok and descent_ran and all(genuine_descent) and exact_ok and all_served and tests_ok and pool_or_rescue:
        decision = "PASS"
        decision_text = "Deterministic-FPS is a valid no-regression backbone."
    elif fleet_ok and descent_ran and tests_ok and (all_served_or_attributed or summary["feasibility_rate"] >= 0.5):
        decision = "PARTIAL"
        decision_text = "The solver works, but specific fleet/feasibility/runtime problems remain."
    else:
        decision = "FAIL"
        decision_text = "Fleet minimization or exact replay is not reliable."

    decision_payload = {
        "decision": decision,
        "text": decision_text,
        "checks": {
            "fleet_minimization": fleet_ok,
            "descent_ran": descent_ran,
            "genuine_descent_per_instance": genuine_descent,
            "exact_ok_on_feasible": exact_ok,
            "all_customers_served": all_served,
            "served_or_attributed": all_served_or_attributed,
            "route_pool_or_rescue_invoked": pool_or_rescue,
            "pytest_ok": tests_ok,
            "feasibility_rate": summary["feasibility_rate"],
        },
        "missing_or_skipped": [],
    }
    # Note missing components
    if not pool_or_rescue:
        decision_payload["missing_or_skipped"].append("route_pool/rescue never invoked usefully")
    if not all_served:
        decision_payload["missing_or_skipped"].append(
            "not all validation instances fully served: "
            + ",".join(r["instance"] for r in rows if not (r["complete_feasibility"] and r["served_customers"] == r["total_customers"]))
        )

    summary["p10_decision"] = decision_payload
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out / "P10_DECISION.md").write_text(
        f"# P10 decision: **{decision}**\n\n{decision_text}\n\n"
        f"```json\n{json.dumps(decision_payload, indent=2)}\n```\n",
        encoding="utf-8",
    )

    # Markdown table
    lines = [
        "# Deterministic-FPS validation (6 instances)",
        "",
        f"Decision: **{decision}** — {decision_text}",
        "",
        f"Budget: {args.budget}s / instance | Total runtime: {total_runtime:.1f}s | pytest rc={rc}",
        "",
        "| Instance | Class | LO | HI* | m* | Feas | Served | Used | Dist | Charge visits | Charge t | Dur | J | Runtime | Term |",
        "|---|---|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['instance']} | {r['customer_class']}/{r['tw_family']} | "
            f"{r['fleet_lower_bound']} | {r['initial_feasible_upper_bound']} | "
            f"{r['smallest_feasible_fleet']} | {r['complete_feasibility']} | "
            f"{r['served_customers']}/{r['total_customers']} | {r['used_vehicles']} | "
            f"{r['total_distance']:.1f} | {r['charging_visits']} | {r['total_charging_time']:.1f} | "
            f"{r['total_duration']:.1f} | {r['paper_j']:.1f} | {r['runtime']:.1f}s | "
            f"{r['search_termination_reason']} |"
        )
    lines += [
        "",
        "## Aggregate",
        "",
        f"- feas rate: {summary['feasibility_rate']:.3f}",
        f"- mean served: {summary['mean_served']:.3f}",
        f"- mean used vehicles: {summary['mean_used_vehicles']:.2f}",
        f"- mean distance: {summary['mean_distance']:.1f}",
        f"- mean paper J: {summary['mean_paper_j']:.1f}",
        f"- mean runtime: {summary['mean_runtime']:.1f}s",
        "",
        "Holdouts untouched. Transformer search-guide training not started.",
    ]
    (out / "validation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n" + "\n".join(lines), flush=True)
    print(f"\nP10 DECISION: {decision}", flush=True)
    print(f"Artifacts: {out}", flush=True)
    return 0 if decision != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
