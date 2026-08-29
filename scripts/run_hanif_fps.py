"""Hanif Fleet Intensification runner (A + B with A->B warm starts).

Frozen paper results live in outputs/{classical,proposed,constructor,alns,vns_ts}.
Pass --out to a new directory. Do not overwrite those folders.

Usage:
  python scripts/run_hanif_fps.py --all --variant both --budget 180 --out outputs/hanif_fps_run_YYYYmmdd
  python scripts/run_hanif_fps.py --pilot --variant both --budget 180 --out outputs/hanif_fps_pilot
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hanif_convert import VARIANT_SPECS, ensure_hanif_variant_converted
from src.data.hanif_dataset import discover_hanif_instances
from src.data.instance_class import instance_class
from src.data.instance_loader import apply_battery_capacity, load_instance
from src.env.cost import CostWeights
from src.evaluation.hanif_protocols import (
    PROTOCOL_NO_WAIT_REPLAY,
    completeness_report,
    evaluation_protocol,
)
from src.evaluation.objective_reference import attach_reference
from src.evaluation.output_paths import FROZEN_RESULT_DIRS, OUTPUTS
from src.evaluation.paper_objective import objective_spec
from src.search.deterministic_fps import solve_deterministic_fps
from src.search.fleet_search import capacity_lower_bound, compute_fleet_bounds
from src.utils.config import load_config

PILOT = ["C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1"]
REPLAY_PROTOCOL = dict(PROTOCOL_NO_WAIT_REPLAY)


def _eval_proto(cfg: dict | None = None) -> dict:
    return evaluation_protocol(cfg)


def _normalize_variant(v: str) -> str:
    key = str(v).strip().upper()
    if key in ("ORIGINAL", "ORIGINAL_TW", "VARIANT_A", "A"):
        return "A"
    if key in ("RELAXED", "RELAXED_TW", "VARIANT_B", "B"):
        return "B"
    if key == "BOTH":
        return "BOTH"
    raise SystemExit(f"Unknown --variant {v!r}; use A, B, or both")


def _method_name(cfg: dict) -> str:
    exp = cfg.get("experiment") or {}
    competitor = str(exp.get("competitor") or "").strip().lower()
    if competitor == "alns":
        return "ALNS"
    if competitor == "vns_ts":
        return "VNS-TS"
    if exp.get("method_name"):
        return str(exp["method_name"])
    if not bool((cfg.get("hanif_fleet_intensify") or {}).get("enabled", True)):
        return "Deterministic-FPS-NoIntensification"
    return "HanifFleetIntensification"


def _citeable_mean(rows: list[dict], key: str, *, complete: bool):
    if not complete or not rows:
        return None
    return sum(float(r[key]) for r in rows) / len(rows)


def _partial_mean(rows: list[dict], key: str):
    if not rows:
        return None
    return sum(float(r[key] or 0.0) for r in rows) / len(rows)


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
    method_name = _method_name(cfg)
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
        "phi": float(result.score.phi),
        "paper_j": float(result.score.paper_objective),
        "charging_visits": int(result.charging_visits or exact.get("charging_visits") or 0),
        "waiting_time": float(exact.get("waiting_time") or 0.0),
        "station_waiting_time": float(exact.get("station_waiting_time") or 0.0),
        "customer_early_waiting_time": float(exact.get("customer_early_waiting_time") or 0.0),
        "charging_time": float(exact.get("charging_time") or result.score.total_charging_time),
        "duration": float(exact.get("duration") or result.score.total_duration),
        "expected_charging_cost": float(exact.get("expected_charging_cost") or 0.0),
        "risk_adjusted_charging_cost": float((exact.get("station_metrics") or {}).get("risk_adjusted_charging_cost") or 0.0),
        "cvar95_charging_cost_exposure": float((exact.get("station_metrics") or {}).get("CVaR95_charging_cost_exposure") or 0.0),
        "station_waiting_time_expected": float(exact.get("station_waiting_time_expected") or exact.get("station_waiting_time") or 0.0),
        "risk_adjusted_station_waiting": float((exact.get("station_metrics") or {}).get("risk_adjusted_station_waiting") or 0.0),
        "cvar95_wait_exposure": float((exact.get("station_metrics") or {}).get("CVaR95_wait_exposure") or 0.0),
        "charged_energy": float(exact.get("total_energy_charged") or 0.0),
        "energy_consumed": float(exact.get("total_energy_consumed") or exact.get("energy_consumed") or 0.0),
        "travel_time": float(exact.get("travel_time") or 0.0),
        "service_time": float(exact.get("service_time") or 0.0),
        "mean_route_minimum_soc": float((exact.get("battery_metrics") or {}).get("mean_route_minimum_soc") or exact.get("mean_route_minimum_soc") or 0.0),
        "median_route_minimum_soc": float((exact.get("battery_metrics") or {}).get("median_route_minimum_soc") or 0.0),
        "routes_requiring_charging": int((exact.get("battery_metrics") or {}).get("routes_requiring_charging") or 0),
        "hard_violations": int(exact.get("hard_violations") or result.score.hard_violation_count),
        "served_ratio": served / max(int(inst.num_customers), 1),
        "minimum_soc": float(exact.get("minimum_soc_fraction") or 0.0),
        "unique_stations_used": int((exact.get("station_metrics") or {}).get("unique_stations_used") or 0),
        "mean_charging_rate_visited": float((exact.get("station_metrics") or {}).get("mean_charging_rate_visited") or 0.0),
        "mean_expected_price_visited": float((exact.get("station_metrics") or {}).get("mean_expected_price_visited") or 0.0),
        "mean_expected_wait_visited": float((exact.get("station_metrics") or {}).get("mean_expected_wait_visited") or 0.0),
        "d_ref": float((cfg.get("objective_reference") or {}).get("D_ref") or 0.0),
        "tstation_ref": float((cfg.get("objective_reference") or {}).get("Tstation_ref") or 0.0),
        "ccharge_ref": float((cfg.get("objective_reference") or {}).get("Ccharge_ref") or 0.0),
        "intensify_initial_k": intens.get("initial_k"),
        "intensify_final_k": intens.get("final_k"),
        "intensify_eliminations": intens.get("eliminations"),
        "warm_start": bool(((result.stages or {}).get("warm_start") or {}).get("accepted")),
        "runtime_s": time.perf_counter() - t0,
        "method": method_name,
        "evaluation_protocol": _eval_proto(cfg)["name"],
        "intensification_enabled": bool(
            (cfg.get("hanif_fleet_intensify") or {}).get("enabled", True)
        ),
        "ab_warm_start_enabled": bool(
            (cfg.get("experiment") or {}).get("ab_warm_start_enabled", True)
        ),
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
                "waiting_time": row["waiting_time"],
            },
            "method": method_name,
            "evaluation_protocol": _eval_proto(cfg),
            "replay_protocol": REPLAY_PROTOCOL,
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
    root = ensure_hanif_variant_converted(variant, output_dir=spec["output"], source_dir=spec["source"])
    out_dir.mkdir(parents=True, exist_ok=True)
    routes_dir = out_dir / "routes"
    weights = CostWeights.from_eval_config(cfg)

    intensify_on = bool((cfg.get("hanif_fleet_intensify") or {}).get("enabled", True))
    print(
        f"\n=== Hanif {spec['label']} ({spec['key']})  "
        f"method={_method_name(cfg)}  intensify={intensify_on}  "
        f"n={len(names)}  budget={budget}s ===",
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

    complete = completeness_report(
        expected_names=names,
        got_names=[r["instance"] for r in rows],
    )
    if rows:
        with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
    summary = {
        "variant": spec["key"],
        "tw_variant": variant,
        "label": spec["label"],
        "method": _method_name(cfg),
        "evaluation_protocol": _eval_proto(cfg),
        "replay_protocol": REPLAY_PROTOCOL,
        "experiment": dict(cfg.get("experiment") or {}),
        "n": len(rows),
        "expected_n": len(names),
        "complete": complete["complete"],
        "completeness": complete,
        "budget_s": budget,
        "wall_s": time.perf_counter() - t_all,
        "feasibility_rate": (
            sum(1 for r in rows if r["feasible"]) / len(rows) if rows and complete["complete"] else None
        ),
        "mean_served_ratio": _citeable_mean(
            [{"v": r["served"] / max(r["n_customers"], 1)} for r in rows],
            "v",
            complete=complete["complete"],
        ) if rows else None,
        "mean_k": _citeable_mean(rows, "used_k", complete=complete["complete"]),
        "mean_runtime_s": _citeable_mean(rows, "runtime_s", complete=complete["complete"]),
        "mean_eliminations": _citeable_mean(
            [{"v": int(r["intensify_eliminations"] or 0)} for r in rows],
            "v",
            complete=complete["complete"],
        ) if rows else None,
        "partial_statistics_do_not_cite": None if complete["complete"] else {
            "n": len(rows),
            "mean_k": _partial_mean(rows, "used_k"),
            "mean_runtime_s": _partial_mean(rows, "runtime_s"),
            "n_feasible": sum(1 for r in rows if r["feasible"]),
        },
        "attributes_used": dict(_eval_proto(cfg)),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2), flush=True)
    return summary, saved_routes


def _parallel_pair_worker(payload: tuple) -> dict[str, dict]:
    """Solve one instance (A then B) in an isolated process."""
    name, cfg_base, budget, out_root, variant = payload
    out_root = Path(out_root)
    rows: dict[str, dict] = {}
    warm_routes = None

    if variant in ("A", "BOTH"):
        cfg_a = copy.deepcopy(cfg_base)
        cfg_a.setdefault("dataset", {}).update({
            "name": "hanif",
            "tw_variant": "A",
            "path": str(VARIANT_SPECS["A"]["output"]),
            "source_dir": str(VARIANT_SPECS["A"]["source"]),
            "use_only_this_dataset": True,
        })
        route_dir_a = out_root / "original_tw" / "routes"
        row_a = _solve_one(
            name=name,
            root=Path(VARIANT_SPECS["A"]["output"]),
            cfg=cfg_a,
            weights=CostWeights.from_eval_config(cfg_a),
            budget=float(budget),
            warm_routes=None,
            routes_dir=route_dir_a,
        )
        row_a["variant"] = VARIANT_SPECS["A"]["key"]
        row_a["tw_variant"] = "A"
        rows["A"] = row_a
        payload_a = json.loads((route_dir_a / f"{name}.json").read_text(encoding="utf-8"))
        warm_routes = _deserialize_routes(payload_a["routes"])

    if variant in ("B", "BOTH"):
        cfg_b = copy.deepcopy(cfg_base)
        cfg_b.setdefault("dataset", {}).update({
            "name": "hanif",
            "tw_variant": "B",
            "path": str(VARIANT_SPECS["B"]["output"]),
            "source_dir": str(VARIANT_SPECS["B"]["source"]),
            "use_only_this_dataset": True,
        })
        row_b = _solve_one(
            name=name,
            root=Path(VARIANT_SPECS["B"]["output"]),
            cfg=cfg_b,
            weights=CostWeights.from_eval_config(cfg_b),
            budget=float(budget),
            warm_routes=(
                warm_routes
                if bool((cfg_b.get("experiment") or {}).get("ab_warm_start_enabled", True))
                else None
            ),
            routes_dir=out_root / "relaxed_tw" / "routes",
        )
        row_b["variant"] = VARIANT_SPECS["B"]["key"]
        row_b["tw_variant"] = "B"
        rows["B"] = row_b
    return rows


def _write_parallel_variant(
    *,
    variant: str,
    rows: list[dict],
    budget: float,
    wall_s: float,
    out_dir: Path,
    expected_names: list[str],
    experiment: dict | None = None,
    failures: list[dict] | None = None,
) -> dict:
    """Write deterministic merged CSV/summary for process-level evaluation."""
    spec = VARIANT_SPECS[variant]
    rows = sorted(rows, key=lambda r: r["instance"])
    complete = completeness_report(
        expected_names=expected_names,
        got_names=[r["instance"] for r in rows],
        failures=failures,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_dir / "results.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    method = (
        str(rows[0].get("method")) if rows else str((experiment or {}).get("method_name") or _method_name({}))
    )
    exp = dict(experiment or {})
    if rows:
        exp.setdefault("intensification_enabled", bool(rows[0].get("intensification_enabled")))
        exp.setdefault("ab_warm_start_enabled", bool(rows[0].get("ab_warm_start_enabled")))
    summary = {
        "variant": spec["key"],
        "tw_variant": variant,
        "label": spec["label"],
        "method": method,
        "evaluation_protocol": evaluation_protocol({"paper_objective": exp.get("paper_objective") or {}}),
        "replay_protocol": REPLAY_PROTOCOL,
        "experiment": exp,
        "n": len(rows),
        "expected_n": len(expected_names),
        "complete": complete["complete"],
        "completeness": complete,
        "budget_s": budget,
        "wall_s": wall_s,
        "parallel_merge": True,
        "feasibility_rate": (
            sum(1 for r in rows if r["feasible"]) / len(rows) if rows and complete["complete"] else None
        ),
        "mean_served_ratio": (
            _citeable_mean(
                [{"v": r["served"] / max(r["n_customers"], 1)} for r in rows],
                "v",
                complete=complete["complete"],
            )
            if rows else None
        ),
        "mean_k": _citeable_mean(rows, "used_k", complete=complete["complete"]),
        "mean_runtime_s": _citeable_mean(rows, "runtime_s", complete=complete["complete"]),
        "mean_eliminations": (
            _citeable_mean(
                [{"v": int(r["intensify_eliminations"] or 0)} for r in rows],
                "v",
                complete=complete["complete"],
            )
            if rows else None
        ),
        "partial_statistics_do_not_cite": None if complete["complete"] else {
            "n": len(rows),
            "mean_k": _partial_mean(rows, "used_k"),
            "mean_runtime_s": _partial_mean(rows, "runtime_s"),
            "n_feasible": sum(1 for r in rows if r["feasible"]),
        },
        "attributes_used": dict(evaluation_protocol({"paper_objective": exp.get("paper_objective") or {}})),
        "rows": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _run_parallel_pairs(
    *,
    names: list[str],
    cfg: dict,
    budget: float,
    out_root: Path,
    variant: str,
    workers: int,
) -> list[dict]:
    """Parallelize independent instances while preserving each A->B dependency."""
    ensure_hanif_variant_converted("A")
    if variant in ("B", "BOTH"):
        ensure_hanif_variant_converted("B")
    t0 = time.perf_counter()
    gathered: dict[str, list[dict]] = {"A": [], "B": []}
    failures: list[dict] = []
    tasks = [
        (name, cfg, float(budget), str(out_root), variant)
        for name in names
    ]
    with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_parallel_pair_worker, task): task[0] for task in tasks}
        for done, future in enumerate(as_completed(futures), 1):
            name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                failures.append({"instance": name, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[parallel {done}/{len(tasks)}] {name} FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            for key, row in result.items():
                gathered[key].append(row)
            values = " ".join(
                f"{key}:K={row['used_k']}" for key, row in sorted(result.items())
            )
            print(f"[parallel {done}/{len(tasks)}] {name} {values}", flush=True)
    wall_s = time.perf_counter() - t0
    experiment = dict(cfg.get("experiment") or {})
    summaries = []
    want_a = variant in ("A", "BOTH")
    want_b = variant in ("B", "BOTH")
    if want_a:
        summaries.append(_write_parallel_variant(
            variant="A", rows=gathered["A"], budget=budget, wall_s=wall_s,
            out_dir=out_root / "original_tw", expected_names=names,
            experiment=experiment, failures=failures,
        ))
    if want_b:
        summaries.append(_write_parallel_variant(
            variant="B", rows=gathered["B"], budget=budget, wall_s=wall_s,
            out_dir=out_root / "relaxed_tw", expected_names=names,
            experiment=experiment, failures=failures,
        ))
    completeness = {
        "expected_n": len(names),
        "workers": int(workers),
        "wall_s": wall_s,
        "failed_instances": failures,
        "variants": {},
    }
    if want_a:
        completeness["variants"]["A"] = completeness_report(
            expected_names=names,
            got_names=[r["instance"] for r in gathered["A"]],
            failures=failures,
        )
    if want_b:
        completeness["variants"]["B"] = completeness_report(
            expected_names=names,
            got_names=[r["instance"] for r in gathered["B"]],
            failures=failures,
        )
    completeness["complete"] = all(v["complete"] for v in completeness["variants"].values())
    completeness["cite_means"] = completeness["complete"]
    (out_root / "completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n", encoding="utf-8",
    )
    if not completeness["complete"]:
        print(
            "INCOMPLETE RUN: refusing to treat partial means as the official result. "
            f"report={json.dumps(completeness)}",
            flush=True,
        )
    return summaries


def _write_combined_summary(summaries: list[dict], out_root: Path, budget: float) -> None:
    if len(summaries) < 2:
        return
    rows_a = {r["instance"]: r for r in summaries[0]["rows"]}
    rows_b = {r["instance"]: r for r in summaries[1]["rows"]}
    regressions = []
    for name, ra in rows_a.items():
        rb = rows_b.get(name)
        if rb and ra["feasible"] and rb["feasible"] and int(rb["used_k"]) > int(ra["used_k"]):
            regressions.append({"instance": name, "k_a": ra["used_k"], "k_b": rb["used_k"]})
    complete = all(bool(s.get("complete")) for s in summaries)
    combined = {
        "n_variants": 2,
        "budget_s": budget,
        "method": str(summaries[0].get("method") or "Deterministic-FPS"),
        "evaluation_protocol": summaries[0].get("evaluation_protocol") or _eval_proto(
            {"paper_objective": (summaries[0].get("experiment") or {}).get("paper_objective")}
        ),
        "replay_protocol": REPLAY_PROTOCOL,
        "experiment": dict(summaries[0].get("experiment") or {}),
        "complete": complete,
        "completeness": {
            s["tw_variant"]: s.get("completeness") for s in summaries
        },
        "milestone_targets": {"A_mean_k_max": 79, "B_mean_k_max": 64, "B_le_A": True},
        "b_worse_than_a": regressions if complete else [],
        "b_worse_than_a_count": len(regressions) if complete else None,
        "variants": {
            s["tw_variant"]: {k: s[k] for k in s if k != "rows"} for s in summaries
        },
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "summary_both.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    all_rows = []
    for summary in summaries:
        all_rows.extend(summary["rows"])
    with (out_root / "results_both.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(json.dumps(combined, indent=2), flush=True)
    if not complete:
        print("INCOMPLETE: combined means are not citeable. See completeness.json.", flush=True)
        return
    print(
        f"Milestone check: A meanK={summaries[0]['mean_k']:.2f} (<=79?)  "
        f"B meanK={summaries[1]['mean_k']:.2f} (<=64?)  "
        f"B>A count={len(regressions)}",
        flush=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hanif_fps.yaml")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--variant", default="both", help="A | B | both")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--workers", type=int, default=1,
        help="Process workers; each task keeps its A->B warm-start pair together",
    )
    ap.add_argument(
        "--disable-intensify", action="store_true",
        help="Matched ablation: skip fleet intensification",
    )
    ap.add_argument(
        "--disable-ab-warm-start", action="store_true",
        help="Do not seed variant B from the variant A incumbent",
    )
    ap.add_argument(
        "--competitor",
        default="none",
        choices=["none", "alns", "vns_ts"],
        help="Literature metaheuristic: ALNS (Keskin/Catay) or VNS/TS (Schneider)",
    )
    ap.add_argument(
        "--battery-q",
        type=float,
        default=None,
        help="Override battery capacity Q (default: instance/ev_specs value, typically 600)",
    )
    ap.add_argument(
        "--objective",
        default=None,
        choices=["classical", "expected", "risk_aware"],
        help="Paper objective mode (default: config paper_objective.mode)",
    )
    ap.add_argument("--risk-blend", type=float, default=None)
    ap.add_argument(
        "--reference-dir",
        default=None,
        help="Shared ObjectiveReference directory (used by objective_v2 A/B/C)",
    )
    args = ap.parse_args()

    out_root = ROOT / args.out
    resolved = out_root.resolve()
    if resolved == OUTPUTS.resolve():
        raise SystemExit("Refusing to overwrite the frozen outputs/ archive")
    for p in FROZEN_RESULT_DIRS:
        fr = p.resolve()
        if resolved == fr or fr in resolved.parents:
            raise SystemExit(f"Refusing to overwrite frozen results: {p.name}")

    cfg = dict(load_config(str(ROOT / args.config)))
    cfg.setdefault("hybrid_search", {})["neural_guidance"] = {"enabled": False}
    fs = dict(cfg.get("fleet_search") or {})
    fs["seeds"] = [0]
    fs["fast_mode"] = True
    fs["construction_backend"] = "dense"
    fs["prefer_expert_construct"] = False
    cfg["fleet_search"] = fs
    competitor = str(args.competitor or "none").strip().lower()
    if competitor in {"alns", "vns_ts"}:
        args.disable_intensify = True
    cfg.setdefault("hanif_fleet_intensify", {})["enabled"] = not args.disable_intensify
    if args.battery_q is not None:
        cfg.setdefault("problem", {})["battery_capacity"] = float(args.battery_q)
    po = dict(cfg.get("paper_objective") or {})
    if args.objective == "classical":
        po["mode"] = "classical_distance"
    elif args.objective == "expected":
        po["mode"] = "expected_heterogeneous"
        po["risk_blend"] = 0.0
    elif args.objective == "risk_aware":
        po["mode"] = "risk_aware_heterogeneous"
        po["risk_blend"] = 0.25 if args.risk_blend is None else float(args.risk_blend)
    if args.risk_blend is not None and args.objective != "expected":
        po["risk_blend"] = float(args.risk_blend)
    cfg["paper_objective"] = po
    spec = objective_spec(cfg)
    method_name = {
        "alns": "ALNS",
        "vns_ts": "VNS-TS",
    }.get(competitor) or (
        "Deterministic-FPS-NoIntensification"
        if args.disable_intensify else "HanifFleetIntensification"
    )
    cfg["experiment"] = {
        "method_name": method_name,
        "competitor": competitor if competitor != "none" else "",
        "intensification_enabled": not args.disable_intensify,
        "ab_warm_start_enabled": not args.disable_ab_warm_start,
        "budget_s": float(args.budget),
        "battery_q": args.battery_q,
        "evaluation_protocol": _eval_proto(cfg)["name"],
        "replay_protocol": REPLAY_PROTOCOL["name"],
        "workers": int(args.workers),
        "paper_objective": spec,
        "objective_reference_dir": str(Path(args.reference_dir).resolve()) if args.reference_dir else "",
    }

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

    if int(args.workers) > 1:
        summaries = _run_parallel_pairs(
            names=names,
            cfg=cfg,
            budget=float(args.budget),
            out_root=out_root,
            variant=variant,
            workers=int(args.workers),
        )
        _write_combined_summary(summaries, out_root, float(args.budget))
        return 0

    summaries = []
    routes_a: dict[str, list] = {}

    if variant in ("A", "BOTH"):
        sum_a, routes_a = _run_variant(
            variant="A", cfg_base=cfg, names=names, budget=float(args.budget),
            out_dir=out_root / "original_tw",
        )
        summaries.append(sum_a)

    if variant in ("B", "BOTH"):
        warm = None
        if not args.disable_ab_warm_start:
            if variant == "BOTH":
                warm = routes_a
            else:
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

    completeness = {
        "expected_n": len(names),
        "workers": 1,
        "variants": {
            s["tw_variant"]: s.get("completeness") for s in summaries
        },
        "complete": all(bool(s.get("complete")) for s in summaries),
    }
    completeness["cite_means"] = completeness["complete"]
    (out_root / "completeness.json").write_text(
        json.dumps(completeness, indent=2) + "\n", encoding="utf-8",
    )
    _write_combined_summary(summaries, out_root, float(args.budget))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
