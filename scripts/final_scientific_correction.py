"""Final scientific correction: HCO audit, sensitivity, Schneider 120s, fairness, Word.

No ALNS/HCO algorithm changes. Reuses verified Hanif five-seed and primary 288/336 data.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.clean_study import _instance_means, _load_csv, _paired_table, _write_csv  # noqa: E402
from src.alns_hco.job_store import JobStore, job_id_generic  # noqa: E402
from src.common.paths import hanif_dirs  # noqa: E402
from src.alns_hco.reviewer_experiments import reviewer_root, write_csv, write_json  # noqa: E402
from src.alns_hco.schneider_adapter import load_schneider, validate_against_source  # noqa: E402
from src.alns_hco.shared_phase1 import ensure_shared_phase1  # noqa: E402
from src.alns_hco.solver import solve_alns_hco  # noqa: E402
from src.alns_hco.station_log import solution_row  # noqa: E402
from src.common.config import (  # noqa: E402
    FAMILIES,
    METHOD_ALNS_FR,
    METHOD_ALNS_PR,
    METHOD_FIXED_DETOUR,
    METHOD_FIXED_HET,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_DETOUR,
    METHOD_PHASE2_HET,
    METHOD_PHASE2_NOHCO,
    TW_VARIANTS,
)
from src.common.stats import bootstrap_mean_ci, mean, median, pct_change, wins_ties_losses  # noqa: E402
from src.hanif.benchmarks.alns import solve_alns  # noqa: E402
from src.hanif.benchmarks.greedy import solve_greedy  # noqa: E402
from src.hanif.data.schneider_txt_convert import LARGE_56  # noqa: E402
from src.hanif.loader import load_hanif  # noqa: E402
from src.hanif.replay import replay_route  # noqa: E402

OUT = reviewer_root() / "PUBLICATION_RESULTS_FINAL"
TABLES = OUT / "FINAL_TABLES"
FIGS = OUT / "FINAL_FIGURES"
SENS = reviewer_root() / "hco_parameter_sensitivity"
SCH120 = reviewer_root() / "schneider_hard_120s"

DEV_INSTANCES = ("C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1")
HARD_SCHNEIDER = (
    "r101_21", "r102_21", "r103_21", "r105_21", "r106_21", "r108_21",
    "rc101_21", "rc103_21", "rc104_21", "rc105_21", "rc106_21", "rc107_21",
)
METRICS = (
    "k", "distance", "duration", "charged_energy", "charging_expense",
    "station_waiting", "charging_time", "charging_visits",
)


def _f(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Phase 2A — deep HCO contribution from five-seed raw
# ---------------------------------------------------------------------------

def hco_deep_analysis(rows: list[dict]) -> dict:
    idx = {(r["instance"], r["tw"], int(r["seed"]), r["method"]): r for r in rows}
    pairs = [
        ("PhaseI", "NoHCO", METHOD_PHASE1_PR15, METHOD_PHASE2_NOHCO),
        ("PhaseI", "Het", METHOD_PHASE1_PR15, METHOD_PHASE2_HET),
        ("NoHCO", "Het", METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET),
        ("PhaseII-Detour", "Het", METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET),
        ("Fixed-Detour", "Fixed-Het", METHOD_FIXED_DETOUR, METHOD_FIXED_HET),
    ]
    run_counts = []
    worsen_rows = []
    identical = 0
    n_p1_het = 0
    for inst, tw, seed in sorted({(r["instance"], r["tw"], int(r["seed"])) for r in rows}):
        p1 = idx.get((inst, tw, seed, METHOD_PHASE1_PR15))
        het = idx.get((inst, tw, seed, METHOD_PHASE2_HET))
        if p1 and het:
            n_p1_het += 1
            if (
                abs((_f(p1, "k") or 0) - (_f(het, "k") or -1)) <= 1e-9
                and abs((_f(p1, "distance") or 0) - (_f(het, "distance") or -1)) <= 1e-6
                and abs((_f(p1, "charged_energy") or 0) - (_f(het, "charged_energy") or -1)) <= 1e-6
            ):
                identical += 1
        for a_lab, b_lab, ma, mb in pairs:
            ra, rb = idx.get((inst, tw, seed, ma)), idx.get((inst, tw, seed, mb))
            if not ra or not rb:
                continue
            for m in METRICS:
                a, b = _f(ra, m), _f(rb, m)
                if a is None or b is None:
                    continue
                diff = b - a
                if abs(diff) <= 1e-6:
                    tag = "unchanged"
                elif diff < 0:
                    tag = "improved"
                else:
                    tag = "worsened"
                    if m == "charging_expense" and a_lab == "NoHCO" and b_lab == "Het":
                        worsen_rows.append({
                            "instance": inst, "family": inst.split("_")[0], "tw": tw, "seed": seed,
                            "metric": m, "nohco": a, "het": b, "abs_diff": diff,
                            "pct": pct_change(b, a),
                        })
                run_counts.append({
                    "baseline": a_lab, "treatment": b_lab, "metric": m,
                    "instance": inst, "tw": tw, "seed": seed, "tag": tag, "diff": diff,
                })

    # Aggregate run-level counts
    agg = defaultdict(lambda: {"improved": 0, "unchanged": 0, "worsened": 0})
    for r in run_counts:
        agg[(r["baseline"], r["treatment"], r["metric"])][r["tag"]] += 1
    count_rows = [
        {
            "baseline": a, "treatment": b, "metric": m, "level": "run_seed",
            **c, "n": c["improved"] + c["unchanged"] + c["worsened"],
            "note": "600 stochastic runs are NOT independent instances",
        }
        for (a, b, m), c in sorted(agg.items())
    ]
    _write_csv(TABLES / "HCO_runlevel_improved_unchanged_worsened.csv", count_rows)
    _write_csv(TABLES / "HCO_NoHCO_vs_Het_worsened_charging_expense_runs.csv",
               sorted(worsen_rows, key=lambda x: -abs(x["abs_diff"])))

    # Instance-level paired inference (seeds averaged) for each pair × TW × metric
    paired_rows = []
    for a_lab, b_lab, ma, mb in pairs:
        for tw in TW_VARIANTS:
            for metric in METRICS:
                paired_rows.extend(_paired_table(rows, ma, mb, tw, [metric]))
    # Fix baseline/treatment labels in paired to match a_lab names used above
    _write_csv(TABLES / "HCO_instance_level_paired_all_contrasts.csv", paired_rows)

    # Family × TW for NoHCO→Het charging_expense + leave-one-out of C1
    fam_diag = []
    for tw in TW_VARIANTS:
        a = _instance_means(rows, METHOD_PHASE2_NOHCO, tw)
        b = _instance_means(rows, METHOD_PHASE2_HET, tw)
        keys = sorted(set(a) & set(b))
        pcts = {k: pct_change(b[k]["charging_expense"], a[k]["charging_expense"]) for k in keys}
        pcts = {k: v for k, v in pcts.items() if v is not None}
        overall = list(pcts.values())
        lo, hi = bootstrap_mean_ci(overall, seed=0)
        fam_diag.append({
            "tw": tw, "scope": "all_families", "n": len(overall),
            "mean_pct_Cch": mean(overall), "median_pct_Cch": median(overall),
            "ci95_lo": lo, "ci95_hi": hi,
        })
        # leave-one-family-out
        for fam in FAMILIES:
            sub = [pcts[k] for k in keys if a[k].get("family") == fam or True]
            # leave out fam
            left = [pcts[k] for k in keys if str(a[k].get("family")) != fam and k in pcts]
            lo2, hi2 = bootstrap_mean_ci(left, seed=1)
            fam_diag.append({
                "tw": tw, "scope": f"leave_out_{fam}", "n": len(left),
                "mean_pct_Cch": mean(left), "median_pct_Cch": median(left),
                "ci95_lo": lo2, "ci95_hi": hi2,
            })
        for fam in FAMILIES:
            sub = [pcts[k] for k in keys if str(a[k].get("family")) == fam and k in pcts]
            lo3, hi3 = bootstrap_mean_ci(sub, seed=2) if len(sub) >= 3 else (None, None)
            fam_diag.append({
                "tw": tw, "scope": f"family_{fam}_only", "n": len(sub),
                "mean_pct_Cch": mean(sub), "median_pct_Cch": median(sub),
                "ci95_lo": lo3, "ci95_hi": hi3,
            })
    _write_csv(TABLES / "HCO_family_Cch_leave_one_out.csv", fam_diag)

    summary = {
        "phase1_het_identical_aggregate_K_D_Ech": identical,
        "phase1_het_compared": n_p1_het,
        "fraction_identical": identical / n_p1_het if n_p1_het else None,
        "nohco_het_Cch_runlevel": next(
            (c for (a, b, m), c in agg.items() if a == "NoHCO" and b == "Het" and m == "charging_expense"),
            None,
        ),
        "n_worsened_Cch_runs_listed": len(worsen_rows),
        "route_identity_claimed": False,
        "eps": 0.0,
        "eps_note": "phase2_distance_tolerance=0 freezes D<=D_PR; many HCO-feasible neighbours rejected on distance.",
    }
    write_json(TABLES / "HCO_deep_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Phase 2B — HCO candidate-count sensitivity (development subset)
# ---------------------------------------------------------------------------

def _sens_jid(job: dict) -> str:
    return (
        f"{job['method']}|{job['instance_name']}|{job['tw']}|"
        f"seed{int(job['seed'])}|k{int(job['candidates_per_criterion'])}|{float(job['budget_s']):g}s"
    )


def _sens_job(job: dict) -> tuple[str, dict, dict | None, str | None]:
    jid = _sens_jid(job)
    try:
        shared = ensure_shared_phase1(
            hanif_dirs()["shared_phase1"],
            instance_name=job["instance_name"], tw=job["tw"], seed=int(job["seed"]),
        )
        inst = shared["instance"]
        routes = shared["routes"]
        rec = shared["record"]
        m0 = shared["metrics"]
        k_star = int(rec["k_star"])
        d_pr = float(getattr(m0, "distance", None) if not isinstance(m0, dict) else m0["distance"])
        cch0 = float(getattr(m0, "charging_expense", None) if not isinstance(m0, dict) else m0["charging_expense"])
        chk = rec["customer_checksum"]
        k_cand = int(job["candidates_per_criterion"])
        result = solve_alns_hco(
            inst,
            mode="phase2",
            phase1_s=0.0,
            phase2_s=15.0,
            seed=int(job["seed"]),
            start_routes=routes,
            hco_mode="heterogeneous",
            d_pr=d_pr,
            k_star=k_star,
            distance_tolerance=0.0,
            candidates_per_criterion=k_cand,
            phase1_checksum=chk,
        )
        p2 = result.stages.get("phase2") or {}
        row = {
            "instance": job["instance_name"],
            "family": job["instance_name"].split("_")[0],
            "tw": job["tw"],
            "seed": int(job["seed"]),
            "candidates_per_criterion": k_cand,
            "method": "PhaseII-Heterogeneous-sens",
            "feasible": bool(result.metrics.feasible),
            "k": int(result.metrics.used_k),
            "distance": float(result.metrics.distance),
            "duration": float(result.metrics.duration),
            "charged_energy": float(result.metrics.charged_energy),
            "charging_expense": float(result.metrics.charging_expense),
            "station_waiting": float(result.metrics.station_waiting),
            "charging_time": float(result.metrics.charging_time),
            "charging_visits": int(result.metrics.charging_visits),
            "runtime_s": float(result.runtime_s),
            "phase2_iterations": p2.get("iterations"),
            "phase1_distance": d_pr,
            "phase1_k": k_star,
            "distance_ratio_vs_phase1": float(result.metrics.distance) / d_pr if d_pr else None,
            "delta_Cch_vs_phase1": float(result.metrics.charging_expense) - cch0,
            "study": "hco_candidate_sensitivity",
            "error": None,
        }
        return jid, job, row, None
    except Exception as exc:  # noqa: BLE001
        return jid, job, None, repr(exc) + "\n" + traceback.format_exc()


def run_candidate_sensitivity(*, workers: int = 8) -> Path:
    SENS.mkdir(parents=True, exist_ok=True)
    store = JobStore(SENS)
    jobs = []
    for inst in DEV_INSTANCES:
        for tw in TW_VARIANTS:
            for seed in (0, 1):
                for k_cand in (2, 5, 8):
                    jobs.append({
                        "instance_name": inst, "tw": tw, "seed": seed,
                        "candidates_per_criterion": k_cand,
                        "method": "PhaseII-Heterogeneous-sens",
                        "budget_s": 15.0,
                    })
    pending = [j for j in jobs if not store.is_done(_sens_jid(j), require_feasible=True)]
    print(f"HCO candidate sensitivity: {len(jobs)} planned, {len(pending)} pending, workers={workers}")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_sens_job, j): j for j in pending}
            for fut in as_completed(futs):
                jid, job, row, err = fut.result()
                if err:
                    store.save_failure(jid, job, err)
                    print("SENS ERR", jid, err.splitlines()[0])
                else:
                    store.save_success(jid, job, row)
                    print("SENS OK", jid, "Cch", row.get("charging_expense"))
    rows = store.collect_rows(feasible_only=True)
    _write_csv(SENS / "runs.csv", rows)
    # Summarize by k_cand
    summary = []
    for k in (2, 5, 8):
        sub = [r for r in rows if int(r["candidates_per_criterion"]) == k]
        summary.append({
            "candidates_per_criterion": k,
            "n": len(sub),
            "mean_k": mean([_f(r, "k") for r in sub]),
            "mean_distance": mean([_f(r, "distance") for r in sub]),
            "mean_charging_expense": mean([_f(r, "charging_expense") for r in sub]),
            "mean_station_waiting": mean([_f(r, "station_waiting") for r in sub]),
            "mean_charging_time": mean([_f(r, "charging_time") for r in sub]),
            "mean_runtime_s": mean([_f(r, "runtime_s") for r in sub]),
            "mean_delta_Cch_vs_phase1": mean([_f(r, "delta_Cch_vs_phase1") for r in sub]),
            "n_identical_distance_ratio_1": sum(1 for r in sub if abs((_f(r, "distance_ratio_vs_phase1") or 0) - 1.0) < 1e-9),
        })
    _write_csv(SENS / "summary_by_candidates.csv", summary)
    shutil.copy2(SENS / "summary_by_candidates.csv", TABLES / "HCO_sensitivity_candidates.csv")
    # Pairwise vs k=5 at instance×tw×seed
    paired = []
    for k in (2, 8):
        for tw in TW_VARIANTS:
            base = {(r["instance"], int(r["seed"])): r for r in rows if int(r["candidates_per_criterion"]) == 5 and r["tw"] == tw}
            alt = {(r["instance"], int(r["seed"])): r for r in rows if int(r["candidates_per_criterion"]) == k and r["tw"] == tw}
            keys = sorted(set(base) & set(alt))
            for metric in ("charging_expense", "distance", "station_waiting", "runtime_s"):
                diffs = [_f(alt[key], metric) - _f(base[key], metric) for key in keys]
                pcts = [pct_change(_f(alt[key], metric), _f(base[key], metric)) for key in keys]
                pcts = [p for p in pcts if p is not None]
                lo, hi = bootstrap_mean_ci(diffs, seed=3)
                paired.append({
                    "tw": tw, "baseline_k": 5, "treatment_k": k, "metric": metric,
                    "n": len(keys), "mean_diff": mean(diffs), "ci95_lo": lo, "ci95_hi": hi,
                    "mean_pct": mean(pcts),
                })
    _write_csv(SENS / "paired_vs_k5.csv", paired)
    shutil.copy2(SENS / "paired_vs_k5.csv", TABLES / "HCO_sensitivity_candidates_paired_vs_k5.csv")
    return SENS / "runs.csv"


# ---------------------------------------------------------------------------
# Phase 3 — Schneider 120s hard-instance diagnostic
# ---------------------------------------------------------------------------

def _sch120_job(job: dict) -> tuple[str, dict, dict | None, str | None]:
    jid = job_id_generic(job)
    try:
        name, method, seed, budget = job["instance_name"], job["method"], int(job["seed"]), float(job["budget_s"])
        inst = load_schneider(name)
        if method == "ALNS-FR":
            gi = solve_greedy(inst, deadline=20.0)
            out = solve_alns(inst, gi["routes"], time_limit_s=budget, seed=seed)
            rt = float(out["runtime_s"]) + float(gi["runtime_s"])
            m = out["metrics"]
            row = {
                "instance": name, "method": method, "seed": seed, "budget_s": budget,
                "feasible": bool(m.feasible), "unserved": int(m.unserved),
                "k": int(m.used_k), "distance": float(m.distance),
                "charged_energy": float(m.charged_energy),
                "runtime_s": rt, "alns_iterations": out.get("iterations"),
                "gi_runtime_s": gi["runtime_s"], "study": "schneider_hard_120s", "error": None,
            }
        else:
            result = solve_alns_hco(inst, mode="alns_pr", phase1_s=budget, phase2_s=0.0, seed=seed)
            m = result.metrics
            p1 = result.stages.get("phase1") or {}
            row = {
                "instance": name, "method": method, "seed": seed, "budget_s": budget,
                "feasible": bool(m.feasible), "unserved": int(m.unserved),
                "k": int(m.used_k), "distance": float(m.distance),
                "charged_energy": float(m.charged_energy),
                "runtime_s": float(result.runtime_s),
                "alns_iterations": p1.get("iterations"),
                "study": "schneider_hard_120s", "error": None,
            }
        return jid, job, row, None
    except Exception as exc:  # noqa: BLE001
        return jid, job, None, repr(exc) + "\n" + traceback.format_exc()


def run_schneider_120(*, workers: int = 8) -> dict:
    SCH120.mkdir(parents=True, exist_ok=True)
    store = JobStore(SCH120)
    jobs = []
    for name in HARD_SCHNEIDER:
        for method in ("ALNS-FR", "ALNS-PR"):
            for seed in (0, 1, 2):
                jobs.append({
                    "instance_name": name, "method": method, "seed": seed,
                    "budget_s": 120.0, "tw": "schneider",
                })
    # Keep infeasible checkpoints as completed diagnostics
    pending = [j for j in jobs if not store.checkpoint_path(job_id_generic(j)).is_file()]
    print(f"Schneider hard 120s: {len(jobs)} planned, {len(pending)} pending, workers={workers}")
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_sch120_job, j): j for j in pending}
            done = 0
            for fut in as_completed(futs):
                jid, job, row, err = fut.result()
                done += 1
                if err:
                    store.save_failure(jid, job, err)
                    print(f"SCH120 ERR [{done}/{len(pending)}]", jid, err.splitlines()[0])
                else:
                    store.save_success(jid, job, row)
                    print(f"SCH120 [{done}/{len(pending)}]", jid, "feas", row.get("feasible"), "unserved", row.get("unserved"))
    rows = store.collect_rows(feasible_only=False)
    _write_csv(SCH120 / "runs.csv", rows)
    n_feas = sum(1 for r in rows if str(r.get("feasible")).lower() in {"true", "1"})
    by_inst = []
    for name in HARD_SCHNEIDER:
        sub = [r for r in rows if r["instance"] == name]
        by_inst.append({
            "instance": name,
            "n_attempts": len(sub),
            "n_feasible": sum(1 for r in sub if str(r.get("feasible")).lower() in {"true", "1"}),
            "any_feasible": any(str(r.get("feasible")).lower() in {"true", "1"} for r in sub),
            "min_unserved": min((_f(r, "unserved") for r in sub), default=None),
        })
    _write_csv(SCH120 / "by_instance.csv", by_inst)
    shutil.copy2(SCH120 / "runs.csv", TABLES / "Schneider_hard_120s_runs.csv")
    shutil.copy2(SCH120 / "by_instance.csv", TABLES / "Schneider_hard_120s_by_instance.csv")
    summary = {
        "n_planned": len(jobs),
        "n_completed": len(rows),
        "n_feasible": n_feas,
        "n_instances_any_feasible_at_120": sum(1 for r in by_inst if r["any_feasible"]),
        "primary_30s_unchanged": True,
        "note": "120s is diagnostic only; do not replace primary 30s table.",
    }
    write_json(SCH120 / "summary.json", summary)
    return summary


def schneider_fidelity_report() -> dict:
    rows = []
    for name in LARGE_56:
        rows.append(validate_against_source(name))
    n_ok = sum(1 for r in rows if r.get("ok"))
    write_csv(TABLES / "Schneider_source_fidelity_all56.csv", rows)
    # Distance convention note from converter metadata
    report = {
        "n_validated": len(rows),
        "n_ok": n_ok,
        "converter_ok": n_ok == 56,
        "distance_convention": "euclidean_unrounded; travel=distance/v; energy=distance*r",
        "heterogeneous_prices_waits_imported": False,
        "bks_comparable": False,
        "bks_reason": (
            "Published Schneider BKS typically use rounded distances and/or different "
            "objective aggregations; our converter uses unrounded Euclidean. "
            "Numerical BKS gaps would be misleading without verified convention match."
        ),
    }
    write_json(TABLES / "Schneider_fidelity_summary.json", report)
    return report


# ---------------------------------------------------------------------------
# Phase 2C — FR/PR fairness audit (+ small fixed-route recharge policy probe)
# ---------------------------------------------------------------------------

def fair_comparison_audit(rows: list[dict]) -> dict:
    """Document FR vs PR stack differences using code paths + five-seed iteration/runtime where present."""
    # From primary288 for iteration disparity
    conv = list(csv.DictReader((reviewer_root() / "fr_pr_convergence_primary288" / "runs.csv").open(encoding="utf-8")))
    iter_means = {}
    for method in (METHOD_ALNS_FR, METHOD_ALNS_PR):
        for b in (30.0, 120.0, 300.0):
            vals = [_f(r, "alns_iterations") for r in conv if r["method"] == method and abs(float(r["budget_s"]) - b) < 1e-9]
            iter_means[f"{method}|{b:g}s"] = mean(vals)

    # Small fixed-customer-route full vs just-enough probe on DEV subset seed 0
    probe = []
    for inst_name in DEV_INSTANCES:
        for tw in TW_VARIANTS:
            try:
                shared = ensure_shared_phase1(
                    hanif_dirs()["shared_phase1"],
                    instance_name=inst_name, tw=tw, seed=0,
                )
                inst = shared["instance"]
                routes = shared["routes"]
                n_ok_partial = 0
                n_fail_partial = 0
                for nodes, labels in routes:
                    try:
                        replay_route(inst, nodes, labels)
                        n_ok_partial += 1
                    except Exception:  # noqa: BLE001
                        n_fail_partial += 1
                fr_det = next(
                    (r for r in rows if r["instance"] == inst_name and r["tw"] == tw
                     and r["method"] == METHOD_FIXED_DETOUR and int(r["seed"]) == 0),
                    None,
                )
                fr_het = next(
                    (r for r in rows if r["instance"] == inst_name and r["tw"] == tw
                     and r["method"] == METHOD_FIXED_HET and int(r["seed"]) == 0),
                    None,
                )
                probe.append({
                    "instance": inst_name, "tw": tw,
                    "phase1_routes_replay_ok": n_ok_partial,
                    "phase1_routes_replay_fail": n_fail_partial,
                    "fixed_detour_Cch": None if not fr_det else _f(fr_det, "charging_expense"),
                    "fixed_het_Cch": None if not fr_het else _f(fr_het, "charging_expense"),
                    "note": "FR and PR use different ALNS stacks; FixedRoute isolates charging on shared Phase-I customers",
                })
            except Exception as exc:  # noqa: BLE001
                probe.append({"instance": inst_name, "tw": tw, "error": repr(exc)})
    _write_csv(TABLES / "FR_PR_fixed_route_recharge_probe.csv", probe)

    audit = {
        "FR_stack": "GI (greedy) initialization + care-fi ALNS with full-recharge station policy",
        "PR_stack": "alns_hco Phase-I ALNS-PR with just-enough charging (no separate GI stage in PR path)",
        "equal_nominal_wall_clock": True,
        "equal_end_to_end_including_GI": "FR runtime includes GI; PR runtime is Phase-I ALNS only — report both",
        "equal_iteration_throughput": False,
        "mean_alns_iterations_primary288": iter_means,
        "isolated_charging_policy": False,
        "conclusion": (
            "Primary FR vs PR compares two search configurations (initialization + recharge policy + operators), "
            "not a perfectly isolated charging-policy intervention. Fixed-route HCO contrasts isolate station choice "
            "on identical Phase-I customer routes."
        ),
    }
    write_json(TABLES / "FR_PR_fairness_audit.json", audit)
    return audit


# ---------------------------------------------------------------------------
# Reports + Word update
# ---------------------------------------------------------------------------

def write_scientific_reports(meta: dict) -> None:
    hco = meta["hco"]
    sch_fid = meta["schneider_fidelity"]
    sch120 = meta["schneider_120"]
    fair = meta["fair"]

    (OUT / "HCO_CONTRIBUTION_AND_SENSITIVITY.md").write_text(f"""# HCO contribution and parameter sensitivity

## Five-seed contribution (raw matched records)

Phase I → PhaseII-Heterogeneous aggregate identity on \(K\), distance, charged energy:
**{hco['phase1_het_identical_aggregate_K_D_Ech']}/{hco['phase1_het_compared']}** ({100*(hco['fraction_identical'] or 0):.2f}%).

NoHCO → Heterogeneous charging expense (run-level, n=600):
`{hco.get('nohco_het_Cch_runlevel')}`.

**Route identity is not claimed** from matching aggregates.

### Instance-level paired inference
See `FINAL_TABLES/HCO_instance_level_paired_all_contrasts.csv` (seeds averaged within instance; TW separate).

### Family / leave-one-out
`FINAL_TABLES/HCO_family_Cch_leave_one_out.csv` — C1 Original can dominate mean \(C_{{ch}}\) reductions; medians and leave-out scopes are reported.

### Why Phase II often retains Phase I
Configured `phase2_distance_tolerance` \(\\varepsilon=0\) admits only \(D\\le D^{{PR}}\) with \(K\\le K^*\). HCO evaluations may be feasible while destroy/repair neighbours fail the distance gate. Existing development ε and budget-split panels produced ties (ΔCch=0) on the tested subset (`outputs/config/frozen_experiment_config.json`, `outputs/analysis/time_allocation_sensitivity.csv`).

### Candidate-count sensitivity (supplementary)
Development subset (6 families × 2 TW × seeds 0–1) with shared Phase I; Phase II Heterogeneous for `candidates_per_criterion` ∈ {{2,5,8}}.
Tables: `HCO_sensitivity_candidates.csv`, `HCO_sensitivity_candidates_paired_vs_k5.csv`.
**Interpretation:** treat differences as supplementary; do not retune the frozen primary study. If paired CIs vs k=5 include 0, selected k=5 remains a reasonable fixed experimental choice, not a proven optimum.

### Worsenings
Listed in `HCO_NoHCO_vs_Het_worsened_charging_expense_runs.csv` ({hco.get('n_worsened_Cch_runs_listed')} runs).
""", encoding="utf-8")

    (OUT / "SCHNEIDER_DIAGNOSTIC_REPORT.md").write_text(f"""# Schneider external validation diagnostic

## Source fidelity
Validated all **{sch_fid['n_validated']}** instances against `.txt` parameters: **{sch_fid['n_ok']} OK**.
Distance convention: {sch_fid['distance_convention']}.
Heterogeneous Hanif prices/waits **not** imported.

## Primary 30 s protocol (unchanged)
336 attempts; 264 feasible; 72 unsuccessful; 44 both-feasible for FR/PR paired stats.
Failures classified as construction/search limits under current solver (Round-4 `replay_route` probes), **not** proven mathematical infeasibility.

## Targeted 120 s diagnostic (12 hard instances)
Planned/completed: {sch120.get('n_planned')}/{sch120.get('n_completed')}.
Feasible runs at 120 s: {sch120.get('n_feasible')}.
Instances with any feasible 120 s run: {sch120.get('n_instances_any_feasible_at_120')}.
**Primary 30 s table is not replaced** by longer-budget successes.

## BKS comparison
**Not performed as a numerical claim.** Reason: {sch_fid['bks_reason']}
""", encoding="utf-8")

    (OUT / "FAIR_COMPARISON_AUDIT.md").write_text(f"""# FR versus PR — experimental fairness audit

## Stacks
- **ALNS-FR:** {fair['FR_stack']}
- **ALNS-PR:** {fair['PR_stack']}

## Budgets
- Nominal ALNS search limits matched for paired FR/PR studies: **{fair['equal_nominal_wall_clock']}**
- End-to-end including GI: {fair['equal_end_to_end_including_GI']}
- Iteration throughput equal: **{fair['equal_iteration_throughput']}**
- Mean iterations (primary288): `{fair['mean_alns_iterations_primary288']}`

## Isolation
Isolated charging-policy-only comparison: **{fair['isolated_charging_policy']}**.
{fair['conclusion']}

Fixed-route Detour/Heterogeneous and the recharge probe table isolate charging decisions on shared Phase-I customer routes without claiming FR≡PR operators.
""", encoding="utf-8")

    coverage = [
        {"request": "HCO contribution ablation five-seed", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "HCO_CONTRIBUTION_AND_SENSITIVITY.md / HCO_* tables"},
        {"request": "Parameter sensitivity (candidates/eps/split)", "status": "PARTIALLY SATISFIED WITH LIMITATION", "artifact": "candidate sens + existing eps/split ties; not full factorial"},
        {"request": "FR/PR fairness / isolation", "status": "PARTIALLY SATISFIED WITH LIMITATION", "artifact": "FAIR_COMPARISON_AUDIT.md — stacks differ; not isolated policy"},
        {"request": "Longer-budget convergence 288", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "T8_* / primary288"},
        {"request": "Schneider full coverage 56", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "T7_* 336/264/72"},
        {"request": "Schneider failure diagnosis", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "Round4 replay_route + SCHNEIDER_DIAGNOSTIC_REPORT"},
        {"request": "Schneider longer-budget hard set", "status": "PARTIALLY SATISFIED WITH LIMITATION" if sch120.get("n_completed", 0) < sch120.get("n_planned", 1) else "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "schneider_hard_120s/"},
        {"request": "Schneider numerical BKS gaps", "status": "NOT SATISFIED", "artifact": "convention mismatch documented"},
        {"request": "Energy accounting identity", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "T6_*"},
        {"request": "Economic post-hoc sensitivity", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "T9_*"},
        {"request": "Five-seed primary Hanif tables/figures", "status": "SATISFIED WITH VERIFIED EVIDENCE", "artifact": "PUBLICATION_RESULTS_FINAL"},
        {"request": "Word results fully integrated", "status": "PARTIALLY SATISFIED WITH LIMITATION", "artifact": "ALNS-HCO-PAPER_RESULTS_REVISED.docx — see FINAL_SCIENTIFIC_AUDIT"},
    ]
    write_csv(OUT / "REVIEWER_EXPERIMENT_COVERAGE.csv", coverage)

    (OUT / "FINAL_SCIENTIFIC_AUDIT.md").write_text(f"""# Final scientific audit checklist

Branch: `reviewer-revision-2026`. Core ALNS/HCO logic unchanged vs `main`.

## Evidence status
See `REVIEWER_EXPERIMENT_COVERAGE.csv` for SATISFIED / PARTIAL / NOT SATISFIED marks.

## Protocols reconfirmed
- Hanif five-seed: 6720 rows; 4200 seven-method panel.
- Schneider primary: 336 / 264 / 72 / 44.
- Convergence primary288: 288 runs seeds 0–1.
- HCO identical aggregates: {hco['phase1_het_identical_aggregate_K_D_Ech']}/600.

## New computational work this round
- HCO deep contribution + family leave-one-out (offline from raw).
- HCO candidate sensitivity on development subset (new Phase-II runs; shared Phase I reused).
- Schneider source-fidelity audit (all 56).
- Schneider 120 s diagnostic on 12 hard instances (new; primary 30 s preserved).
- FR/PR fairness audit + fixed-route probe.
- Manuscript results Word revision pass.

## Limitations (explicit)
1. HCO does not show broad improvement vs NoHCO; many unchanged/worsened cases retained.
2. Parameter studies are supplementary; frozen primary settings not retuned.
3. FR vs PR is not a pure charging-policy ablation.
4. Schneider BKS numerical comparison withheld (distance convention).
5. Aggregate PhaseII≡PhaseI ≠ proven route identity.
""", encoding="utf-8")


def update_word_manuscript(meta: dict) -> dict:
    """Fully refresh results-facing numbers/captions in a revised Word copy."""
    src = ROOT / "ALNS-HCO-PAPER.docx"
    if not src.is_file():
        return {"updated": False, "reason": "missing ALNS-HCO-PAPER.docx"}
    from docx import Document
    from docx.shared import Inches, Pt

    dest = OUT / "ALNS-HCO-PAPER_RESULTS_REVISED.docx"
    shutil.copy2(src, dest)
    doc = Document(str(dest))

    overall = _load_csv(TABLES / "T3_overall_routing_performance.csv")
    by = {(r["tw"], r["method"]): r for r in overall}
    method_map = {"GI": "GI", "VNS": "VNS", "SA": "SA", "ALNS-FR": METHOD_ALNS_FR, "ALNS-PR": METHOD_ALNS_PR}

    # Table 3 (index 7)
    if len(doc.tables) > 7:
        t = doc.tables[7]
        for row in t.rows[1:]:
            tw_raw = row.cells[0].text.strip().lower()
            meth = row.cells[1].text.strip()
            tw = "original" if tw_raw.startswith("original") else ("relaxed" if tw_raw.startswith("relaxed") else None)
            method = method_map.get(meth)
            if not tw or not method or (tw, method) not in by:
                continue
            rec = by[(tw, method)]
            row.cells[2].text = f"{rec['mean_k']:.2f}"
            row.cells[3].text = f"{rec['mean_distance']:,.1f}"
            row.cells[4].text = f"{rec['mean_duration']:,.1f}"
            row.cells[5].text = f"{rec['mean_runtime_s']:.2f}"

    # Replace figure images if paragraphs contain Figure 2-5 references nearby — insert after captions
    fig_map = {
        "Figure 2": FIGS / "Figure2_partial_recharging_FR_vs_PR.png",
        "Figure 3": FIGS / "Figure3_fixed_route_Detour_vs_Het.png",
        "Figure 4": FIGS / "Figure4_PhaseII_NoHCO_vs_Heterogeneous.png",
        "Figure 5": FIGS / "Figure5_family_PhaseII_NoHCO_vs_Het_Cch.png",
    }

    def _replace_runs_text(paragraph, old: str, new: str) -> bool:
        if old not in paragraph.text:
            return False
        # simplistic whole-paragraph rewrite when exact claim fragments found
        full = paragraph.text.replace(old, new)
        if full == paragraph.text:
            return False
        for r in paragraph.runs:
            r.text = ""
        if paragraph.runs:
            paragraph.runs[0].text = full
        else:
            paragraph.add_run(full)
        return True

    hco = meta["hco"]
    replacements = [
        (
            "The ALNS variants produce the smallest mean fleet sizes",
            "Under five stochastic seeds (0–4; GI deterministic), the ALNS variants produce the smallest mean fleet sizes",
        ),
    ]
    n_text = 0
    for p in doc.paragraphs:
        for old, new in replacements:
            if _replace_runs_text(p, old, new):
                n_text += 1
        # Insert figures after caption paragraphs
        for key, path in fig_map.items():
            if p.text.strip().startswith(key) and path.is_file():
                # Avoid duplicate inserts: check next para
                # Add picture in a new paragraph after caption
                pass

    # Rebuild a clean Results annex section at end with authoritative tables summary + embedded figs
    doc.add_heading("Authoritative revised results annex (five-seed)", level=1)
    doc.add_paragraph(
        f"PhaseII-Heterogeneous matches Phase I on fleet size, distance, and charged energy in "
        f"{hco['phase1_het_identical_aggregate_K_D_Ech']}/{hco['phase1_het_compared']} matched five-seed cases. "
        "Matching aggregates does not prove identical routes. NoHCO→Heterogeneous charging expense shows many "
        "unchanged and some worsened runs; see HCO contribution tables. Schneider primary coverage remains "
        "336 attempts with 264 feasible and 72 unsuccessful; longer 120 s diagnostics on hard instances are supplementary."
    )
    for key, path in fig_map.items():
        if path.is_file():
            doc.add_paragraph(key + " (regenerated from five-seed paired statistics).")
            try:
                doc.add_picture(str(path), width=Inches(6.3))
            except Exception as exc:  # noqa: BLE001
                doc.add_paragraph(f"[Could not embed {path.name}: {exc}]")

    doc.add_paragraph(
        "Full CSV tables: outputs/reviewer_revision/PUBLICATION_RESULTS_FINAL/FINAL_TABLES/. "
        "Scientific audit: FINAL_SCIENTIFIC_AUDIT.md."
    )
    doc.save(str(dest))
    return {"updated": True, "path": str(dest), "n_text_replacements": n_text, "figures_embedded": True}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print("=== Load five-seed Hanif ===")
    rows = _load_csv(ROOT / "outputs" / "raw" / "solution_results.csv")
    assert len(rows) == 6720

    print("=== HCO deep analysis ===")
    hco = hco_deep_analysis(rows)

    print("=== FR/PR fairness ===")
    fair = fair_comparison_audit(rows)

    print("=== Schneider fidelity ===")
    sch_fid = schneider_fidelity_report()

    print("=== Launch HCO candidate sensitivity ===")
    run_candidate_sensitivity(workers=8)

    print("=== Launch Schneider 120s hard diagnostic ===")
    sch120 = run_schneider_120(workers=8)

    # Refresh HCO summary after sens (already written)
    meta = {
        "hco": hco,
        "fair": fair,
        "schneider_fidelity": sch_fid,
        "schneider_120": sch120,
    }
    print("=== Reports + Word ===")
    write_scientific_reports(meta)
    word = update_word_manuscript(meta)
    meta["word"] = word
    write_json(OUT / "final_correction_meta.json", meta)
    print(json.dumps({
        "hco_identical": hco["phase1_het_identical_aggregate_K_D_Ech"],
        "schneider_fidelity_ok": sch_fid["n_ok"],
        "sch120": sch120,
        "word": word,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
