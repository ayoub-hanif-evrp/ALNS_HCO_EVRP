"""Scope B: 56 Schneider ALNS-FR × seed 0 × 30s with route archives + external verification.

Does NOT modify the original primary 336-run archive.
Matches primary Schneider FR config: GI deadline=20s, ALNS budget=30s, seed=0.
"""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.clean_study import _load_csv, _write_csv  # noqa: E402
from src.alns_hco.reviewer_experiments import reviewer_root, write_csv, write_json  # noqa: E402
from src.alns_hco.schneider_adapter import load_schneider  # noqa: E402
from src.common.paired_stats import paired_instance_effects, summarize_paired  # noqa: E402
from src.common.stats import bootstrap_mean_ci, mean  # noqa: E402
from src.hanif.benchmarks.alns import solve_alns  # noqa: E402
from src.hanif.benchmarks.greedy import solve_greedy  # noqa: E402
from src.hanif.data.schneider_txt_convert import LARGE_56, schneider_family  # noqa: E402
from src.hanif.evaluation.schneider_bks import compare_row  # noqa: E402
from src.hanif.replay import replay_solution  # noqa: E402

OUT = reviewer_root() / "PUBLICATION_RESULTS_FINAL"
SCOPE = reviewer_root() / "schneider_scope_b"
CKPT = SCOPE / "checkpoints"
SOL = SCOPE / "verifier_solutions"
LOGS = OUT / "QC_LOGS"
TABLES = OUT / "FINAL_TABLES"
INST_DIR = ROOT / "dataset" / "Schneider" / "evrptw_instances"
TOOLS = reviewer_root() / "tools"
JAR = TOOLS / "evrptw_verifier_dist" / "evrptw-verifier-0.2.0.jar"
JAVA = TOOLS / "jdk17" / "bin" / "java.exe"
MAIN = "ghiermann.otl.evrptw.verifier.Main"
PRIMARY_RUNS = reviewer_root() / "schneider" / "runs_including_infeasible.csv"
PRIMARY_CK = reviewer_root() / "schneider" / "checkpoints"

GI_DEADLINE = 20.0
BUDGET = 30.0
SEED = 0
METHOD = "ALNS-FR"


def _export_routes(inst, routes):
    out = []
    for nodes, labs in routes:
        seq = []
        for n in nodes:
            node = inst.nodes[int(n)]
            sid = getattr(node, "string_id", None) or str(n)
            seq.append(str(sid))
        out.append({
            "nodes_int": [int(n) for n in nodes],
            "string_ids": seq,
            "labels": list(labs) if labs is not None else None,
        })
    return out


def _collapse(ids: list[str]) -> list[str]:
    out = []
    for s in ids:
        if out and out[-1] == s:
            continue
        out.append(s)
    return out


def _coords(instance_txt: Path) -> dict[str, tuple[float, float]]:
    coords = {}
    with instance_txt.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] == "StringID":
                continue
            if len(parts) >= 4 and parts[1] in {"d", "f", "c"}:
                coords[parts[0]] = (float(parts[2]), float(parts[3]))
            elif "/" in line and parts[0] in {"Q", "C", "r", "g", "v"}:
                break
    return coords


def _euclid_total(coords, routes_ids: list[list[str]]) -> float:
    tot = 0.0
    for ids in routes_ids:
        for a, b in zip(ids, ids[1:]):
            xa, ya = coords[a]
            xb, yb = coords[b]
            tot += math.sqrt((xa - xb) ** 2 + (ya - yb) ** 2)
    return tot


def run_one(name: str) -> dict:
    inst = load_schneider(name)
    t0 = time.perf_counter()
    gi = solve_greedy(inst, deadline=GI_DEADLINE)
    out = solve_alns(inst, gi["routes"], time_limit_s=BUDGET, seed=SEED)
    routes = out["routes"]
    m = replay_solution(inst, routes)
    export = _export_routes(inst, routes)
    wall = time.perf_counter() - t0
    payload = {
        "instance": name,
        "family": schneider_family(name),
        "method": METHOD,
        "seed": SEED,
        "budget_s": BUDGET,
        "gi_deadline_s": GI_DEADLINE,
        "config_note": "Matches primary Schneider FR runner (scripts/run_schneider_experiments.py)",
        "feasible": bool(m.feasible),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
        "k": int(m.used_k),
        "distance": float(m.distance),
        "duration": float(m.duration),
        "gi_runtime_s": float(gi["runtime_s"]),
        "alns_runtime_s": float(out["runtime_s"]),
        "solver_runtime_s": float(out["runtime_s"]) + float(gi["runtime_s"]),
        "wall_s": wall,
        "alns_iterations": out.get("iterations"),
        "routes": export,
        "study": "schneider_scope_b",
        "does_not_replace_primary336": True,
    }
    return payload


def _worker(name: str):
    try:
        return name, run_one(name), None
    except Exception as exc:  # noqa: BLE001
        return name, None, traceback.format_exc()


def load_primary_index() -> dict[str, dict]:
    rows = _load_csv(PRIMARY_RUNS)
    idx = {}
    for r in rows:
        if r.get("method") != METHOD:
            continue
        if abs(float(r.get("budget_s") or 0) - BUDGET) > 1e-9:
            continue
        if int(float(r.get("seed"))) != SEED:
            continue
        idx[r["instance"]] = r
    return idx


def compare_primary(payload: dict, primary: dict | None) -> dict:
    if not primary:
        return {"repro_status": "PRIMARY_MISSING", "k_match": None, "distance_match": None, "feasible_match": None}
    pk = int(float(primary["k"]))
    pd = float(primary["distance"])
    pf = str(primary.get("feasible")).lower() in {"1", "true"}
    k_match = int(payload["k"]) == pk
    feas_match = bool(payload["feasible"]) == pf
    dist_match = abs(float(payload["distance"]) - pd) < 1e-6
    if k_match and dist_match and feas_match:
        status = "REPRODUCED_EXACT"
    elif feas_match and k_match and abs(float(payload["distance"]) - pd) / max(pd, 1e-9) < 1e-4:
        status = "REPRODUCED_NEAR"
    else:
        status = "DISCREPANCY"
    return {
        "repro_status": status,
        "primary_feasible": pf,
        "primary_k": pk,
        "primary_distance": pd,
        "primary_unserved": primary.get("unserved"),
        "k_match": k_match,
        "distance_match": dist_match,
        "feasible_match": feas_match,
        "distance_abs_diff": float(payload["distance"]) - pd,
    }


def write_sol(path: Path, cost: float, routes: list[list[str]], comment: str):
    lines = [f"# {comment}", f"{cost:.3f}"]
    for r in routes:
        lines.append(", ".join(r))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_verifier(instance_txt: Path, sol: Path) -> dict:
    if not JAVA.is_file() or not JAR.is_file():
        return {"external_valid": False, "error": "java_or_jar_missing", "stdout": ""}
    cmd = [str(JAVA), "-cp", str(JAR), MAIN, "-d", str(instance_txt), str(sol)]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    out = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    valid = bool(re.search(r"\[_valid_\]", out))
    m_cost = re.search(r"Total cost = ([0-9.]+)", out)
    missing = re.search(r"customers missing: (.+)", out)
    return {
        "external_valid": valid,
        "verifier_total_cost": float(m_cost.group(1)) if m_cost else None,
        "customers_missing": missing.group(1).strip() if missing else "",
        "stdout": out,
        "returncode": proc.returncode,
    }


def collapse_study(inst, export_routes: list[dict]) -> dict:
    """Compare raw vs consecutive-identical collapse under our replay."""
    raw = [(r["nodes_int"], r["labels"]) for r in export_routes]
    m_raw = replay_solution(inst, raw)

    # Collapse consecutive identical string IDs, map back to ints via string_id
    sid_to_int = {}
    for i, node in enumerate(inst.nodes):
        sid = getattr(node, "string_id", None) or str(i)
        sid_to_int[str(sid)] = int(i)

    collapsed_routes = []
    n_removed = 0
    for r in export_routes:
        ids = r["string_ids"]
        labs = r["labels"] or [None] * len(ids)
        new_ids = []
        new_labs = []
        for s, lab in zip(ids, labs):
            if new_ids and new_ids[-1] == s:
                n_removed += 1
                continue
            new_ids.append(s)
            new_labs.append(lab)
        nodes = [sid_to_int[s] for s in new_ids]
        collapsed_routes.append((nodes, new_labs))
    m_col = replay_solution(inst, collapsed_routes)
    return {
        "n_consecutive_removed": n_removed,
        "raw_feasible": bool(m_raw.feasible),
        "collapsed_feasible": bool(m_col.feasible),
        "raw_distance": float(m_raw.distance),
        "collapsed_distance": float(m_col.distance),
        "distance_change": float(m_col.distance) - float(m_raw.distance),
        "raw_unserved": int(m_raw.unserved),
        "collapsed_unserved": int(m_col.unserved),
        "feasibility_preserved": bool(m_raw.feasible) == bool(m_col.feasible) and int(m_raw.unserved) == int(m_col.unserved),
        "collapse_conceals_violation": bool(m_raw.feasible) is False and bool(m_col.feasible) is True,
    }


def bks_row(payload: dict, cost_3dp: float, external_valid: bool) -> dict:
    info = compare_row("fr", payload["instance"], int(payload["k"]), cost_3dp)
    our_k = int(payload["k"])
    bks_k = info.get("bks_k")
    bks_d = info.get("bks_d")
    if not external_valid or not payload["feasible"]:
        return {
            **{f"bks_{k}": info.get(k) for k in ("bks_k", "bks_d", "bks_source_name")},
            "lex_verdict": "NOT_COMPARED_INVALID_OR_INFEASIBLE",
            "distance_gap_pct": None,
            "fleet_delta": None,
        }
    if bks_k is None:
        return {"bks_k": None, "bks_d": None, "bks_source_name": None,
                "lex_verdict": "NO_BKS_INVENTORY", "distance_gap_pct": None, "fleet_delta": None}
    fleet_delta = our_k - int(bks_k)
    if fleet_delta < 0:
        verdict = "FLEET_BETTER_THAN_BKS"
    elif fleet_delta > 0:
        verdict = "FLEET_WORSE_THAN_BKS"
    else:
        verdict = "FLEET_TIE"
    gap = None if bks_d is None else 100.0 * (cost_3dp - float(bks_d)) / float(bks_d)
    return {
        "bks_k": bks_k,
        "bks_d": bks_d,
        "bks_source_name": info.get("bks_source_name"),
        "fleet_delta": fleet_delta,
        "lex_verdict": verdict,
        "distance_gap_pct": gap,
        "distance_gap_is_secondary": fleet_delta != 0,
    }


def plain_vs_het() -> dict:
    raw = _load_csv(ROOT / "outputs" / "raw" / "solution_results.csv")
    summaries = []
    effects_all = []
    for tw in ("original", "relaxed"):
        for metric in ("k", "distance", "charging_expense", "charged_energy", "station_waiting"):
            eff = paired_instance_effects(
                raw, method_a="PhaseII-Plain", method_b="PhaseII-Heterogeneous",
                metric=metric, tw=tw, budget_s=None, require_budget=False,
            )
            # paired_instance_effects may filter by budget - check if empty
            if not eff:
                # manual seed-average pairing
                from collections import defaultdict
                cells = defaultdict(list)
                for r in raw:
                    if r.get("method") not in {"PhaseII-Plain", "PhaseII-Heterogeneous"}:
                        continue
                    if r.get("tw") != tw:
                        continue
                    if str(r.get("feasible")).lower() not in {"1", "true"}:
                        continue
                    v = r.get(metric)
                    if v in (None, ""):
                        continue
                    cells[(r["instance"], r["method"])].append(float(v))
                # instances with both methods
                insts = sorted({i for i, m in cells})
                eff = []
                for inst in insts:
                    a = cells.get((inst, "PhaseII-Plain"))
                    b = cells.get((inst, "PhaseII-Heterogeneous"))
                    if not a or not b:
                        continue
                    am, bm = mean(a), mean(b)
                    diff = bm - am
                    pct = None if abs(am) < 1e-12 else 100.0 * diff / am
                    eff.append({
                        "instance": inst, "a_mean": am, "b_mean": bm,
                        "diff_b_minus_a": diff, "pct_b_vs_a": pct, "n_matched_seeds": min(len(a), len(b)),
                    })
            sm = summarize_paired(eff, seed=0)
            # count distance wins
            n_plain_strictly_better_dist = None
            if metric == "distance":
                n_plain_strictly_better_dist = sum(1 for e in eff if e["diff_b_minus_a"] > 0)  # Het - Plain > 0 => Plain better
                n_het_better = sum(1 for e in eff if e["diff_b_minus_a"] < 0)
                n_tie = sum(1 for e in eff if abs(e["diff_b_minus_a"]) < 1e-9)
                sm["n_plain_lower_distance"] = n_plain_strictly_better_dist
                sm["n_het_lower_distance"] = n_het_better
                sm["n_distance_ties"] = n_tie
            if metric == "k":
                sm["n_equal_k"] = sum(1 for e in eff if abs(e["diff_b_minus_a"]) < 1e-9)
            summaries.append({"tw": tw, "metric": metric, "baseline": "PhaseII-Plain",
                              "treatment": "PhaseII-Heterogeneous", **sm})
            if metric == "distance":
                effects_all.extend([{**e, "tw": tw} for e in eff])
    _write_csv(TABLES / "T_PhaseII_Plain_vs_Heterogeneous_paired.csv", summaries)
    _write_csv(TABLES / "T_PhaseII_Plain_vs_Heterogeneous_distance_by_instance.csv", effects_all)
    return {"summaries": summaries, "n_distance_effects": len(effects_all)}


def analyze_and_verify(payloads: list[dict], primary_idx: dict) -> list[dict]:
    rows = []
    for payload in payloads:
        name = payload["instance"]
        CKPT.mkdir(parents=True, exist_ok=True)
        SOL.mkdir(parents=True, exist_ok=True)
        ck_path = CKPT / f"ALNS-FR__{name}__seed{SEED}__30s_routes.json"
        ck_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        cmp = compare_primary(payload, primary_idx.get(name))
        inst = load_schneider(name)
        col = collapse_study(inst, payload["routes"])

        coords = _coords(INST_DIR / f"{name}.txt")
        raw_ids = [r["string_ids"] for r in payload["routes"]]
        col_ids = [_collapse(r) for r in raw_ids]
        cost_raw = _euclid_total(coords, raw_ids)
        cost_col = _euclid_total(coords, col_ids)

        # Prefer collapsed for Schneider-style single full-recharge visit IF collapse does not conceal violations
        use_collapse = (not col["collapse_conceals_violation"]) and col["n_consecutive_removed"] > 0
        # Always test BOTH
        sol_raw = SOL / f"{name}__raw.sol.txt"
        sol_col = SOL / f"{name}__collapsed.sol.txt"
        write_sol(sol_raw, cost_raw, raw_ids, f"{name} ScopeB raw FR seed0")
        write_sol(sol_col, cost_col, col_ids, f"{name} ScopeB collapsed FR seed0")

        ver_raw = run_verifier(INST_DIR / f"{name}.txt", sol_raw)
        ver_col = run_verifier(INST_DIR / f"{name}.txt", sol_col)
        (LOGS / f"scopeB_verifier_{name}_raw.txt").write_text(ver_raw.get("stdout", ""), encoding="utf-8")
        (LOGS / f"scopeB_verifier_{name}_collapsed.txt").write_text(ver_col.get("stdout", ""), encoding="utf-8")

        # Reporting export: collapsed only if it does not conceal; else raw
        report_variant = "collapsed" if (use_collapse and not col["collapse_conceals_violation"]) else "raw"
        ver_rep = ver_col if report_variant == "collapsed" else ver_raw
        cost_rep = float(f"{(cost_col if report_variant == 'collapsed' else cost_raw):.3f}")
        bks = bks_row(payload, cost_rep, bool(ver_rep.get("external_valid")) and bool(payload["feasible"]))

        rows.append({
            "instance": name,
            "family": payload["family"],
            "seed": SEED,
            "budget_s": BUDGET,
            "gi_deadline_s": GI_DEADLINE,
            "feasible": payload["feasible"],
            "unserved": payload["unserved"],
            "k": payload["k"],
            "distance": payload["distance"],
            "solver_runtime_s": payload["solver_runtime_s"],
            "alns_iterations": payload["alns_iterations"],
            "checkpoint_path": str(ck_path),
            **cmp,
            **{f"collapse_{k}": v for k, v in col.items()},
            "sol_cost_raw_3dp": float(f"{cost_raw:.3f}"),
            "sol_cost_collapsed_3dp": float(f"{cost_col:.3f}"),
            "external_valid_raw": bool(ver_raw.get("external_valid")),
            "external_valid_collapsed": bool(ver_col.get("external_valid")),
            "customers_missing_raw": ver_raw.get("customers_missing", ""),
            "customers_missing_collapsed": ver_col.get("customers_missing", ""),
            "verifier_cost_raw": ver_raw.get("verifier_total_cost"),
            "verifier_cost_collapsed": ver_col.get("verifier_total_cost"),
            "report_variant": report_variant,
            "report_external_valid": bool(ver_rep.get("external_valid")),
            "report_sol_cost_3dp": cost_rep,
            **bks,
        })
    return rows


def write_analysis(rows: list[dict], plain_info: dict) -> None:
    n = len(rows)
    n_feas = sum(1 for r in rows if str(r["feasible"]).lower() in {"1", "true", "True"})
    n_repro = sum(1 for r in rows if r["repro_status"] == "REPRODUCED_EXACT")
    n_near = sum(1 for r in rows if r["repro_status"] == "REPRODUCED_NEAR")
    n_disc = sum(1 for r in rows if r["repro_status"] == "DISCREPANCY")
    n_valid_raw = sum(1 for r in rows if r["external_valid_raw"])
    n_valid_col = sum(1 for r in rows if r["external_valid_collapsed"])
    n_valid_rep = sum(1 for r in rows if r["report_external_valid"])
    n_conceal = sum(1 for r in rows if r.get("collapse_collapse_conceals_violation"))
    n_fleet_tie = sum(1 for r in rows if r.get("lex_verdict") == "FLEET_TIE")
    n_fleet_worse = sum(1 for r in rows if r.get("lex_verdict") == "FLEET_WORSE_THAN_BKS")
    n_fleet_better = sum(1 for r in rows if r.get("lex_verdict") == "FLEET_BETTER_THAN_BKS")

    # BKS provenance note from inventory
    md = [
        "# Schneider Scope B — 56 ALNS-FR (seed 0, 30 s)",
        "",
        "## Protocol",
        f"- Instances: all **{n}** large Schneider (LARGE_56)",
        f"- Method: ALNS-FR; seed={SEED}; GI deadline={GI_DEADLINE}s; ALNS budget={BUDGET}s",
        "- Matches `scripts/run_schneider_experiments.py` FR path (primary config).",
        "- **Does not replace** the original primary 336-run archive.",
        "",
        "## Reproducibility vs primary seed-0 FR checkpoints",
        f"- Exact match (feasible+K+distance): **{n_repro}/{n}**",
        f"- Near distance match: **{n_near}/{n}**",
        f"- Discrepancy: **{n_disc}/{n}**",
        f"- Scope-B feasible: **{n_feas}/{n}** (primary seed0 FR was 44/56)",
        "",
        "## External verifier (ghiermann/evrptw-verifier)",
        "- Distance convention in verifier source: **unrounded Euclidean arcs**; total rounded to **3** decimals.",
        "- Station visit ⇒ full recharge (matches FR policy).",
        f"- External valid (raw exports): **{n_valid_raw}/{n}**",
        f"- External valid (collapsed consecutive identical IDs): **{n_valid_col}/{n}**",
        f"- Report variant valid: **{n_valid_rep}/{n}**",
        f"- Collapse conceals violation (forbidden for claims): **{n_conceal}**",
        "",
        "## BKS comparison (lexicographic; externally valid complete solutions only)",
        f"- Fleet better / tie / worse vs inventory BKS: **{n_fleet_better} / {n_fleet_tie} / {n_fleet_worse}**",
        "- Distance gaps reported for fleet ties; marked secondary when fleet differs.",
        "- Inventory sources: Schneider et al. (2014) / Schiffer & Walther as coded in `schneider_bks.py`.",
        "",
        "## Recommendation",
    ]
    coverage = n_valid_rep / n if n else 0
    repro_rate = (n_repro + n_near) / n if n else 0
    if repro_rate >= 0.9 and coverage >= 0.7:
        md.append(
            "Scope B reproducibility and verification coverage support optionally extending to "
            "**168 FR runs (56×3 seeds)** before considering the full 336 (do not rerun PR for this verifier)."
        )
        recommend_168 = True
    else:
        md.append(
            "Scope B should be reviewed carefully before a 168-run extension: "
            f"repro_rate={repro_rate:.2f}, external_valid_report={coverage:.2f}."
        )
        recommend_168 = False

    md += [
        "",
        "## Artifacts",
        "- Table: `FINAL_TABLES/Schneider_ScopeB_56_FR_seed0.csv`",
        "- Routes: `outputs/reviewer_revision/schneider_scope_b/checkpoints/`",
        "- Verifier logs: `QC_LOGS/scopeB_verifier_*.txt`",
        "",
    ]
    (OUT / "SCHNEIDER_SCOPE_B_ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (SCOPE / "ANALYSIS.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    write_json(SCOPE / "summary.json", {
        "n": n, "n_feasible": n_feas, "n_reproduced_exact": n_repro, "n_reproduced_near": n_near,
        "n_discrepancy": n_disc, "n_external_valid_raw": n_valid_raw, "n_external_valid_collapsed": n_valid_col,
        "n_report_external_valid": n_valid_rep, "n_collapse_conceals": n_conceal,
        "n_fleet_better": n_fleet_better, "n_fleet_tie": n_fleet_tie, "n_fleet_worse": n_fleet_worse,
        "recommend_168_fr": recommend_168, "plain_vs_het": plain_info,
    })
    write_json(LOGS / "schneider_scope_b_summary.json", {
        "n": n, "n_feasible": n_feas, "n_reproduced_exact": n_repro, "recommend_168_fr": recommend_168,
    })


def update_author_package(rows: list[dict], plain_info: dict) -> None:
    # Patch FINAL_VERIFIED with Scope B + Plain vs Het section
    plain_lines = [
        "",
        "## PhaseII-Plain vs PhaseII-Heterogeneous (five-seed; unfavorable control evidence)",
        "",
        "PhaseII-Plain uses Phase-I destroy operators (no Phase-II charging-aware destroy / HCO).",
        "PhaseII-NoHCO uses Phase-II destroy/search without HCO station selection.",
        "PhaseII-Heterogeneous adds HCO (heterogeneous scoring).",
        "",
        "Paired seed-averaged results (treatment − baseline = Het − Plain):",
        "",
        "| TW | Metric | n | Mean diff (Het−Plain) | Mean paired % | 95% CI (diff) | Notes |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for s in plain_info["summaries"]:
        note = ""
        if s["metric"] == "distance":
            note = f"Plain lower on {s.get('n_plain_lower_distance')}/{s['n_instances']}"
        if s["metric"] == "k":
            note = f"equal K on {s.get('n_equal_k')}/{s['n_instances']}"
        md = s.get("mean_diff")
        mp = s.get("mean_pct")
        lo, hi = s.get("ci95_diff_lo"), s.get("ci95_diff_hi")
        plain_lines.append(
            f"| {s['tw']} | {s['metric']} | {s['n_instances']} | "
            f"{'—' if md is None else f'{md:.6g}'} | "
            f"{'—' if mp is None else f'{mp:+.3f}%'} | "
            f"[{'' if lo is None else f'{lo:.4g}'}, {'' if hi is None else f'{hi:.4g}'}] | {note} |"
        )
    plain_lines += [
        "",
        "**Interpretation:** Plain achieves lower distance than Heterogeneous on all 60 base instances in each TW "
        "with equal mean fleet size — HCO Heterogeneous is not a universal routing improvement over the Plain "
        "negative control. Source: `FINAL_TABLES/T_PhaseII_Plain_vs_Heterogeneous_*.csv`.",
        "",
        "## Schneider Scope B (new)",
        "See `SCHNEIDER_SCOPE_B_ANALYSIS.md` and `FINAL_TABLES/Schneider_ScopeB_56_FR_seed0.csv`.",
        "Primary 336 archive unchanged. External verifier applied to all 56 Scope-B FR exports.",
        "",
    ]

    auth = OUT / "FINAL_VERIFIED_RESULTS_FOR_AUTHOR.md"
    text = auth.read_text(encoding="utf-8") if auth.is_file() else "# FINAL VERIFIED RESULTS FOR AUTHOR\n"
    marker = "## PhaseII-Plain vs PhaseII-Heterogeneous"
    if marker in text:
        # replace from marker to next ## Limitations or end section 7
        pre = text.split(marker)[0].rstrip()
        # keep limitations if present after
        rest = text.split(marker, 1)[1]
        # drop until "## 7. Limitations" or "## Limitations" if present
        for cut in ("## 7. Limitations", "## Limitations", "## 8. QC"):
            if cut in rest:
                rest = cut + rest.split(cut, 1)[1]
                text = pre + "\n" + "\n".join(plain_lines) + "\n" + rest
                break
        else:
            text = pre + "\n" + "\n".join(plain_lines)
    else:
        # insert before Limitations
        if "## 7. Limitations" in text:
            a, b = text.split("## 7. Limitations", 1)
            text = a.rstrip() + "\n" + "\n".join(plain_lines) + "\n## 7. Limitations" + b
        else:
            text = text.rstrip() + "\n" + "\n".join(plain_lines) + "\n"
    auth.write_text(text, encoding="utf-8")

    # Update checklist
    checklist_path = OUT / "FINAL_REVIEWER_EVIDENCE_CHECKLIST.csv"
    rows_c = list(csv.DictReader(checklist_path.open(encoding="utf-8"))) if checklist_path.is_file() else []
    # normalize keys
    def upsert(request_id, request, status, evidence):
        for r in rows_c:
            if r.get("request_id") == request_id or r.get("request") == request:
                r["request_id"] = request_id
                r["request"] = request
                r["status"] = status
                r["evidence"] = evidence
                return
        rows_c.append({"request_id": request_id, "request": request, "status": status, "evidence": evidence})

    n_valid = sum(1 for r in rows if r["report_external_valid"])
    upsert("R2.BKS", "Verified numerical Schneider BKS gaps for primary panel", "PARTIAL",
           f"Scope B: {n_valid}/56 externally valid FR seed0; primary 336 unchanged; SCHNEIDER_SCOPE_B_ANALYSIS.md")
    upsert("R2.1", "Schneider coverage + unsuccessful instances", "SATISFIED",
           "Primary 336/264/72 + Scope B 56 FR seed0 with routes; hard C65/C64/C75")
    upsert("R1.5b", "PhaseII-Plain vs Heterogeneous five-seed (unfavorable)", "SATISFIED",
           "T_PhaseII_Plain_vs_Heterogeneous_paired.csv; Plain lower distance on 60/60 per TW")
    # ensure fieldnames
    keys = ["request_id", "request", "status", "evidence"]
    with checklist_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows_c:
            w.writerow({k: r.get(k, "") for k in keys})

    # Update Schneider comparability
    (OUT / "SCHNEIDER_BENCHMARK_COMPARABILITY.md").write_text(
        (SCOPE / "ANALYSIS.md").read_text(encoding="utf-8")
        + "\n## Relation to primary 336\n"
        "Primary protocol **336/264/72** remains the official coverage table. Scope B is an **additive** "
        "route-archive + external-verifier study on FR seed 0 only.\n",
        encoding="utf-8",
    )
    write_json(TABLES / "Schneider_BKS_status.json", {
        "bks_comparison": "PARTIAL",
        "scope_b": {
            "n": len(rows),
            "n_report_external_valid": n_valid,
            "table": "FINAL_TABLES/Schneider_ScopeB_56_FR_seed0.csv",
            "analysis": "SCHNEIDER_SCOPE_B_ANALYSIS.md",
        },
        "primary_protocol_unchanged": {"n_attempts": 336, "n_feasible": 264, "n_unsuccessful": 72},
        "numerical_bks_gap_claimed_for_primary336": False,
    })


def main() -> int:
    SCOPE.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    SOL.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)

    names = list(LARGE_56)
    assert len(names) == 56, len(names)
    primary_idx = load_primary_index()
    assert len(primary_idx) == 56, len(primary_idx)

    # Resume: skip existing complete checkpoints
    payloads = []
    todo = []
    for name in names:
        p = CKPT / f"ALNS-FR__{name}__seed{SEED}__30s_routes.json"
        if p.is_file():
            payloads.append(json.loads(p.read_text(encoding="utf-8")))
        else:
            todo.append(name)
    print(f"Scope B: {len(payloads)} cached, {len(todo)} to run", flush=True)

    workers = 4
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_worker, n): n for n in todo}
            for fut in as_completed(futs):
                name, payload, err = fut.result()
                if err or payload is None:
                    print("FAIL", name, err, flush=True)
                    raise RuntimeError(f"Scope B failed on {name}: {err}")
                path = CKPT / f"ALNS-FR__{name}__seed{SEED}__30s_routes.json"
                path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                payloads.append(payload)
                print(f"done {name} feas={payload['feasible']} k={payload['k']} d={payload['distance']:.3f}", flush=True)

    payloads.sort(key=lambda d: d["instance"])
    print("Verifying + comparing...", flush=True)
    rows = analyze_and_verify(payloads, primary_idx)
    _write_csv(TABLES / "Schneider_ScopeB_56_FR_seed0.csv", rows)
    _write_csv(SCOPE / "results_56.csv", rows)

    print("Plain vs Het...", flush=True)
    plain_info = plain_vs_het()
    write_analysis(rows, plain_info)
    update_author_package(rows, plain_info)

    # QC
    r = subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=line"], cwd=str(ROOT), capture_output=True, text=True)
    (LOGS / "pytest_scope_b.txt").write_text(r.stdout + "\n" + r.stderr, encoding="utf-8")
    r2 = subprocess.run([sys.executable, "scripts/verify_no_algo_diff.py"], cwd=str(ROOT), capture_output=True, text=True)
    (LOGS / "verify_no_algo_diff.txt").write_text(r2.stdout + "\n" + r2.stderr, encoding="utf-8")
    write_json(LOGS / "scope_b_qc.json", {
        "pytest_ok": r.returncode == 0,
        "pytest_tail": (r.stdout.strip().splitlines()[-1:] or [""])[0],
        "algo_diff_ok": r2.returncode == 0,
        "n_rows": len(rows),
    })
    print(json.dumps(json.loads((SCOPE / "summary.json").read_text(encoding="utf-8")), indent=2))
    return 0 if r.returncode == 0 and r2.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
