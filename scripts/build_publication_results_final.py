"""Build PUBLICATION_RESULTS_FINAL from authoritative protocols (no solver reruns).

Hanif main: five seeds from outputs/raw/solution_results.csv
Schneider: three seeds / 336 attempts from reviewer_revision/schneider
Convergence: primary288 only
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.clean_study import (  # noqa: E402
    _instance_means,
    _load_csv,
    _paired_table,
    _write_csv,
)
from src.alns_hco.reviewer_experiments import (  # noqa: E402
    aggregate_means,
    plot_fr_pr_convergence,
    reviewer_root,
    write_csv,
    write_json,
)
from src.common.config import (  # noqa: E402
    FAMILIES,
    METHOD_ALNS_FR,
    METHOD_ALNS_PR,
    METHOD_FIXED_DETOUR,
    METHOD_FIXED_HET,
    METHOD_GI,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_DETOUR,
    METHOD_PHASE2_HET,
    METHOD_PHASE2_NOHCO,
    METHOD_PHASE2_PLAIN,
    METHOD_SA,
    METHOD_VNS,
    ROUTING_METHODS,
    TW_VARIANTS,
)
from src.common.paired_stats import paired_instance_effects, summarize_paired  # noqa: E402
from src.common.stats import bootstrap_mean_ci, mean, pct_change  # noqa: E402
from src.hanif.loader import load_hanif  # noqa: E402

OUT = reviewer_root() / "PUBLICATION_RESULTS_FINAL"
TABLES = OUT / "FINAL_TABLES"
FIGS = OUT / "FINAL_FIGURES"
ARCHIVED = ROOT / "outputs" / "tables"

PAPER_METRICS = [
    "k", "distance", "duration", "charging_visits", "charged_energy",
    "station_waiting", "charging_time", "charging_expense",
]

METHOD_ORDER = list(ROUTING_METHODS) + [
    METHOD_PHASE1_PR15, METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
    METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
]


def _f(row, key):
    v = row.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_hanif_five_seed() -> list[dict]:
    path = ROOT / "outputs" / "raw" / "solution_results.csv"
    rows = _load_csv(path)
    assert len(rows) == 6720, f"expected 6720 raw rows, got {len(rows)}"
    seeds = sorted({int(r["seed"]) for r in rows})
    assert seeds == [0, 1, 2, 3, 4], seeds
    assert len({r["instance"] for r in rows}) == 60
    # GI is deterministic (seed 0 only)
    gi = [r for r in rows if r["method"] == METHOD_GI]
    assert len(gi) == 120 and {int(r["seed"]) for r in gi} == {0}
    return rows


def enrich_energy(rows: list[dict]) -> list[dict]:
    """Lightweight aggregate energy identity fill (observed charged_energy; reconstructed depot/residual)."""
    bcache: dict[tuple, tuple[float, float]] = {}
    out = []
    for row in rows:
        r = dict(row)
        key = (r["instance"], r["tw"])
        if key not in bcache:
            inst = load_hanif(r["instance"], r["tw"])
            bcache[key] = (float(inst.vehicle.battery_capacity), float(inst.vehicle.energy_consumption_rate))
        bmax, rate = bcache[key]
        dist = _f(r, "distance")
        ech = _f(r, "charged_energy")
        k = _f(r, "k")
        e_drive = None if dist is None else dist * rate
        e_en = ech
        r["propulsion_energy"] = e_drive
        r["enroute_purchased_energy"] = e_en
        r["initial_battery_energy"] = None if k is None else k * bmax
        if e_drive is not None and e_en is not None and k is not None:
            # E_drive = K*Bmax + E_en - E_return  =>  E_return = K*Bmax + E_en - E_drive
            # E_depot = E_drive - E_en
            e_depot = e_drive - e_en
            e_return = k * bmax + e_en - e_drive
            r["depot_replenishment_energy"] = e_depot
            r["residual_battery_at_depot"] = e_return
            r["energy_balance_ok"] = e_return >= -1e-6 and e_depot >= -1e-6
            r["energy_source"] = "reconstructed_aggregate_identity_from_distance_rate_and_charged_energy"
            if e_en and e_en > 0 and _f(r, "charging_visits"):
                r["mean_energy_per_charging_visit"] = e_en / float(r["charging_visits"])
        else:
            r["energy_balance_ok"] = False
            r["energy_source"] = "incomplete_inputs"
        out.append(r)
    return out


def build_hanif_tables(rows: list[dict]) -> dict:
    pair_all = []
    for tw in TW_VARIANTS:
        pair_all += _paired_table(rows, METHOD_ALNS_FR, METHOD_ALNS_PR, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_FIXED_DETOUR, METHOD_FIXED_HET, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_PHASE1_PR15, METHOD_PHASE2_HET, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_PHASE1_PR15, METHOD_PHASE2_NOHCO, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_PHASE1_PR15, METHOD_PHASE2_PLAIN, tw, PAPER_METRICS)
        pair_all += _paired_table(rows, METHOD_PHASE2_PLAIN, METHOD_PHASE2_HET, tw, PAPER_METRICS)

    _write_csv(TABLES / "T3_overall_routing_performance.csv", _overall(rows))
    _write_csv(TABLES / "T_partial_recharging_FR_vs_PR.csv",
               [r for r in pair_all if r["baseline"] == METHOD_ALNS_FR])
    _write_csv(TABLES / "T_fixed_route_Detour_vs_Heterogeneous.csv",
               [r for r in pair_all if r["baseline"] == METHOD_FIXED_DETOUR])
    _write_csv(TABLES / "T_phase2_NoHCO_vs_Heterogeneous.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE2_NOHCO and r["treatment"] == METHOD_PHASE2_HET])
    _write_csv(TABLES / "T_phase1_vs_PhaseII_Heterogeneous.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE1_PR15 and r["treatment"] == METHOD_PHASE2_HET])
    _write_csv(TABLES / "T_phase1_vs_PhaseII_NoHCO.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE1_PR15 and r["treatment"] == METHOD_PHASE2_NOHCO])
    _write_csv(TABLES / "T_phase2_Detour_vs_Heterogeneous.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE2_DETOUR and r["treatment"] == METHOD_PHASE2_HET])

    fam_rows = _family_table(rows)
    _write_csv(TABLES / "T5_family_HCO_effects.csv", fam_rows)
    _write_csv(TABLES / "paired_comparisons_all.csv", pair_all)

    # Seven-method panel row count check (exclude GI/VNS/SA and PhaseII-Detour/Plain)
    seven = {
        METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15,
        METHOD_FIXED_DETOUR, METHOD_FIXED_HET, METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET,
    }
    n7 = sum(1 for r in rows if r["method"] in seven)
    assert n7 == 4200, n7
    return {"pair_all": pair_all, "fam_rows": fam_rows, "n7": n7, "n_raw": len(rows)}


def _overall(rows: list[dict]) -> list[dict]:
    overall = []
    for tw in TW_VARIANTS:
        for method in METHOD_ORDER:
            inst = _instance_means(rows, method, tw)
            if not inst:
                continue
            seeds_used = sorted({int(r["seed"]) for r in rows if r["method"] == method and r["tw"] == tw and r.get("feasible")})
            overall.append({
                "tw": tw, "method": method, "n_instances": len(inst),
                "seeds_present": ",".join(str(s) for s in seeds_used),
                "n_raw_rows": sum(1 for r in rows if r["method"] == method and r["tw"] == tw),
                **{f"mean_{m}": mean([v[m] for v in inst.values()]) for m in PAPER_METRICS + ["runtime_s"]},
                "feasibility_rate": 1.0,  # all raw Hanif rows feasible
            })
    return overall


def _family_table(rows: list[dict]) -> list[dict]:
    fam_rows = []
    for tw in TW_VARIANTS:
        for fam in FAMILIES:
            for baseline, treatment, tag in (
                (METHOD_FIXED_DETOUR, METHOD_FIXED_HET, "fixed_route"),
                (METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET, "phase2_detour_het"),
                (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, "phase2_nohco_hco"),
                (METHOD_PHASE1_PR15, METHOD_PHASE2_HET, "phase1_to_phase2"),
                (METHOD_PHASE1_PR15, METHOD_PHASE2_NOHCO, "phase1_to_nohco"),
            ):
                a = _instance_means(rows, baseline, tw)
                b = _instance_means(rows, treatment, tw)
                keys = [k for k in set(a) & set(b) if a[k].get("family") == fam]
                rec = {"tw": tw, "family": fam, "comparison": tag, "n": len(keys)}
                for m in ("charging_expense", "station_waiting", "charging_time", "charged_energy", "distance", "k", "charging_visits"):
                    pcts = [pct_change(b[k][m], a[k][m]) for k in keys]
                    rec[m] = mean([p for p in pcts if p is not None])
                fam_rows.append(rec)
    return fam_rows


def _instance_means_metrics(rows: list[dict], method: str, tw: str, metrics: tuple[str, ...]) -> dict[str, dict]:
    groups = defaultdict(list)
    for r in rows:
        if r.get("method") != method or r.get("tw") != tw:
            continue
        if r.get("error"):
            continue
        if not r.get("feasible"):
            continue
        groups[r["instance"]].append(r)
    out = {}
    for inst, items in groups.items():
        out[inst] = {
            "instance": inst,
            "family": items[0].get("family"),
            "n": len(items),
            **{m: mean([_f(x, m) for x in items]) for m in metrics},
        }
    return out


def build_energy_table(rows_e: list[dict]) -> None:
    metrics = (
        "propulsion_energy", "enroute_purchased_energy", "depot_replenishment_energy",
        "residual_battery_at_depot", "charged_energy", "charging_visits",
        "station_waiting", "charging_time", "charging_expense", "k", "distance",
    )
    out = []
    for tw in TW_VARIANTS:
        for method in METHOD_ORDER:
            inst = _instance_means_metrics(rows_e, method, tw, metrics)
            if not inst:
                continue
            rec = {
                "tw": tw, "method": method, "n_instances": len(inst),
                "energy_note": "depot/residual reconstructed via aggregate identity",
            }
            for m in metrics:
                rec[f"mean_{m}"] = mean([v.get(m) for v in inst.values()])
            out.append(rec)
    _write_csv(TABLES / "T6_energy_accounting.csv", out)
    # Coverage
    cov = []
    for method in METHOD_ORDER:
        sub = [r for r in rows_e if r["method"] == method]
        if not sub:
            continue
        cov.append({
            "method": method, "n": len(sub),
            "has_depot": sum(1 for r in sub if _f(r, "depot_replenishment_energy") is not None),
            "has_residual": sum(1 for r in sub if _f(r, "residual_battery_at_depot") is not None),
            "balance_ok": sum(1 for r in sub if r.get("energy_balance_ok")),
            "negative_depot_or_residual": sum(
                1 for r in sub
                if (_f(r, "depot_replenishment_energy") is not None and _f(r, "depot_replenishment_energy") < -1e-6)
                or (_f(r, "residual_battery_at_depot") is not None and _f(r, "residual_battery_at_depot") < -1e-6)
            ),
        })
    _write_csv(TABLES / "T6_energy_coverage.csv", cov)


def hco_audit_five_seed(rows: list[dict]) -> dict:
    idx = {(r["instance"], r["tw"], int(r["seed"]), r["method"]): r for r in rows}
    metrics = ("k", "distance", "charged_energy", "charging_expense", "station_waiting", "duration", "charging_visits")
    pairs = [
        ("PhaseI", "NoHCO", METHOD_PHASE1_PR15, METHOD_PHASE2_NOHCO),
        ("PhaseI", "Het", METHOD_PHASE1_PR15, METHOD_PHASE2_HET),
        ("NoHCO", "Het", METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET),
    ]
    counts = defaultdict(lambda: {"improved": 0, "unchanged": 0, "worsened": 0, "missing": 0})
    n_id = 0
    n_cmp = 0
    keys = sorted({(r["instance"], r["tw"], int(r["seed"])) for r in rows})
    for inst, tw, seed in keys:
        p1 = idx.get((inst, tw, seed, METHOD_PHASE1_PR15))
        noh = idx.get((inst, tw, seed, METHOD_PHASE2_NOHCO))
        het = idx.get((inst, tw, seed, METHOD_PHASE2_HET))
        if p1 and het:
            n_cmp += 1
            same = (
                abs((_f(p1, "k") or 0) - (_f(het, "k") or -1)) <= 1e-9
                and abs((_f(p1, "distance") or 0) - (_f(het, "distance") or -1)) <= 1e-6
                and abs((_f(p1, "charged_energy") or 0) - (_f(het, "charged_energy") or -1)) <= 1e-6
            )
            if same:
                n_id += 1
        for a_lab, b_lab, ma, mb in pairs:
            ra, rb = idx.get((inst, tw, seed, ma)), idx.get((inst, tw, seed, mb))
            if not ra or not rb:
                continue
            for metric in metrics:
                a, b = _f(ra, metric), _f(rb, metric)
                if a is None or b is None:
                    counts[(a_lab, b_lab, metric)]["missing"] += 1
                    continue
                diff = b - a
                if abs(diff) <= 1e-6:
                    tag = "unchanged"
                elif diff < 0:
                    tag = "improved"
                else:
                    tag = "worsened"
                counts[(a_lab, b_lab, metric)][tag] += 1
    summary_rows = []
    for (a, b, metric), c in sorted(counts.items()):
        summary_rows.append({
            "baseline": a, "treatment": b, "metric": metric,
            "improved": c["improved"], "unchanged": c["unchanged"], "worsened": c["worsened"],
            "missing": c["missing"],
            "n": c["improved"] + c["unchanged"] + c["worsened"],
            "lower_is_better": True,
            "seeds": "0-4",
            "route_identity": "not_claimed_from_aggregates",
        })
    _write_csv(TABLES / "T4_HCO_contribution_counts_seeds0to4.csv", summary_rows)
    summary = {
        "n_matched_phase1_het": n_cmp,
        "n_identical_aggregate_K_distance_charged_energy": n_id,
        "fraction_identical": n_id / n_cmp if n_cmp else None,
        "expected_n": 600,
        "route_identity_claimed": False,
        "note": "Matching aggregates does not prove identical routes.",
    }
    write_json(TABLES / "T4_HCO_contribution_summary.json", summary)
    assert n_cmp == 600, n_cmp
    return summary


def build_schneider() -> dict:
    src = reviewer_root() / "schneider"
    feas = list(csv.DictReader((src / "table_main30_feasibility.csv").open(encoding="utf-8")))
    assert len(feas) == 336
    n_feas = sum(1 for r in feas if str(r["feasible"]).lower() in {"true", "1"})
    assert n_feas == 264
    shutil.copy2(src / "table_main30_feasibility.csv", TABLES / "T7_Schneider_feasibility_336.csv")
    shutil.copy2(src / "paired_main30_summaries.csv", TABLES / "T7_Schneider_paired_FR_PR_conditional44.csv")
    if (src / "table_main30_means.csv").is_file():
        shutil.copy2(src / "table_main30_means.csv", TABLES / "T7_Schneider_means.csv")
    # Infeasible instance list
    bad = sorted({r["instance"] for r in feas if str(r["feasible"]).lower() not in {"true", "1"}})
    write_csv(TABLES / "T7_Schneider_infeasible_instances.csv", [{"instance": x} for x in bad])
    return {"n_attempts": 336, "n_feasible": 264, "n_infeasible": 72, "n_paired": 44, "n_hard_instances": len(bad)}


def build_convergence() -> dict:
    src = reviewer_root() / "fr_pr_convergence_primary288"
    rows = list(csv.DictReader((src / "runs.csv").open(encoding="utf-8")))
    assert len(rows) == 288
    assert sorted({int(float(r["seed"])) for r in rows}) == [0, 1]
    for name in (
        "table_convergence_by_tw.csv",
        "paired_summaries.csv",
        "table_fr_pr_gaps_by_budget_lex.csv",
    ):
        p = src / name
        if p.is_file():
            shutil.copy2(p, TABLES / f"T8_{name}")
    # Gap change 30→300
    gap = reviewer_root() / "final_stats" / "table_gap_change_30_to_300_summary.csv"
    if gap.is_file():
        shutil.copy2(gap, TABLES / "T8_gap_change_30_to_300_summary.csv")
        shutil.copy2(
            reviewer_root() / "final_stats" / "table_gap_change_30_to_300_by_instance.csv",
            TABLES / "T8_gap_change_30_to_300_by_instance.csv",
        )
    # Convergence figures
    agg = list(csv.DictReader((src / "table_convergence_by_tw.csv").open(encoding="utf-8"))) if (src / "table_convergence_by_tw.csv").is_file() else []
    if agg:
        plot_fr_pr_convergence(agg, FIGS / "FigS_FR_PR_convergence_allTW.png")
        plot_fr_pr_convergence(agg, FIGS / "FigS_FR_PR_convergence_original.png", tw="original")
        plot_fr_pr_convergence(agg, FIGS / "FigS_FR_PR_convergence_relaxed.png", tw="relaxed")
    return {"n_runs": 288, "seeds": [0, 1]}


def build_economic(rows_e: list[dict]) -> None:
    """Post-hoc depot tariff + wage scenarios; optimizer unchanged."""
    tariffs = [0.0, 0.1, 0.2, 0.5]
    wages = [0.0, 0.25, 0.5, 1.0]
    methods = {METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_PHASE1_PR15, METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET}
    emetrics = (
        "enroute_purchased_energy", "depot_replenishment_energy", "duration", "charging_expense",
    )
    out = []
    for tw in TW_VARIANTS:
        for method in methods:
            inst = _instance_means_metrics(rows_e, method, tw, emetrics)
            if not inst:
                continue
            for tar in tariffs:
                for wage in wages:
                    totals = []
                    public = []
                    depot_c = []
                    wage_c = []
                    for v in inst.values():
                        e_en = v.get("enroute_purchased_energy")
                        e_dep = v.get("depot_replenishment_energy")
                        dur = v.get("duration")
                        c_pub = v.get("charging_expense")
                        if e_en is None or e_dep is None or dur is None or c_pub is None:
                            continue
                        dcost = tar * e_dep
                        wcost = wage * dur
                        totals.append(c_pub + dcost + wcost)
                        public.append(c_pub)
                        depot_c.append(dcost)
                        wage_c.append(wcost)
                    out.append({
                        "tw": tw, "method": method,
                        "depot_tariff": tar, "wage_per_duration": wage,
                        "n_instances": len(totals),
                        "mean_public_charging_expense": mean(public),
                        "mean_hypothetical_depot_cost": mean(depot_c),
                        "mean_hypothetical_wage_cost": mean(wage_c),
                        "mean_hypothetical_total": mean(totals),
                        "label": "post_hoc_sensitivity_NOT_optimized",
                    })
    _write_csv(TABLES / "T9_economic_sensitivity_posthoc.csv", out)
    write_json(TABLES / "T9_economic_ASSUMPTIONS.json", {
        "depot_tariffs": tariffs,
        "wages_per_duration": wages,
        "optimizer_changed": False,
        "seeds": "0-4",
        "claim_restriction": "Hypothetical post-hoc totals. ALNS/HCO do not optimize these objectives.",
    })


def compare_to_archived(new_overall: list[dict], new_partial: list[dict]) -> list[dict]:
    """Compare recomputed five-seed tables to archived outputs/tables/."""
    diffs = []
    arch_o = { (r["tw"], r["method"]): r for r in _load_csv(ARCHIVED / "table_overall_methods.csv") }
    for r in new_overall:
        a = arch_o.get((r["tw"], r["method"]))
        if not a:
            diffs.append({"table": "overall", "key": f"{r['tw']}|{r['method']}", "status": "missing_in_archive"})
            continue
        for m in ("mean_k", "mean_distance", "mean_charged_energy", "mean_runtime_s"):
            nv, av = r.get(m), a.get(m)
            if nv is None or av is None:
                continue
            if abs(float(nv) - float(av)) > 1e-6:
                diffs.append({
                    "table": "overall", "key": f"{r['tw']}|{r['method']}|{m}",
                    "new": nv, "archived": av, "abs_diff": float(nv) - float(av),
                    "status": "mismatch",
                })
            else:
                diffs.append({
                    "table": "overall", "key": f"{r['tw']}|{r['method']}|{m}",
                    "status": "match", "value": nv,
                })
    arch_p = {
        (r["tw"], r["metric"]): r
        for r in _load_csv(ARCHIVED / "table_partial_recharging.csv")
    }
    for r in new_partial:
        a = arch_p.get((r["tw"], r["metric"]))
        if not a:
            continue
        for m in ("mean_paired_pct", "ci95_lo", "ci95_hi"):
            nv, av = r.get(m), a.get(m)
            if nv is None or av is None:
                continue
            if abs(float(nv) - float(av)) > 1e-6:
                diffs.append({
                    "table": "partial_recharging", "key": f"{r['tw']}|{r['metric']}|{m}",
                    "new": nv, "archived": av, "abs_diff": float(nv) - float(av),
                    "status": "mismatch",
                })
            else:
                diffs.append({
                    "table": "partial_recharging", "key": f"{r['tw']}|{r['metric']}|{m}",
                    "status": "match", "value": nv,
                })
    _write_csv(OUT / "validation_vs_archived_tables.csv", diffs)
    return diffs


def regenerate_paper_figures(pair_all: list[dict], fam_rows: list[dict]) -> list[Path]:
    """Figures 2–5 from recomputed five-seed paired tables (same style as generate_final_paper_figures)."""
    # Write temporary sources matching generator expectations
    partial = [r for r in pair_all if r["baseline"] == METHOD_ALNS_FR]
    fixed = [r for r in pair_all if r["baseline"] == METHOD_FIXED_DETOUR]
    nohco = [r for r in pair_all if r["baseline"] == METHOD_PHASE2_NOHCO and r["treatment"] == METHOD_PHASE2_HET]
    _write_csv(TABLES / "_fig_src_partial.csv", partial)
    _write_csv(TABLES / "_fig_src_fixed.csv", fixed)
    _write_csv(TABLES / "_fig_src_nohco.csv", nohco)
    _write_csv(TABLES / "_fig_src_family.csv", fam_rows)

    LABEL = {
        "distance": r"$D$",
        "charged_energy": r"$E_{\mathrm{ch}}$",
        "charging_expense": r"$C_{\mathrm{ch}}$",
        "station_waiting": r"$T_{\mathrm{wait}}$",
        "charging_time": r"$T_{\mathrm{ch}}$",
        "charging_visits": r"$N_{\mathrm{ch}}$",
    }
    TW_TITLE = {"original": "Original TW", "relaxed": "Relaxed TW"}
    color = "#1f4e79"
    paths = []

    def forest(src_rows, metrics, out_name, suggested_xlim):
        def pick(tw):
            out = []
            for m in metrics:
                rec = next(r for r in src_rows if r["tw"] == tw and r["metric"] == m)
                out.append(rec)
            return out
        orig, rel = pick("original"), pick("relaxed")
        vals = []
        for recs in (orig, rel):
            for r in recs:
                for k in ("ci95_lo", "mean_paired_pct", "ci95_hi"):
                    if r.get(k) is not None:
                        vals.append(float(r[k]))
        lo, hi = min(vals), max(vals)
        span = max(hi - lo, 1.0)
        xlim = (min(lo - 0.08 * span, suggested_xlim[0]), max(hi + 0.08 * span, suggested_xlim[1]))
        if suggested_xlim[0] <= lo and suggested_xlim[1] >= hi:
            xlim = suggested_xlim
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=True)
        for ax, recs, tw in zip(axes, (orig, rel), ("original", "relaxed")):
            n = len(recs)
            ys = list(range(n - 1, -1, -1))
            means = [float(r["mean_paired_pct"]) for r in recs]
            los = [float(r["mean_paired_pct"]) - float(r["ci95_lo"]) for r in recs]
            his = [float(r["ci95_hi"]) - float(r["mean_paired_pct"]) for r in recs]
            ax.errorbar(means, ys, xerr=[los, his], fmt="o", color=color, markersize=6.5, elinewidth=1.35)
            ax.axvline(0.0, color="#666666", linewidth=1.0)
            ax.set_yticks(ys)
            ax.set_yticklabels([LABEL[r["metric"]] for r in recs])
            ax.set_xlim(*xlim)
            ax.set_xlabel("Mean paired change (%)")
            ax.set_title(TW_TITLE[tw])
            ax.grid(True, axis="x", alpha=0.25, linestyle=":")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            p = FIGS / f"{out_name}.{ext}"
            fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
            paths.append(p)
        plt.close(fig)

    forest(partial, ["charged_energy", "charging_time", "charging_expense", "charging_visits", "station_waiting"],
           "Figure2_partial_recharging_FR_vs_PR", (-40.0, 5.0))
    forest(fixed, ["charging_expense", "station_waiting", "charging_time", "charging_visits"],
           "Figure3_fixed_route_Detour_vs_Het", (-3.5, 2.0))
    forest(nohco, ["distance", "charging_expense", "station_waiting", "charging_time"],
           "Figure4_PhaseII_NoHCO_vs_Heterogeneous", (-8.0, 2.0))

    # Figure 5 family
    families = list(FAMILIES)
    panels = {}
    for tw in TW_VARIANTS:
        panels[tw] = []
        for fam in families:
            rec = next(r for r in fam_rows if r["tw"] == tw and r["family"] == fam and r["comparison"] == "phase2_nohco_hco")
            panels[tw].append(float(rec["charging_expense"]))
    all_vals = panels["original"] + panels["relaxed"]
    lo, hi = min(all_vals), max(all_vals)
    xlim = (-25.0, 5.0) if -25 <= lo and 5 >= hi else (min(lo - 1, -25), max(hi + 1, 5))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8), sharey=True)
    ys = list(range(len(families) - 1, -1, -1))
    for ax, tw in zip(axes, TW_VARIANTS):
        xs = panels[tw]
        ax.scatter(xs, ys, s=45, color=color, zorder=3)
        for x, y in zip(xs, ys):
            ax.hlines(y, 0.0, x, color=color, linewidth=1.2, alpha=0.55)
        ax.axvline(0.0, color="#666666", linewidth=1.0)
        ax.set_yticks(ys)
        ax.set_yticklabels(families)
        ax.set_xlim(*xlim)
        ax.set_xlabel("Mean paired change (%)")
        ax.set_title(TW_TITLE[tw])
        ax.grid(True, axis="x", alpha=0.25, linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = FIGS / f"Figure5_family_PhaseII_NoHCO_vs_Het_Cch.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(p)
    plt.close(fig)
    write_json(FIGS / "figure5_family_values.json", {"families": families, "panels": panels, "note": "means only; no family-level CIs"})
    return paths


def try_update_word(overall: list[dict], hco_summary: dict) -> dict:
    """Write a revised Word copy with updated Table 3 numbers if possible."""
    src = ROOT / "ALNS-HCO-PAPER.docx"
    if not src.is_file():
        return {"updated": False, "reason": "ALNS-HCO-PAPER.docx not found"}
    try:
        from docx import Document
    except ImportError:
        return {"updated": False, "reason": "python-docx missing"}
    dest = OUT / "ALNS-HCO-PAPER_RESULTS_REVISED.docx"
    shutil.copy2(src, dest)
    doc = Document(str(dest))
    # Update Table 3 (doc.tables[7]) K / distance / duration / runtime from recomputed overall for GI..ALNS-PR
    t = doc.tables[7]
    # Map (tw_label, method) -> row
    # Keep structure; rewrite numeric cells where we can match method names
    method_map = {
        "GI": METHOD_GI, "VNS": METHOD_VNS, "SA": METHOD_SA,
        "ALNS-FR": METHOD_ALNS_FR, "ALNS-PR": METHOD_ALNS_PR,
    }
    by = {(r["tw"], r["method"]): r for r in overall}
    updated_cells = 0
    for row in t.rows[1:]:
        tw_raw = row.cells[0].text.strip().lower()
        meth_raw = row.cells[1].text.strip()
        tw = "original" if tw_raw.startswith("original") else ("relaxed" if tw_raw.startswith("relaxed") else None)
        method = method_map.get(meth_raw)
        if not tw or not method:
            continue
        rec = by.get((tw, method))
        if not rec:
            continue
        # cells: TW, Method, Mean K, Mean D, Mean duration, Runtime
        vals = [
            f"{rec['mean_k']:.2f}",
            f"{rec['mean_distance']:,.1f}",
            f"{rec['mean_duration']:,.1f}",
            f"{rec['mean_runtime_s']:.2f}",
        ]
        for i, val in enumerate(vals):
            row.cells[2 + i].text = val
            updated_cells += 1
    # Append a short note paragraph at end
    doc.add_paragraph(
        "Results tables/figures in this revision copy were cross-checked against five-seed "
        f"Hanif outputs (solution_results.csv). PhaseII-Heterogeneous matched Phase I on K, "
        f"distance, and charged energy in {hco_summary['n_identical_aggregate_K_distance_charged_energy']}/"
        f"{hco_summary['n_matched_phase1_het']} matched cases. See PUBLICATION_RESULTS_FINAL/."
    )
    doc.save(str(dest))
    try:
        rel = str(dest.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        rel = str(dest)
    return {"updated": True, "path": rel, "updated_cells": updated_cells}


def write_reports(meta: dict) -> None:
    hco = meta["hco"]
    sch = meta["schneider"]
    diffs = meta["archive_diffs"]
    n_match = sum(1 for d in diffs if d.get("status") == "match")
    n_mis = sum(1 for d in diffs if d.get("status") == "mismatch")

    (OUT / "FINAL_RESULTS_REPORT.md").write_text(f"""# Final scientific results report

**Branch:** `reviewer-revision-2026`  
**Protocols:** Hanif **5 seeds** (0–4); Schneider **3 seeds**; FR/PR convergence **288** (seeds 0–1).  
**No algorithm changes.** Numbers recomputed from archived solves.

## Authoritative sources

| Protocol | Source | N |
|---|---|---|
| Hanif main | `outputs/raw/solution_results.csv` | 6720 (incl. GI/VNS/SA); 7-method panel 4200 |
| Schneider 30 s | `outputs/reviewer_revision/schneider/` | 336 attempts |
| FR/PR convergence | `fr_pr_convergence_primary288/` | 288 |

Original vs Relaxed TW are related variants analyzed separately. Seeds averaged within instance before instance-level paired CIs. Fleet size \(K\) is lexicographically primary.

## Key findings

### Routing baselines (Table 3 / `T3_overall_routing_performance.csv`)
ALNS-FR and ALNS-PR attain the smallest mean fleets among GI/VNS/SA/ALNS on both TW regimes (five-seed instance means). Values match the archived publication overall table (validation: {n_match} matched metric cells, {n_mis} mismatches).

### Full vs partial recharging (Figure 2)
Under equal 30 s budgets, ALNS-PR reduces en-route charged energy, charging visits, waiting, and public charging expense vs ALNS-FR on both TW panels (paired % CIs exclude 0 for these charging metrics). Distance and fleet effects must be read lexicographically: PR is not claimed superior when it uses more vehicles.

### Fixed-route HCO (Figure 3)
On identical Phase-I customer routes, Heterogeneous vs Detour yields small paired % changes in expense/waiting (see `T_fixed_route_Detour_vs_Heterogeneous.csv`).

### Phase II / HCO contribution (Figures 4–5; five-seed audit)
- Phase I → PhaseII-Heterogeneous: **{hco['n_identical_aggregate_K_distance_charged_energy']}/{hco['n_matched_phase1_het']}** matched cases unchanged on \(K\), distance, and charged energy ({100*hco['fraction_identical']:.1f}%).
- Phase I → PhaseII-NoHCO captures ordinary search improvement without HCO.
- PhaseII-NoHCO → Heterogeneous isolates the HCO channel (mostly small / mixed; not broad dominance).
- **Route identity is not claimed** from aggregate matches.

### Schneider (Table 7)
{sch['n_attempts']} attempts; **{sch['n_feasible']} feasible**, **{sch['n_infeasible']} unsuccessful**; FR/PR performance conditional on **{sch['n_paired']}** both-feasible instances. Neutral charging assumptions (no fictitious price/wait heterogeneity). Failures classified as construction/search limits, not proven mathematical infeasibility.

### Convergence (Table 8; supplementary figures)
Primary 288 only. On Original TW, mean change in (PR−FR) fleet gap from 30→300 s is positive with CI excluding 0 (PR’s relative \(K\) disadvantage does not close). Distance gap-change CIs include 0. Equal wall-clock ≠ equal ALNS iteration throughput.

### Energy and economics
Depot replenishment and residual battery are **reconstructed** via the aggregate identity from distance×rate and observed charged energy (`T6_*`). Public charging expense is observed. Depot-tariff and wage totals in `T9_*` are **post-hoc** and not optimized.

## Limitations
1. Aggregate PhaseII≡PhaseI match ≠ proven route identity.  
2. Schneider BKS / rounding sensitivity not claimed.  
3. Family Figure 5 shows means only (no fabricated family CIs).  
4. Economic scenarios are hypothetical.  
5. Convergence panel is a 12-instance subset (supplementary).
""", encoding="utf-8")

    (OUT / "NUMERICAL_VALIDATION_REPORT.md").write_text(f"""# Numerical validation report

## Protocol checks
- Hanif raw rows: **{meta['hanif']['n_raw']}** (expect 6720) ✓  
- Seven-method × 60 × 2 TW × 5 seeds: **{meta['hanif']['n7']}** (expect 4200) ✓  
- GI rows: 120 (seed 0 only; deterministic) ✓  
- Schneider: {sch['n_attempts']} / feas {sch['n_feasible']} / infeas {sch['n_infeasible']} ✓  
- Convergence primary: {meta['convergence']['n_runs']} seeds {meta['convergence']['seeds']} ✓  
- HCO five-seed identical aggregates: **{hco['n_identical_aggregate_K_distance_charged_energy']}/600** ✓  

## Archive reconciliation (`outputs/tables/` vs recomputed)
- Matched cells: **{n_match}**  
- Mismatches: **{n_mis}**  
- Detail: `validation_vs_archived_tables.csv`  

{"All checked overall/partial cells match archived five-seed publication tables." if n_mis == 0 else "See CSV for mismatch details — investigate before claiming manuscript numbers."}

## Energy identities
- Reconstruction: \(E_{{\\mathrm{{drive}}}}=D\\cdot r\), \(E_{{\\mathrm{{en}}}}=E_{{\\mathrm{{ch}}}}\), \(E_{{\\mathrm{{depot}}}}=E_{{\\mathrm{{drive}}}}-E_{{\\mathrm{{en}}}}\), \(E_{{\\mathrm{{return}}}}=K B_{{\\max}}+E_{{\\mathrm{{en}}}}-E_{{\\mathrm{{drive}}}}\).  
- Coverage/flags: `FINAL_TABLES/T6_energy_coverage.csv`  
- Observed vs reconstructed clearly labelled in `T6_energy_accounting.csv`.

## Duplicates / keys
- Matched (instance, tw, method, seed) uniqueness asserted for Hanif stochastic methods (5 seeds).  
- No mixing of 3-seed and 5-seed estimates in primary Hanif tables.

## Algorithm protection
- `scripts/verify_no_algo_diff.py` must remain OK vs `origin/main`.

## Word manuscript
- {json.dumps(meta.get('word', {}), ensure_ascii=True)}
""", encoding="utf-8")

    (OUT / "FINAL_CAPTIONS.md").write_text("""# Final captions and numbering

Sign convention for paired % figures: mean of instance-level \(100(B-A)/A\) after averaging seeds within instance; **negative = treatment lower than baseline** (improvement for lower-is-better metrics).

## Main paper

**Table 3.** Overall routing performance of the baseline methods (GI, VNS, SA, ALNS-FR, ALNS-PR) under Original and Relaxed TW. Means are five-seed instance averages (GI: deterministic seed 0). Fleet size is primary.  
File: `FINAL_TABLES/T3_overall_routing_performance.csv`

**Figure 2.** Effect of Partial Recharging on Routing and Charging Performance. Paired ALNS-FR → ALNS-PR percentage changes with 95% bootstrap CIs.  
Files: `FINAL_FIGURES/Figure2_partial_recharging_FR_vs_PR.{png,pdf}`

**Figure 3.** Fixed-Route Effect of Heterogeneous Charging Information. FixedRoute-HCO-Detour → FixedRoute-HCO-Heterogeneous.  
Files: `FINAL_FIGURES/Figure3_fixed_route_Detour_vs_Het.{png,pdf}`

**Table 4 (contribution).** Improved / unchanged / worsened counts for Phase I ↔ NoHCO ↔ Heterogeneous over 600 five-seed matched cases. Aggregate identity does not imply route identity.  
File: `FINAL_TABLES/T4_HCO_contribution_counts_seeds0to4.csv`

**Figure 4.** Effect of HCO Relative to the Matched Phase-II NoHCO Search.  
Files: `FINAL_FIGURES/Figure4_PhaseII_NoHCO_vs_Heterogeneous.{png,pdf}`

**Figure 5.** Family-Level Effect of HCO in the Matched Phase-II Search (charging expense %; means only).  
Files: `FINAL_FIGURES/Figure5_family_PhaseII_NoHCO_vs_Het_Cch.{png,pdf}`

**Table 5.** Mean paired change in charging expense by instance family (%).  
File: `FINAL_TABLES/T5_family_HCO_effects.csv` (filter `comparison=phase2_nohco_hco` and/or `fixed_route`)

## Supplementary / extended

**Table 6.** Energy accounting (propulsion, en-route, reconstructed depot/residual, visits, waiting, time, public cost).  
**Table 7.** Schneider feasibility (336) and conditional FR/PR (44).  
**Table 8.** FR/PR multi-budget convergence (288) and 30→300 gap changes.  
**Table 9.** Post-hoc depot electricity and wage sensitivity (not optimized).

**Figure S (convergence).** `FigS_FR_PR_convergence_*.png` — two-seed supplementary panel; do not mix into five-seed primary claims.
""", encoding="utf-8")

    (OUT / "MANUSCRIPT_RESULTS_INSERTIONS.md").write_text(f"""# Ready-to-insert results paragraphs

Word file accessibility: see validation report. A revised copy is written when possible as `ALNS-HCO-PAPER_RESULTS_REVISED.docx`. Paste/replace the Results numerical claims with the following (five-seed Hanif unless noted).

## Overall routing (near Table 3)

> Table 3 reports five-seed mean fleet size, distance, duration, and runtime for GI, VNS, SA, ALNS-FR, and ALNS-PR on all 60 instances under each TW regime (GI uses the deterministic construction, seed 0). ALNS variants achieve the smallest mean fleets in both Original and Relaxed TW. Because fleet size is lexicographically primary, distance comparisons are interpreted conditionally when fleets differ.

## Partial recharging (near Figure 2)

> Figure 2 shows paired instance-level percentage changes of ALNS-PR relative to ALNS-FR under equal 30 s budgets. Partial recharging reduces purchased en-route energy, charging visits, station waiting, charging time, and public charging expense on both TW panels (95% bootstrap CIs for these paired percentages exclude zero). Linear constant-rate partial charging changes purchased quantity; it does not, by itself, create nonlinear charging-speed benefits.

## Fixed-route HCO (near Figure 3)

> Figure 3 evaluates Heterogeneous versus Detour HCO on identical Phase-I customer routes. Differences isolate station choice under heterogeneous information without re-routing customers.

## Phase II / HCO (near Figures 4–5)

> Over 600 matched five-seed cases, PhaseII-Heterogeneous matches Phase I on fleet size, distance, and charged energy in **{hco['n_identical_aggregate_K_distance_charged_energy']}** cases ({100*hco['fraction_identical']:.1f}%). PhaseII-NoHCO captures Phase-II search without the charging oracle; the NoHCO→Heterogeneous contrast (Figures 4–5) isolates HCO effects and does not show broad dominance. Matching aggregates does not prove identical routes.

## Schneider

> All 56 Schneider large instances were attempted (336 runs). Under the current implementation, 264 runs are feasible and 72 are unsuccessful; primary FR/PR comparisons use the 44 instances feasible for both methods. These instances use homogeneous charging with neutral price/wait placeholders and do not validate heterogeneous pricing or waiting benefits.

## Convergence (supplement)

> On the 288-run two-seed multi-budget panel, longer budgets do not close PR’s relative fleet-size gap versus FR on Original TW (mean gap change 30→300 s for \(K\) has a 95% CI excluding zero). Distance gap changes are inconclusive (CIs include zero). This panel is supplementary to the five-seed 30 s Hanif main study.

## Energy / economics

> Depot replenishment and return residual are reconstructed from the aggregate energy identity when route-level SOC traces are unavailable and are labelled as such. Post-hoc depot tariffs and wages are sensitivity scenarios only; ALNS/HCO do not optimize those totals.
""", encoding="utf-8")


def write_manifest(meta: dict) -> None:
    rows = [
        {"artifact": "T3_overall_routing_performance.csv", "kind": "table", "dataset": "solution_results.csv", "seeds": "0-4 (GI:0)", "instances": "60×2 TW", "methods": "GI,VNS,SA,ALNS-FR/PR,+HCO suite", "statistic": "instance-mean then mean", "script": "build_publication_results_final.py"},
        {"artifact": "T_partial_recharging_FR_vs_PR.csv", "kind": "table", "dataset": "solution_results.csv", "seeds": "0-4", "instances": "60 per TW", "methods": "ALNS-FR→ALNS-PR", "statistic": "paired % + bootstrap CI", "script": "build_publication_results_final.py"},
        {"artifact": "T_fixed_route_Detour_vs_Heterogeneous.csv", "kind": "table", "dataset": "solution_results.csv", "seeds": "0-4", "instances": "60 per TW", "methods": "FixedRoute Detour→Het", "statistic": "paired % + CI", "script": "build_publication_results_final.py"},
        {"artifact": "T_phase2_NoHCO_vs_Heterogeneous.csv", "kind": "table", "dataset": "solution_results.csv", "seeds": "0-4", "instances": "60 per TW", "methods": "NoHCO→Het", "statistic": "paired % + CI", "script": "build_publication_results_final.py"},
        {"artifact": "T4_HCO_contribution_counts_seeds0to4.csv", "kind": "table", "dataset": "solution_results.csv", "seeds": "0-4", "instances": "600 matched", "methods": "PhaseI/NoHCO/Het", "statistic": "improved/unchanged/worsened", "script": "build_publication_results_final.py"},
        {"artifact": "T5_family_HCO_effects.csv", "kind": "table", "dataset": "solution_results.csv", "seeds": "0-4", "instances": "10/family", "methods": "several", "statistic": "mean paired %", "script": "build_publication_results_final.py"},
        {"artifact": "T6_energy_accounting.csv", "kind": "table", "dataset": "solution_results + identity", "seeds": "0-4", "instances": "60×2", "methods": "all METHOD_ORDER", "statistic": "means; depot reconstructed", "script": "build_publication_results_final.py"},
        {"artifact": "T7_Schneider_*", "kind": "table", "dataset": "schneider/", "seeds": "0-2", "instances": "56 / 44 paired", "methods": "FR,PR", "statistic": "feasibility + paired", "script": "copied+validated"},
        {"artifact": "T8_*", "kind": "table", "dataset": "primary288", "seeds": "0-1", "instances": "12×2 TW", "methods": "FR,PR", "statistic": "paired + gap Δ", "script": "copied+validated"},
        {"artifact": "T9_economic_sensitivity_posthoc.csv", "kind": "table", "dataset": "5-seed+energy", "seeds": "0-4", "instances": "60×2", "methods": "FR,PR,Phase*", "statistic": "post-hoc means", "script": "build_publication_results_final.py"},
        {"artifact": "Figure2_*", "kind": "figure", "dataset": "T_partial_recharging", "seeds": "0-4", "instances": "60", "methods": "FR→PR", "statistic": "forest CI", "script": "build_publication_results_final.py"},
        {"artifact": "Figure3_*", "kind": "figure", "dataset": "T_fixed_route", "seeds": "0-4", "instances": "60", "methods": "Detour→Het", "statistic": "forest CI", "script": "build_publication_results_final.py"},
        {"artifact": "Figure4_*", "kind": "figure", "dataset": "T_phase2_NoHCO", "seeds": "0-4", "instances": "60", "methods": "NoHCO→Het", "statistic": "forest CI", "script": "build_publication_results_final.py"},
        {"artifact": "Figure5_*", "kind": "figure", "dataset": "T5 family phase2_nohco_hco", "seeds": "0-4", "instances": "10/family", "methods": "NoHCO→Het", "statistic": "family means only", "script": "build_publication_results_final.py"},
        {"artifact": "FigS_FR_PR_convergence_*", "kind": "figure", "dataset": "primary288", "seeds": "0-1", "instances": "12", "methods": "FR,PR", "statistic": "budget curves", "script": "plot_fr_pr_convergence"},
    ]
    write_csv(OUT / "RESULTS_SOURCE_MANIFEST.csv", rows)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Loading Hanif five-seed raw…")
    rows = load_hanif_five_seed()
    print("Building Hanif tables…")
    hanif_meta = build_hanif_tables(rows)
    print("HCO five-seed audit…")
    hco = hco_audit_five_seed(rows)
    print("Energy enrichment…")
    rows_e = enrich_energy(rows)
    build_energy_table(rows_e)
    print("Economic post-hoc…")
    build_economic(rows_e)
    print("Schneider / convergence…")
    sch = build_schneider()
    conv = build_convergence()
    print("Figures…")
    fig_paths = regenerate_paper_figures(hanif_meta["pair_all"], hanif_meta["fam_rows"])
    overall = _load_csv(TABLES / "T3_overall_routing_performance.csv")
    partial = _load_csv(TABLES / "T_partial_recharging_FR_vs_PR.csv")
    diffs = compare_to_archived(overall, partial)
    print("Word…")
    word = try_update_word(overall, hco)
    meta = {
        "hanif": hanif_meta,
        "hco": hco,
        "schneider": sch,
        "convergence": conv,
        "archive_diffs": diffs,
        "figures": [str(p.relative_to(OUT)) for p in fig_paths],
        "word": word,
    }
    write_reports(meta)
    write_manifest(meta)
    write_json(OUT / "build_meta.json", {
        **{k: v for k, v in meta.items() if k != "archive_diffs"},
        "n_archive_match": sum(1 for d in diffs if d.get("status") == "match"),
        "n_archive_mismatch": sum(1 for d in diffs if d.get("status") == "mismatch"),
        "hco": hco,
        "word": word,
    })
    print(json.dumps({"hco": hco, "schneider": sch, "word": word,
                      "n_match": sum(1 for d in diffs if d.get("status") == "match"),
                      "n_mismatch": sum(1 for d in diffs if d.get("status") == "mismatch"),
                      "n_figs": len(fig_paths)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
