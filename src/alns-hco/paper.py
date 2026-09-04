"""Generate Hanif paper tables, figures, captions, and RESULTS_SUMMARY from master CSV."""
from __future__ import annotations

import csv
import json
import math
import platform
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.common.config import (
    FAMILIES, METHOD_ALNS_FR, METHOD_ALNS_HCO, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO,
    METHOD_GI, METHOD_SA, METHOD_VNS, N_RUNS, PAPER_METHODS, SEEDS, TIE_TOL,
    TOTAL_BUDGET_S, TW_VARIANTS, experiment_config,
)
from src.common.paths import hanif_dirs
from src.common.stats import bootstrap_mean_ci, mean, median, pct_change, stdev, wilcoxon_p, wins_ties_losses

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": False,
    "figure.facecolor": "white",
})

PAIR_METRICS = (
    ("k", "fleet $K$"),
    ("distance", "distance"),
    ("duration", "duration"),
    ("customer_waiting", "customer waiting"),
    ("charging_visits", "charging visits"),
    ("charged_energy", "charged energy"),
    ("station_waiting", "station waiting"),
    ("charging_time", "charging time"),
    ("charging_expense", "charging expense"),
    ("runtime_s", "runtime (s)"),
)
# Mean of instance-level percentages is unstable when denominators vary widely.
UNSTABLE_PAIRED_PCT = {"customer_waiting"}
PHASE1_CHANGE_TOL = 1e-6


def _f(raw, default=None):
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            rec = dict(raw)
            for k in (
                "k", "distance", "duration", "customer_waiting", "charging_visits",
                "charged_energy", "station_waiting", "charging_time", "charging_expense",
                "runtime_s", "served", "unserved", "hard_violations", "seed",
                "phase1_s", "phase2_s",
                "phase1_k", "phase1_distance", "phase1_charging_expense", "k_star",
            ):
                if k in rec:
                    rec[k] = _f(rec[k])
            rec["feasible"] = str(rec.get("feasible")).lower() in {"true", "1", "yes"}
            rows.append(rec)
        return rows


def _instance_means(rows: list[dict], method: str, tw: str) -> dict[tuple, dict]:
    groups = defaultdict(list)
    for r in rows:
        if r.get("method") == method and r.get("tw") == tw and r.get("feasible"):
            groups[(r["instance"], r.get("family"))].append(r)
    out = {}
    for key, items in groups.items():
        out[key] = {
            "instance": key[0],
            "family": key[1],
            "n": len(items),
            **{
                m: mean([x[m] for x in items if x.get(m) is not None])
                for m, _ in PAIR_METRICS
            },
        }
    return out


def _write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = fields or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _save_fig(fig, dest: Path) -> None:
    """Save manuscript figures as PNG only (no PDF/SVG sidecars)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


def _seed_key(r: dict) -> tuple:
    return (r["instance"], r.get("tw"), int(r.get("seed") or 0))


def _cross_run_k_mismatches(rows: list[dict], left: str, right: str) -> dict[str, tuple[int, int]]:
    a = {_seed_key(r): r for r in rows if r.get("method") == left and r.get("feasible")}
    b = {_seed_key(r): r for r in rows if r.get("method") == right and r.get("feasible")}
    out = {}
    for tw in TW_VARIANTS:
        n_both = n_diff = 0
        for key, ra in a.items():
            if key[1] != tw:
                continue
            rb = b.get(key)
            if rb is None:
                continue
            n_both += 1
            if abs((ra.get("k") or 0.0) - (rb.get("k") or 0.0)) > PHASE1_CHANGE_TOL:
                n_diff += 1
        out[tw] = (n_diff, n_both)
    return out


def _within_run_phase1(rows: list[dict], method: str, tw: str) -> dict:
    sub = [r for r in rows if r.get("method") == method and r.get("tw") == tw and r.get("feasible")]
    fam = defaultdict(int)
    n_has = n_k = n_dc = 0
    p1k, fk, p1d, fd, p1c, fc = [], [], [], [], [], []
    for r in sub:
        if r.get("phase1_distance") is None and r.get("phase1_k") is None:
            continue
        n_has += 1
        p1k.append(r.get("phase1_k"))
        fk.append(r.get("k"))
        p1d.append(r.get("phase1_distance"))
        fd.append(r.get("distance"))
        p1c.append(r.get("phase1_charging_expense"))
        fc.append(r.get("charging_expense"))
        if r.get("phase1_k") is not None and abs((r.get("k") or 0.0) - r["phase1_k"]) > PHASE1_CHANGE_TOL:
            n_k += 1
        dchg = r.get("phase1_distance") is not None and abs((r.get("distance") or 0.0) - r["phase1_distance"]) > PHASE1_CHANGE_TOL
        cchg = r.get("phase1_charging_expense") is not None and abs((r.get("charging_expense") or 0.0) - r["phase1_charging_expense"]) > PHASE1_CHANGE_TOL
        if dchg or cchg:
            n_dc += 1
            fam[r.get("family") or "?"] += 1
    return {
        "tw": tw,
        "method": method,
        "n_seed_runs": len(sub),
        "n_with_phase1": n_has,
        "n_k_changed": n_k,
        "n_distance_or_charging_expense_changed": n_dc,
        "changed_by_family": ", ".join(f"{k}:{v}" for k, v in sorted(fam.items())) if fam else "",
        "mean_phase1_k": mean([x for x in p1k if x is not None]),
        "mean_final_k": mean([x for x in fk if x is not None]),
        "mean_phase1_distance": mean([x for x in p1d if x is not None]),
        "mean_final_distance": mean([x for x in fd if x is not None]),
        "mean_phase1_charging_expense": mean([x for x in p1c if x is not None]),
        "mean_final_charging_expense": mean([x for x in fc if x is not None]),
    }


def table01_setup(dirs, n_inst: int) -> None:
    cfg = experiment_config()
    rows = [
        {"setting": "benchmark", "value": "Hanif 1000-customer EVRPTW"},
        {"setting": "instances per TW variant", "value": n_inst or 60},
        {"setting": "customers per instance", "value": 1000},
        {"setting": "TW variants", "value": "original, relaxed"},
        {"setting": "independent runs (ALNS-FR/PR/HCO)", "value": cfg["n_runs"]},
        {"setting": "seeds", "value": list(SEEDS)},
        {"setting": "ALNS-FR / ALNS-PR budget (s)", "value": TOTAL_BUDGET_S},
        {"setting": "ALNS-HCO total budget (s)", "value": f"{TOTAL_BUDGET_S} (Phase I {cfg['alns_hco_phase1_s']} + Phase II {cfg['alns_hco_phase2_s']})"},
        {"setting": "Python", "value": cfg["python"]},
        {"setting": "OS", "value": cfg["platform"]},
        {"setting": "CPU", "value": cfg["processor"] or platform.processor()},
        {"setting": "ALNS destroy size", "value": f"{HYPERPARAMETERS['destroy_size_min']}–{HYPERPARAMETERS['destroy_size_max']}"},
        {"setting": "HCO", "value": "deterministic route-level charging heuristic (one oracle)"},
    ]
    _write_csv(dirs["paper_tables"] / "table01_experimental_setup.csv", rows)
    _write_csv(dirs["paper_data"] / "table01_experimental_setup.csv", rows)
    md = ["# Table 1. Experimental setup", "", "| Setting | Value |", "|---|---|"]
    tex = [r"\begin{tabular}{ll}", r"\toprule", r"Setting & Value \\", r"\midrule"]
    for r in rows:
        md.append(f"| {r['setting']} | {r['value']} |")
        tex.append(f"{r['setting']} & {r['value']} \\\\")
    md.append("")
    tex += [r"\bottomrule", r"\end{tabular}", ""]
    (dirs["paper_tables"] / "table01_experimental_setup.md").write_text("\n".join(md), encoding="utf-8")
    (dirs["paper_tables"] / "table01_experimental_setup.tex").write_text("\n".join(tex), encoding="utf-8")


def table02_comparison(dirs, rows: list[dict]) -> None:
    methods = list(PAPER_METHODS)
    out = []
    md = [
        "# Table 2. Algorithm comparison",
        "",
        "Per-instance means over seeds, then mean over the 60 feasible instances. Lower is better.",
        "Displayed values are means only. CSV columns `std_k` and `std_distance` are the SD **across instances**, not across seeds.",
        "Those SDs are large because families are heterogeneous (C1 often needs 100+ vehicles; R2 often needs about 20–30).",
        "They are omitted from the displayed table so they are not read as algorithm instability.",
        "",
    ]
    for tw in TW_VARIANTS:
        md += [
            f"## {tw.title()} TW",
            "",
            "| Method | n | Feasible | Mean $K$ | Mean distance | Mean duration | Mean runtime (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for method in methods:
            sub = [r for r in rows if r.get("method") == method and r.get("tw") == tw]
            inst = _instance_means(rows, method, tw)
            n_inst = len({r["instance"] for r in sub})
            rec = {
                "tw": tw, "method": method, "n_rows": len(sub), "n_instances": n_inst,
                "feasible_instances": len(inst),
                "mean_k": mean([v["k"] for v in inst.values()]),
                "std_k": stdev([v["k"] for v in inst.values()]),
                "std_scope": "across_instances",
                "mean_distance": mean([v["distance"] for v in inst.values()]),
                "std_distance": stdev([v["distance"] for v in inst.values()]),
                "mean_duration": mean([v["duration"] for v in inst.values()]),
                "mean_runtime_s": mean([v["runtime_s"] for v in inst.values()]),
            }
            out.append(rec)
            md.append(
                f"| {method} | {n_inst} | {rec['feasible_instances']}/{n_inst} | {_fmt(rec['mean_k'], 2)} | "
                f"{_fmt(rec['mean_distance'], 1)} | {_fmt(rec['mean_duration'], 1)} | {_fmt(rec['mean_runtime_s'], 2)} |"
            )
        md.append("")
    _write_csv(dirs["paper_tables"] / "table02_algorithm_comparison.csv", out)
    _write_csv(dirs["paper_data"] / "table02_algorithm_comparison.csv", out)
    (dirs["paper_tables"] / "table02_algorithm_comparison.md").write_text("\n".join(md), encoding="utf-8")
    (dirs["paper_tables"] / "table02_algorithm_comparison.tex").write_text(
        "\n".join([
            "% Means only. CSV std_* columns are SD across instances, not seeds.",
            "\\begin{tabular}{llrrrrr}", "\\toprule",
            "TW & method & n feas. & $K$ & distance & duration & s \\\\", "\\midrule",
        ] + [
            f"{r['tw']} & {r['method']} & {r['feasible_instances']} & {_fmt(r['mean_k'])} & "
            f"{_fmt(r['mean_distance'],1)} & {_fmt(r['mean_duration'],1)} & {_fmt(r['mean_runtime_s'])} \\\\"
            for r in out
        ] + ["\\bottomrule", "\\end{tabular}", ""]),
        encoding="utf-8",
    )


def _paired(rows, tw: str):
    pr = _instance_means(rows, METHOD_ALNS_PR, tw)
    hco = _instance_means(rows, METHOD_ALNS_HCO, tw)
    keys = sorted(set(pr) & set(hco))
    return pr, hco, keys


def table03_effect(dirs, rows: list[dict]) -> None:
    stats_rows = []
    md = [
        "# Table 3. ALNS-PR vs ALNS-HCO",
        "",
        "Paired instance-level means (seeds averaged first). Relative change is the mean of",
        r"$100(M_{\mathrm{HCO}}-M_{\mathrm{PR}})/M_{\mathrm{PR}}$. Negative = reduction.",
        "",
        "The $K$ gap is not an HCO effect. ALNS-PR spends the full 30 s on Phase I; ALNS-HCO splits the same budget into about 15 s Phase I + 15 s Phase II and then freezes $K=K^*$. Phase II cannot change fleet size.",
        "",
        "Customer-waiting mean paired percentages are marked $\\dagger$ and are **not a headline result**. Small waiting denominators can dominate the mean of ratios: in Original TW the average waiting falls while the mean paired percentage is positive.",
        "",
    ]
    for tw in TW_VARIANTS:
        pr, hco, keys = _paired(rows, tw)
        md += [
            f"## {tw.title()} TW (n={len(keys)} paired instances)",
            "",
            "| Metric | ALNS-PR | ALNS-HCO | Δ | mean % | wins/ties/losses | 95% CI % | p |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for metric, label in PAIR_METRICS:
            a = [pr[k][metric] for k in keys]
            b = [hco[k][metric] for k in keys]
            pcts = [pct_change(bi, ai) for ai, bi in zip(a, b)]
            pcts_f = [p for p in pcts if p is not None]
            wtl = wins_ties_losses(b, a, tol=TIE_TOL)
            lo, hi = bootstrap_mean_ci(pcts_f, seed=0)
            pval = wilcoxon_p([(bi - ai) for ai, bi in zip(a, b) if ai is not None and bi is not None])
            unstable = metric in UNSTABLE_PAIRED_PCT
            rec = {
                "tw": tw, "metric": metric, "n": len(keys),
                "alns_pr_mean": mean(a), "alns_hco_mean": mean(b),
                "abs_diff": None if mean(a) is None or mean(b) is None else mean(b) - mean(a),
                "mean_paired_pct": mean(pcts_f),
                "pct_definition": "mean of paired 100*(HCO-PR)/PR",
                "pct_stable": (not unstable),
                "headline": (not unstable) and metric != "runtime_s",
                "wins": wtl[0], "ties": wtl[1], "losses": wtl[2],
                "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_p": pval,
            }
            stats_rows.append(rec)
            mark = r" $\dagger$" if unstable else ""
            pct_cell = _fmt(rec["mean_paired_pct"], 2) + mark
            ci = "" if lo is None else f"[{_fmt(lo,2)}, {_fmt(hi,2)}]"
            if unstable:
                ci = ci + mark if ci else mark.strip()
            md.append(
                f"| {label}{mark} | {_fmt(rec['alns_pr_mean'], 2)} | {_fmt(rec['alns_hco_mean'], 2)} | "
                f"{_fmt(rec['abs_diff'], 2)} | {pct_cell} | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} | {ci} | {_fmt(pval, 4)} |"
            )
        md.append("")
    md += [
        r"$\dagger$ Mean of instance-level percentages is unstable for customer waiting. Use the absolute $\Delta$, not the percentage, if this metric is discussed at all.",
        "",
    ]
    _write_csv(dirs["paper_tables"] / "table03_main_hco_effect.csv", stats_rows)
    _write_csv(dirs["paper_data"] / "table03_main_hco_effect.csv", stats_rows)
    _write_csv(dirs["statistics"] / "statistical_tests.csv", stats_rows)
    _write_csv(dirs["statistics"] / "table03_paired_alns_pr_vs_hco.csv", stats_rows)
    (dirs["paper_tables"] / "table03_main_hco_effect.md").write_text("\n".join(md), encoding="utf-8")
    tex = [
        "\\begin{tabular}{lrrrrlrr}", "\\toprule",
        "metric & PR & HCO & $\\Delta$ & \\% & W/T/L & CI\\% & $p$ \\\\", "\\midrule",
    ]
    for r in stats_rows:
        mark = r" $\dagger$" if not r["pct_stable"] else ""
        tex.append(
            f"{r['tw']}/{r['metric']}{mark} & {_fmt(r['alns_pr_mean'])} & {_fmt(r['alns_hco_mean'])} & "
            f"{_fmt(r['abs_diff'])} & {_fmt(r['mean_paired_pct'])}{mark} & "
            f"{r['wins']}/{r['ties']}/{r['losses']} & [{_fmt(r['ci95_lo'])},{_fmt(r['ci95_hi'])}] & {_fmt(r['wilcoxon_p'],4)} \\\\"
        )
    tex += [
        "\\bottomrule", "\\end{tabular}",
        r"\par\vspace{0.4em}{\small $\dagger$ Customer-waiting mean paired \% is unstable; do not use it as a headline result.}",
        "",
    ]
    (dirs["paper_tables"] / "table03_main_hco_effect.tex").write_text("\n".join(tex), encoding="utf-8")


def table04_ablation(dirs, rows: list[dict]) -> None:
    methods = [METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO, METHOD_ALNS_HCO]
    cols = [
        "k", "distance", "duration", "charging_visits", "charged_energy",
        "station_waiting", "charging_time", "charging_expense", "runtime_s",
    ]
    mismatch = _cross_run_k_mismatches(rows, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO)
    n_mismatch = sum(v[0] for v in mismatch.values())
    n_pairs = sum(v[1] for v in mismatch.values())
    out = []
    within = []
    md = [
        "# Table 4. Ablation",
        "",
        "ALNS-FR = full recharge (30 s). ALNS-PR = partial recharge (30 s Phase I).",
        "ALNS-PR+HCO = a **separate** 30 s ALNS-PR run, then HCO on that run's frozen sequences.",
        "ALNS-HCO = 15 s Phase I + 15 s Phase II with frozen $K=K^*$.",
        "",
        "## Panel A. Cross-run method means",
        "",
        "These four rows are independent 30 s searches. **Do not read ALNS-PR → ALNS-PR+HCO as an exact fixed-route ablation.**",
        "The wall-clock stopping criterion can yield a different Phase-I solution even at the same seed.",
        f"Fleet size $K$ differs between the stored ALNS-PR and ALNS-PR+HCO runs on **{n_mismatch} of {n_pairs}** instance–seed pairs"
        + " ("
        + "; ".join(f"{tw}: {mismatch[tw][0]}/{mismatch[tw][1]}" for tw in TW_VARIANTS)
        + "). HCO itself never changes $K$ (Panel B). The larger mean $K$ of ALNS-HCO vs ALNS-PR is the shorter Phase-I budget, not an HCO fleet-size effect.",
        "",
    ]
    tex = [
        r"% Panel A: independent 30 s runs. PR vs PR+HCO is not a perfectly controlled fixed-route comparison.",
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"TW & method & $K$ & $D$ & $T$ & $N_{\mathrm{ch}}$ & $E_{\mathrm{ch}}$ & $T_{\mathrm{wait}}$ & $T_{\mathrm{ch}}$ & $C_{\mathrm{ch}}$ \\",
        r"\midrule",
    ]
    for tw in TW_VARIANTS:
        md += [
            f"### {tw.title()} TW",
            "",
            "| Method | $K$ | distance | duration | $N_{ch}$ | energy | wait | ch. time | ch. cost | s |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for method in methods:
            inst = _instance_means(rows, method, tw)
            rec = {"tw": tw, "method": method, "n": len(inst), "panel": "A_cross_run"}
            for c in cols:
                rec[c] = mean([v[c] for v in inst.values()])
            out.append(rec)
            md.append(
                f"| {method} | {_fmt(rec['k'])} | {_fmt(rec['distance'],1)} | {_fmt(rec['duration'],1)} | "
                f"{_fmt(rec['charging_visits'],2)} | {_fmt(rec['charged_energy'],1)} | {_fmt(rec['station_waiting'],1)} | "
                f"{_fmt(rec['charging_time'],1)} | {_fmt(rec['charging_expense'],1)} | {_fmt(rec['runtime_s'])} |"
            )
            tex.append(
                f"{tw} & {method} & {_fmt(rec['k'])} & {_fmt(rec['distance'],1)} & {_fmt(rec['duration'],1)} & "
                f"{_fmt(rec['charging_visits'],2)} & {_fmt(rec['charged_energy'],1)} & {_fmt(rec['station_waiting'],1)} & "
                f"{_fmt(rec['charging_time'],1)} & {_fmt(rec['charging_expense'],1)} \\\\"
            )
        md.append("")
    tex += [r"\bottomrule", r"\end{tabular}", ""]

    md += [
        "## Panel B. Within-run Phase I vs final (the controlled HCO comparison)",
        "",
        "Each row uses the `phase1_*` fields stored **inside the same run**. This is the clean test of whether HCO / Phase II changed the Phase-I route.",
        "HCO preserves $K$ in every seed-run. Distance or charging expense changed in only a small Original-TW subset, concentrated in C1/C2, and in **none** of the Relaxed-TW seed-runs.",
        "",
        "| TW | Method | n seed-runs | $K$ changed | $D$ or $C_{ch}$ changed | mean $K$ Phase I | mean $K$ final | mean $D$ Phase I | mean $D$ final | mean $C_{ch}$ Phase I | mean $C_{ch}$ final | families with a change |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    tex += [
        r"% Panel B: within-run phase1_* vs final. This is the controlled HCO comparison.",
        r"\begin{tabular}{llrrrrrrrrl}",
        r"\toprule",
        r"TW & method & $n$ & $K$ chg. & $D$/$C$ chg. & $K_1$ & $K$ & $D_1$ & $D$ & $C_1$ & $C$ \\",
        r"\midrule",
    ]
    for tw in TW_VARIANTS:
        for method in (METHOD_ALNS_PR_HCO, METHOD_ALNS_HCO):
            rec = _within_run_phase1(rows, method, tw)
            within.append(rec)
            md.append(
                f"| {tw.title()} | {method} | {rec['n_seed_runs']} | {rec['n_k_changed']} | "
                f"{rec['n_distance_or_charging_expense_changed']} | {_fmt(rec['mean_phase1_k'])} | {_fmt(rec['mean_final_k'])} | "
                f"{_fmt(rec['mean_phase1_distance'],1)} | {_fmt(rec['mean_final_distance'],1)} | "
                f"{_fmt(rec['mean_phase1_charging_expense'],1)} | {_fmt(rec['mean_final_charging_expense'],1)} | "
                f"{rec['changed_by_family'] or '—'} |"
            )
            tex.append(
                f"{tw} & {method} & {rec['n_seed_runs']} & {rec['n_k_changed']} & "
                f"{rec['n_distance_or_charging_expense_changed']} & {_fmt(rec['mean_phase1_k'])} & {_fmt(rec['mean_final_k'])} & "
                f"{_fmt(rec['mean_phase1_distance'],1)} & {_fmt(rec['mean_final_distance'],1)} & "
                f"{_fmt(rec['mean_phase1_charging_expense'],1)} & {_fmt(rec['mean_final_charging_expense'],1)} \\\\"
            )
    md.append("")
    tex += [r"\bottomrule", r"\end{tabular}", ""]

    mismatch_rows = [
        {"tw": tw, "left": METHOD_ALNS_PR, "right": METHOD_ALNS_PR_HCO,
         "n_instance_seed_pairs": mismatch[tw][1], "n_k_differs": mismatch[tw][0],
         "note": "independent 30 s runs; not a controlled fixed-route pair"}
        for tw in TW_VARIANTS
    ]
    _write_csv(dirs["ablation"] / "table04_ablation.csv", out)
    _write_csv(dirs["ablation"] / "table04_within_run_phase1.csv", within)
    _write_csv(dirs["ablation"] / "table04_pr_vs_prhco_k_mismatch.csv", mismatch_rows)
    _write_csv(dirs["paper_tables"] / "table04_within_run_phase1.csv", within)
    (dirs["ablation"] / "table04_ablation.md").write_text("\n".join(md), encoding="utf-8")
    (dirs["paper_tables"] / "table04_ablation.md").write_text("\n".join(md), encoding="utf-8")
    (dirs["ablation"] / "table04_ablation.tex").write_text("\n".join(tex), encoding="utf-8")
    (dirs["paper_tables"] / "table04_ablation.tex").write_text("\n".join(tex), encoding="utf-8")


def table05_family(dirs, rows: list[dict]) -> None:
    metrics = ["k", "distance", "duration", "station_waiting", "charging_time", "charged_energy", "charging_expense"]
    out = []
    md = ["# Table 5. Family analysis", "", "Mean paired % change of ALNS-HCO vs ALNS-PR.", ""]
    for tw in TW_VARIANTS:
        pr, hco, keys = _paired(rows, tw)
        md += [f"## {tw.title()} TW", "", "| Family | n | " + " | ".join(metrics) + " |",
               "|---|---:|" + "|".join("---:" for _ in metrics) + "|"]
        for fam in FAMILIES:
            fk = [k for k in keys if k[1] == fam]
            rec = {"tw": tw, "family": fam, "n": len(fk)}
            cells = []
            for m in metrics:
                pcts = [pct_change(hco[k][m], pr[k][m]) for k in fk]
                rec[m] = mean([p for p in pcts if p is not None])
                cells.append(_fmt(rec[m], 2))
            out.append(rec)
            md.append(f"| {fam} | {len(fk)} | " + " | ".join(cells) + " |")
        md.append("")
    _write_csv(dirs["paper_tables"] / "table05_family_analysis.csv", out)
    _write_csv(dirs["paper_data"] / "table05_family_analysis.csv", out)
    (dirs["paper_tables"] / "table05_family_analysis.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "TW & family & $K$ & $D$ & $T$ & wait & $t_{ch}$ & energy & $C$ \\\\", "\\midrule"]
    for r in out:
        tex.append(
            f"{r['tw']} & {r['family']} & " + " & ".join(_fmt(r[m]) for m in metrics) + " \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    (dirs["paper_tables"] / "table05_family_analysis.tex").write_text("\n".join(tex), encoding="utf-8")


def table_a01(dirs) -> None:
    cfg = experiment_config()
    rows = [{"item": k, "value": v} for k, v in cfg.items()]
    rows.append({"item": "hco_description", "value": HYPERPARAMETERS["hco"]})
    _write_csv(dirs["paper_tables"] / "tableA01_reproducibility.csv", rows)
    md = ["# Table A1. Reproducibility", "", "| Item | Value |", "|---|---|"]
    for r in rows:
        md.append(f"| {r['item']} | {r['value']} |")
    (dirs["paper_tables"] / "tableA01_reproducibility.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (dirs["paper_tables"] / "tableA01_reproducibility.tex").write_text(
        "\\begin{tabular}{ll}\n\\toprule\nItem & Value \\\\\n\\midrule\n"
        + "\n".join(f"{r['item']} & {r['value']} \\\\" for r in rows)
        + "\n\\bottomrule\n\\end{tabular}\n",
        encoding="utf-8",
    )



def _captions(dirs, n_inst: int) -> None:
    n = n_inst or "N"
    text = f"""# Captions

Table 1. Experimental settings for the Hanif 1000-customer study ({n} instances per TW variant, {N_RUNS} seeds for stochastic methods). Lower is better for routing and charging metrics.

Table 2. Mean fleet size, distance, duration, and runtime. Stochastic methods are instance-level means over seeds, then averaged over instances. Displayed cells are means only. Standard deviations in the CSV are across the {n} heterogeneous instances, not across seeds, and are omitted from the table so they are not mistaken for algorithm instability.

Table 3. Paired ALNS-PR vs ALNS-HCO comparison. Relative change is the mean of instance-level $100(M_{{HCO}}-M_{{PR}})/M_{{PR}}$. Negative values are reductions. Wins/ties/losses use tolerance {TIE_TOL}. CI is a paired bootstrap 95% interval on that mean percentage. $p$ is Wilcoxon signed-rank on paired differences. The $K$ difference is a Phase-I budget effect (30 s vs about 15 s), not an HCO fleet-size effect. Customer-waiting percentages are unstable and are not a headline result.

Table 4. Ablation in two panels. Panel A reports independent 30 s runs: full recharge (ALNS-FR), partial recharge (ALNS-PR), a separately run ALNS-PR followed by HCO on that run's frozen sequences (ALNS-PR+HCO), and the full method (ALNS-HCO). Panel A is **not** an exact fixed-route comparison of ALNS-PR vs ALNS-PR+HCO. Panel B is the controlled comparison: each run's stored Phase-I state versus its own final solution. HCO never changes $K$. Phase II / HCO changes distance or charging expense in only a small Original-TW subset.

Table 5. Mean paired percentage change of ALNS-HCO vs ALNS-PR by Solomon-style family.

Figure 1. Two-phase ALNS-HCO framework (author drawing). Phase I is ALNS-PR and returns $X^{{\\mathrm{{PR}}}}$ with $K^*=K(X^{{\\mathrm{{PR}}}})$. Phase II is charging-aware ALNS with frozen $K=K^*$, HCO inside the search, acceptance/rejection, and output $X^*$. This figure is not generated by the result compiler.

Figure 2. Illustrative HCO charging decision. A customer arc is battery-infeasible; a restricted set of stations $\\{{s_1,s_2,s_3\\}}$ is scored by detour, $\\bar P_s$, $\\bar W_s$, and $r_s$; HCO inserts a station $s^*$, assigns a just-enough partial charge $e_s$, and replays route feasibility. The diagram is a methodological illustration, not an experimental result, and is not evidence of selected-station quality in the master CSV.

Figure 3. Partial-recharging effect. Dots are the mean of instance-level $100(M_{{\\mathrm{{PR}}}}-M_{{\\mathrm{{FR}}}})/M_{{\\mathrm{{FR}}}}$; whiskers are paired bootstrap 95% confidence intervals. Negative values are reductions. The large drops in $E_{{\\mathrm{{ch}}}}$, $T_{{\\mathrm{{ch}}}}$, and $C_{{\\mathrm{{ch}}}}$ are the main successful charging gains in this study. Four-stage cross-run profiles are not shown here; see Table 4 and the optional supplement figure.

Figure 4. Paired ALNS-HCO versus ALNS-PR. Dots are the mean of instance-level $100(M_{{\\mathrm{{HCO}}}}-M_{{\\mathrm{{PR}}}})/M_{{\\mathrm{{PR}}}}$; whiskers are paired bootstrap 95% confidence intervals. The vertical line is zero. Negative values are reductions. Do not interpret the $K$ shift as HCO increasing fleet size.

Figure 5. Family-level charging effects of ALNS-HCO versus ALNS-PR for $N_{{\\mathrm{{ch}}}}$, $T_{{\\mathrm{{wait}}}}$, and $C_{{\\mathrm{{ch}}}}$. Blue is a reduction. Family signs differ; Original C1 is the strongest positive HCO case, while several families show higher charging expense.
"""
    (dirs["paper"] / "captions.md").write_text(text, encoding="utf-8")


def _summary(dirs, rows: list[dict]) -> None:
    lines = [
        "# Hanif ALNS-HCO results (factual)",
        "",
        "Interpretation locked from the master CSV. Do not claim that HCO increases fleet size,",
        "broadly reorganizes charging plans, or consistently selects cheaper/faster stations.",
        "",
    ]
    feas = [r for r in rows if r.get("feasible")]
    lines.append(f"- Master rows: {len(rows)}; feasible rows: {len(feas)}.")
    mismatch = _cross_run_k_mismatches(rows, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO)
    n_mismatch = sum(v[0] for v in mismatch.values())
    n_pairs = sum(v[1] for v in mismatch.values())
    lines.append(
        f"- ALNS-PR vs separately run ALNS-PR+HCO: $K$ differs on {n_mismatch} of {n_pairs} instance–seed pairs. "
        "That comparison is not a perfectly controlled fixed-route ablation."
    )
    for tw in TW_VARIANTS:
        pr, hco, keys = _paired(rows, tw)
        if not keys:
            lines.append(f"- {tw.title()} TW: no paired ALNS-PR / ALNS-HCO instances yet.")
            continue

        def m(method, field):
            inst = _instance_means(rows, method, tw)
            return mean([v[field] for v in inst.values()])

        pcts = [pct_change(hco[k]["charging_expense"], pr[k]["charging_expense"]) for k in keys]
        wtl = wins_ties_losses([hco[k]["charging_expense"] for k in keys], [pr[k]["charging_expense"] for k in keys])
        lo, hi = bootstrap_mean_ci([p for p in pcts if p is not None], seed=0)
        fr_e = m(METHOD_ALNS_FR, "charged_energy")
        pr_e = m(METHOD_ALNS_PR, "charged_energy")
        fr_c = m(METHOD_ALNS_FR, "charging_expense")
        pr_c = m(METHOD_ALNS_PR, "charging_expense")
        e_drop = None if fr_e in (None, 0) or pr_e is None else 100.0 * (pr_e - fr_e) / fr_e
        c_drop = None if fr_c in (None, 0) or pr_c is None else 100.0 * (pr_c - fr_c) / fr_c
        p1_hco = _within_run_phase1(rows, METHOD_ALNS_HCO, tw)
        p1_prh = _within_run_phase1(rows, METHOD_ALNS_PR_HCO, tw)
        lines += [
            f"## {tw.title()} TW (n={len(keys)} paired instances)",
            f"- ALNS-FR → ALNS-PR charged-energy change (ratio of means): {_fmt(e_drop)}%.",
            f"- ALNS-FR → ALNS-PR charging-expense change (ratio of means): {_fmt(c_drop)}%.",
            f"- ALNS-PR mean K: {_fmt(m(METHOD_ALNS_PR, 'k'))}",
            f"- ALNS-HCO mean K: {_fmt(m(METHOD_ALNS_HCO, 'k'))} (Phase I is about 15 s then K is frozen; this is not an HCO fleet-size effect).",
            f"- ALNS-PR mean distance: {_fmt(m(METHOD_ALNS_PR, 'distance'), 1)}",
            f"- ALNS-HCO mean distance: {_fmt(m(METHOD_ALNS_HCO, 'distance'), 1)}",
            f"- Charging-cost mean paired %: {_fmt(mean([p for p in pcts if p is not None]))} (95% CI [{_fmt(lo)}, {_fmt(hi)}])",
            f"- Charging-cost wins/ties/losses: {wtl[0]}/{wtl[1]}/{wtl[2]}",
            f"- ALNS-HCO within-run Phase II changed distance or charging expense in {p1_hco['n_distance_or_charging_expense_changed']} of {p1_hco['n_seed_runs']} seed-runs (K changed: {p1_hco['n_k_changed']}; {p1_hco['changed_by_family'] or 'no families'}).",
            f"- ALNS-PR+HCO within-run HCO changed distance or charging expense in {p1_prh['n_distance_or_charging_expense_changed']} of {p1_prh['n_seed_runs']} seed-runs (K changed: {p1_prh['n_k_changed']}; {p1_prh['changed_by_family'] or 'no families'}).",
            f"- ALNS-PR mean runtime (s): {_fmt(m(METHOD_ALNS_PR, 'runtime_s'))}",
            f"- ALNS-HCO mean runtime (s): {_fmt(m(METHOD_ALNS_HCO, 'runtime_s'))}",
            "",
        ]
    lines += [
        "## Reporting rules",
        "- Large energy / charging-time / charging-cost gains are ALNS-FR → ALNS-PR.",
        "- ALNS-PR → ALNS-HCO mainly reduces charging visits and station waiting, often with more energy per visit.",
        "- Do not write that HCO selects significantly cheaper or faster chargers: the master CSV has no selected-station $\\bar P_s$, $\\bar W_s$, or $r_s$ logs.",
        "- Do not use customer-waiting percentage change as a headline result.",
        "",
    ]
    (dirs["paper"] / "RESULTS_SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


def _copy_method(dirs) -> None:
    src = Path(__file__).resolve().parent / "METHOD.md"
    if src.is_file():
        (dirs["paper"] / "METHOD.md").write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


def compile_hanif_paper(master_path: Path) -> None:
    dirs = hanif_dirs()
    rows = _load(master_path)
    n_inst = len({r["instance"] for r in rows if r.get("tw") == "original"})
    table01_setup(dirs, n_inst)
    table02_comparison(dirs, rows)
    table03_effect(dirs, rows)
    table04_ablation(dirs, rows)
    table05_family(dirs, rows)
    table_a01(dirs)
    from src.alns_hco.figures import generate_all_figures
    generate_all_figures(dirs, rows, master_path=master_path)
    _captions(dirs, n_inst)
    _summary(dirs, rows)
    _copy_method(dirs)
    (dirs["paper"] / "experiment_config.json").write_text(
        json.dumps(experiment_config(), indent=2) + "\n", encoding="utf-8",
    )
