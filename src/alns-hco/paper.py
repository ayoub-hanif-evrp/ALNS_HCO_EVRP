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
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(dest.with_suffix(".svg"), bbox_inches="tight")
    try:
        fig.savefig(dest.with_suffix(".pdf"), bbox_inches="tight")
    except Exception:
        pass
    plt.close(fig)


def _fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return ""
    return f"{x:.{nd}f}"


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
    md = ["# Table 2. Algorithm comparison", "", "Per-instance means over seeds, then mean over feasible instances. Lower is better.", ""]
    tex_blocks = []
    for tw in TW_VARIANTS:
        md += [f"## {tw.title()} TW", "", "| Method | n | Feasible | Mean K | Mean distance | Mean duration | Mean runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|"]
        tex_blocks.append(f"% {tw}")
        for method in methods:
            sub = [r for r in rows if r.get("method") == method and r.get("tw") == tw]
            inst = _instance_means(rows, method, tw)
            feas = [r for r in sub if r.get("feasible")]
            n_inst = len({r["instance"] for r in sub})
            rec = {
                "tw": tw, "method": method, "n_rows": len(sub), "n_instances": n_inst,
                "feasible_instances": len(inst),
                "mean_k": mean([v["k"] for v in inst.values()]),
                "std_k": stdev([v["k"] for v in inst.values()]),
                "mean_distance": mean([v["distance"] for v in inst.values()]),
                "std_distance": stdev([v["distance"] for v in inst.values()]),
                "mean_duration": mean([v["duration"] for v in inst.values()]),
                "mean_runtime_s": mean([v["runtime_s"] for v in inst.values()]),
            }
            out.append(rec)
            k = _fmt(rec["mean_k"], 2)
            if rec["std_k"] is not None and method != METHOD_GI:
                k = f"{k} ± {_fmt(rec['std_k'], 2)}"
            md.append(
                f"| {method} | {n_inst} | {rec['feasible_instances']}/{n_inst} | {k} | "
                f"{_fmt(rec['mean_distance'], 1)} | {_fmt(rec['mean_duration'], 1)} | {_fmt(rec['mean_runtime_s'], 2)} |"
            )
        md.append("")
    _write_csv(dirs["paper_tables"] / "table02_algorithm_comparison.csv", out)
    _write_csv(dirs["paper_data"] / "table02_algorithm_comparison.csv", out)
    (dirs["paper_tables"] / "table02_algorithm_comparison.md").write_text("\n".join(md), encoding="utf-8")
    (dirs["paper_tables"] / "table02_algorithm_comparison.tex").write_text(
        "\n".join(["\\begin{tabular}{llrrrrr}", "\\toprule",
                   "TW & method & n feas. & $K$ & distance & duration & s \\\\", "\\midrule"]
                  + [f"{r['tw']} & {r['method']} & {r['feasible_instances']} & {_fmt(r['mean_k'])} & {_fmt(r['mean_distance'],1)} & {_fmt(r['mean_duration'],1)} & {_fmt(r['mean_runtime_s'])} \\\\" for r in out]
                  + ["\\bottomrule", "\\end{tabular}", ""]),
        encoding="utf-8",
    )


def _paired(rows, tw: str):
    pr = _instance_means(rows, METHOD_ALNS_PR, tw)
    hco = _instance_means(rows, METHOD_ALNS_HCO, tw)
    keys = sorted(set(pr) & set(hco))
    return pr, hco, keys


def table03_effect(dirs, rows: list[dict]) -> None:
    stats_rows = []
    md = ["# Table 3. ALNS-PR vs ALNS-HCO", "",
          "Paired instance-level means (seeds averaged first). Relative change is the mean of",
          r"$100(M_{\mathrm{HCO}}-M_{\mathrm{PR}})/M_{\mathrm{PR}}$. Negative = reduction.", ""]
    for tw in TW_VARIANTS:
        pr, hco, keys = _paired(rows, tw)
        md += [f"## {tw.title()} TW (n={len(keys)} paired instances)", "",
               "| Metric | ALNS-PR | ALNS-HCO | Δ | mean % | wins/ties/losses | 95% CI % | p |",
               "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for metric, label in PAIR_METRICS:
            a = [pr[k][metric] for k in keys]
            b = [hco[k][metric] for k in keys]
            pcts = [pct_change(bi, ai) for ai, bi in zip(a, b)]
            pcts_f = [p for p in pcts if p is not None]
            wtl = wins_ties_losses(b, a, tol=TIE_TOL)
            lo, hi = bootstrap_mean_ci(pcts_f, seed=0)
            pval = wilcoxon_p([ (bi - ai) for ai, bi in zip(a, b) if ai is not None and bi is not None])
            rec = {
                "tw": tw, "metric": metric, "n": len(keys),
                "alns_pr_mean": mean(a), "alns_hco_mean": mean(b),
                "abs_diff": None if mean(a) is None or mean(b) is None else mean(b) - mean(a),
                "mean_paired_pct": mean(pcts_f),
                "pct_definition": "mean of paired 100*(HCO-PR)/PR",
                "wins": wtl[0], "ties": wtl[1], "losses": wtl[2],
                "ci95_lo": lo, "ci95_hi": hi, "wilcoxon_p": pval,
            }
            stats_rows.append(rec)
            ci = "" if lo is None else f"[{_fmt(lo,2)}, {_fmt(hi,2)}]"
            md.append(
                f"| {label} | {_fmt(rec['alns_pr_mean'], 2)} | {_fmt(rec['alns_hco_mean'], 2)} | "
                f"{_fmt(rec['abs_diff'], 2)} | {_fmt(rec['mean_paired_pct'], 2)} | "
                f"{wtl[0]}/{wtl[1]}/{wtl[2]} | {ci} | {_fmt(pval, 4)} |"
            )
        md.append("")
    _write_csv(dirs["paper_tables"] / "table03_main_hco_effect.csv", stats_rows)
    _write_csv(dirs["paper_data"] / "table03_main_hco_effect.csv", stats_rows)
    _write_csv(dirs["statistics"] / "statistical_tests.csv", stats_rows)
    _write_csv(dirs["statistics"] / "table03_paired_alns_pr_vs_hco.csv", stats_rows)
    (dirs["paper_tables"] / "table03_main_hco_effect.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["\\begin{tabular}{lrrrrlrr}", "\\toprule",
           "metric & PR & HCO & $\\Delta$ & \\% & W/T/L & CI\\% & $p$ \\\\", "\\midrule"]
    for r in stats_rows:
        tex.append(
            f"{r['tw']}/{r['metric']} & {_fmt(r['alns_pr_mean'])} & {_fmt(r['alns_hco_mean'])} & "
            f"{_fmt(r['abs_diff'])} & {_fmt(r['mean_paired_pct'])} & "
            f"{r['wins']}/{r['ties']}/{r['losses']} & [{_fmt(r['ci95_lo'])},{_fmt(r['ci95_hi'])}] & {_fmt(r['wilcoxon_p'],4)} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    (dirs["paper_tables"] / "table03_main_hco_effect.tex").write_text("\n".join(tex), encoding="utf-8")


def table04_ablation(dirs, rows: list[dict]) -> None:
    methods = [METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO, METHOD_ALNS_HCO]
    cols = ["k", "distance", "duration", "charged_energy", "station_waiting", "charging_time", "charging_expense", "runtime_s"]
    out = []
    md = ["# Table 4. Ablation", "",
          "ALNS-FR = full recharge. ALNS-PR = partial recharge. ALNS-PR+HCO = HCO on frozen sequences. ALNS-HCO = full method.", ""]
    for tw in TW_VARIANTS:
        md += [f"## {tw.title()} TW", "", "| Method | K | distance | duration | energy | wait | ch. time | ch. cost | s |",
               "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for method in methods:
            inst = _instance_means(rows, method, tw)
            rec = {"tw": tw, "method": method, "n": len(inst)}
            for c in cols:
                rec[c] = mean([v[c] for v in inst.values()])
            out.append(rec)
            md.append(
                f"| {method} | {_fmt(rec['k'])} | {_fmt(rec['distance'],1)} | {_fmt(rec['duration'],1)} | "
                f"{_fmt(rec['charged_energy'],1)} | {_fmt(rec['station_waiting'],1)} | "
                f"{_fmt(rec['charging_time'],1)} | {_fmt(rec['charging_expense'],1)} | {_fmt(rec['runtime_s'])} |"
            )
        md.append("")
    _write_csv(dirs["ablation"] / "table04_ablation.csv", out)
    (dirs["ablation"] / "table04_ablation.md").write_text("\n".join(md), encoding="utf-8")
    tex = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
           "TW & method & $K$ & $D$ & $T$ & energy & wait & $t_{ch}$ & $C$ \\\\", "\\midrule"]
    for r in out:
        tex.append(
            f"{r['tw']} & {r['method']} & {_fmt(r['k'])} & {_fmt(r['distance'],1)} & {_fmt(r['duration'],1)} & "
            f"{_fmt(r['charged_energy'],1)} & {_fmt(r['station_waiting'],1)} & {_fmt(r['charging_time'],1)} & {_fmt(r['charging_expense'],1)} \\\\"
        )
    tex += ["\\bottomrule", "\\end{tabular}", ""]
    (dirs["ablation"] / "table04_ablation.tex").write_text("\n".join(tex), encoding="utf-8")


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

Table 2. Mean fleet size, distance, duration, and runtime. Stochastic methods are instance-level means over seeds, then averaged over instances.

Table 3. Paired ALNS-PR vs ALNS-HCO comparison. Relative change is the mean of instance-level $100(M_{{HCO}}-M_{{PR}})/M_{{PR}}$. Negative values are reductions. Wins/ties/losses use tolerance {TIE_TOL}. CI is a paired bootstrap 95% interval on that mean percentage. $p$ is Wilcoxon signed-rank on paired differences.

Table 4. Ablation: full recharge (ALNS-FR), partial recharge (ALNS-PR), HCO on frozen sequences (ALNS-PR+HCO), and the full method (ALNS-HCO).

Table 5. Mean paired percentage change of ALNS-HCO vs ALNS-PR by Solomon-style family.

Figure 1. Two-phase ALNS-HCO framework (author drawing). Phase I is ALNS-PR and returns $X^{{\\mathrm{{PR}}}}$ with $K^*=K(X^{{\\mathrm{{PR}}}})$. Phase II is charging-aware ALNS with frozen $K=K^*$, HCO inside the search, acceptance/rejection, and output $X^*$. This figure is not generated by the result compiler.

Figure 2. Illustrative HCO charging decision. A customer arc is battery-infeasible; a restricted set of stations $\\{{s_1,s_2,s_3\\}}$ is scored by detour, $\\bar P_s$, $\\bar W_s$, and $r_s$; HCO inserts a station $s^*$, assigns a just-enough partial charge $e_s$, and replays route feasibility. The diagram is a methodological illustration, not an experimental result.

Figure 3. Charging ablation profile. Each series is the instance-mean metric indexed to ALNS-FR $=100$, for ALNS-FR $\\to$ ALNS-PR $\\to$ ALNS-PR+HCO $\\to$ ALNS-HCO. The large drop occurs when partial charging is introduced. Fixed-route HCO changes little; full Phase-II ALNS-HCO then trades waiting time against energy and charging expense.

Figure 4. Paired ALNS-HCO versus ALNS-PR. Dots are the mean of instance-level $100(M_{{\\mathrm{{HCO}}}}-M_{{\\mathrm{{PR}}}})/M_{{\\mathrm{{PR}}}}$; whiskers are paired bootstrap 95% confidence intervals. The vertical line is zero. Negative values are reductions.

Figure 5. Family-level charging effects of ALNS-HCO versus ALNS-PR for $N_{{\\mathrm{{ch}}}}$, $T_{{\\mathrm{{wait}}}}$, and $C_{{\\mathrm{{ch}}}}$. Blue is a reduction.
"""
    (dirs["paper"] / "captions.md").write_text(text, encoding="utf-8")


def _summary(dirs, rows: list[dict]) -> None:
    lines = ["# Hanif ALNS-HCO results (factual)", ""]
    feas = [r for r in rows if r.get("feasible")]
    lines.append(f"- Master rows: {len(rows)}; feasible rows: {len(feas)}.")
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
        lines += [
            f"## {tw.title()} TW (n={len(keys)} paired instances)",
            f"- ALNS-PR mean K: {_fmt(m(METHOD_ALNS_PR, 'k'))}",
            f"- ALNS-HCO mean K: {_fmt(m(METHOD_ALNS_HCO, 'k'))}",
            f"- ALNS-PR mean distance: {_fmt(m(METHOD_ALNS_PR, 'distance'), 1)}",
            f"- ALNS-HCO mean distance: {_fmt(m(METHOD_ALNS_HCO, 'distance'), 1)}",
            f"- Charging-cost mean paired %: {_fmt(mean([p for p in pcts if p is not None]))} (95% CI [{_fmt(lo)}, {_fmt(hi)}])",
            f"- Charging-cost wins/ties/losses: {wtl[0]}/{wtl[1]}/{wtl[2]}",
            f"- ALNS-PR mean runtime (s): {_fmt(m(METHOD_ALNS_PR, 'runtime_s'))}",
            f"- ALNS-HCO mean runtime (s): {_fmt(m(METHOD_ALNS_HCO, 'runtime_s'))}",
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
