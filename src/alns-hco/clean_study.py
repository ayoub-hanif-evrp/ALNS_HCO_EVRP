"""Compile validation, statistics, tables, figures, and research-question answers."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.alns_hco.hyperparameters import HYPERPARAMETERS
from src.common.config import (
    FAMILIES, METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
    METHOD_PHASE1_PR15, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET, METHOD_PHASE2_NOHCO,
    METHOD_PHASE2_PLAIN, ROUTING_METHODS, SEEDS, TIE_TOL, TW_VARIANTS, experiment_config,
)
from src.common.paths import hanif_dirs
from src.common.stats import bootstrap_mean_ci, mean, median, pct_change, wilcoxon_p, wins_ties_losses

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 300, "font.size": 9})


def _load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        rows = []
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if v in ("", None):
                    r[k] = None
                    continue
                if k in {"feasible", "k_frozen_ok"}:
                    r[k] = str(v).lower() in {"1", "true", "yes"}
                else:
                    try:
                        if "." in str(v) or "e" in str(v).lower():
                            r[k] = float(v)
                        else:
                            r[k] = int(v)
                    except (TypeError, ValueError):
                        pass
            rows.append(r)
        return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _instance_means(rows: list[dict], method: str, tw: str) -> dict[str, dict]:
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
            **{
                m: mean([x[m] for x in items if x.get(m) is not None])
                for m in (
                    "k", "distance", "duration", "charging_visits", "charged_energy",
                    "station_waiting", "charging_time", "charging_expense", "runtime_s",
                    "Cch_over_Ech", "Twait_over_Nch", "Tch_over_Ech",
                )
            },
        }
    return out


def _paired_table(rows, method_a, method_b, tw, metrics) -> list[dict]:
    a = _instance_means(rows, method_a, tw)
    b = _instance_means(rows, method_b, tw)
    keys = sorted(set(a) & set(b))
    out = []
    for metric in metrics:
        aa = [a[k][metric] for k in keys]
        bb = [b[k][metric] for k in keys]
        pcts = [pct_change(bi, ai) for ai, bi in zip(aa, bb)]
        pcts_f = [p for p in pcts if p is not None]
        diffs = [bi - ai for ai, bi in zip(aa, bb) if ai is not None and bi is not None]
        wtl = wins_ties_losses(bb, aa, tol=TIE_TOL)
        lo, hi = bootstrap_mean_ci(pcts_f, seed=0)
        out.append({
            "tw": tw,
            "baseline": method_a,
            "treatment": method_b,
            "metric": metric,
            "n": len(keys),
            "mean_A": mean(aa),
            "mean_B": mean(bb),
            "mean_abs_diff": mean(diffs),
            "mean_paired_pct": mean(pcts_f),
            "median_paired_pct": median(pcts_f),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "wins": wtl[0], "ties": wtl[1], "losses": wtl[2],
            "wilcoxon_p": wilcoxon_p(diffs),
            "pct_definition": "mean of instance-level 100*(B-A)/A",
        })
    return out


def validate(rows, station_rows, dirs) -> list[str]:
    errs = []
    if any("schneider" in str(r.get("instance", "")).lower() for r in rows):
        errs.append("Schneider instance found in solution_results")
    n_err = sum(1 for r in rows if r.get("error"))
    if n_err:
        errs.append(f"{n_err} failed jobs logged with error=...")

    # Shared checksum identity across dependent methods
    by_key = defaultdict(dict)
    dep = {
        METHOD_PHASE1_PR15, METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
        METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
    }
    for r in rows:
        if r.get("method") in dep and r.get("phase1_checksum"):
            by_key[(r["instance"], r["tw"], r["seed"])][r["method"]] = r["phase1_checksum"]
    for key, mp in by_key.items():
        vals = set(mp.values())
        if len(vals) > 1:
            errs.append(f"Phase-I checksum mismatch across branches {key}: {mp}")

    # Fixed-route / Phase-II must keep K = K*
    for r in rows:
        if r.get("method") in {
            METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
            METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
        }:
            if r.get("k") is not None and r.get("k_star") is not None and int(r["k"]) != int(r["k_star"]):
                errs.append(f"K!=K* for {r['method']} {r['instance']} seed={r['seed']}")
            if r.get("k_frozen_ok") is False:
                errs.append(f"k_frozen_ok=False for {r['method']} {r['instance']} seed={r['seed']}")
            if r.get("feasible") is False:
                errs.append(f"infeasible final solution for {r['method']} {r['instance']} seed={r['seed']}")

    # Mode labels
    for r in rows:
        if r.get("method") in {METHOD_FIXED_DETOUR, METHOD_PHASE2_DETOUR} and r.get("hco_mode") not in (None, "detour"):
            errs.append(f"Detour method has hco_mode={r.get('hco_mode')}")
        if r.get("method") in {METHOD_FIXED_HET, METHOD_PHASE2_HET} and r.get("hco_mode") not in (None, "heterogeneous"):
            errs.append(f"Heterogeneous method has hco_mode={r.get('hco_mode')}")
        if r.get("method") == METHOD_PHASE2_PLAIN and r.get("hco_mode") not in (None, "plain"):
            errs.append(f"PhaseII-Plain has hco_mode={r.get('hco_mode')}")
        if r.get("method") == METHOD_PHASE2_NOHCO and r.get("hco_mode") not in (None, "nohco"):
            errs.append(f"PhaseII-NoHCO has hco_mode={r.get('hco_mode')}")

    # Duplicate method/instance/seed rows
    seen = set()
    for r in rows:
        key = (r.get("instance"), r.get("tw"), r.get("method"), r.get("seed"))
        if key in seen:
            errs.append(f"duplicate row {key}")
        seen.add(key)

    # Regression tests for HCO bugs (must pass before trusting tables)
    import subprocess
    import sys
    test_cmd = [sys.executable, "-m", "pytest", "tests/test_hco_final_fix.py", "-q", "--tb=line"]
    try:
        proc = subprocess.run(
            test_cmd, cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=120,
        )
        test_ok = proc.returncode == 0
        test_summary = (proc.stdout or "").strip().splitlines()[-1:] or ["(no pytest output)"]
        test_block = [
            "",
            "## HCO regression tests (tests/test_hco_final_fix.py)",
            f"command: {' '.join(test_cmd)}",
            f"exit_code: {proc.returncode}",
            f"summary: {test_summary[0]}",
        ]
        if not test_ok:
            errs.append("HCO regression tests failed — see validation_report.txt")
            test_block.append((proc.stdout or "")[-2000:])
            test_block.append((proc.stderr or "")[-1000:])
    except Exception as exc:  # noqa: BLE001
        test_block = ["", "## HCO regression tests", f"failed to run: {exc!r}"]
        errs.append(f"HCO regression tests could not run: {exc!r}")

    report = []
    report.append(f"solution rows: {len(rows)}")
    report.append(f"station visit rows: {len(station_rows)}")
    report.append(f"failed jobs: {n_err}")
    report.append(f"errors found: {len(errs)}")
    report.extend(errs[:200] if errs else ["all checks passed"])
    report.extend(test_block)
    (dirs["validation"] / "validation_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    return errs


def summarize_hco_activity(dirs) -> None:
    path = dirs["raw"] / "hco_activity.csv"
    rows = _load_csv(path)
    if not rows:
        return
    # By TW and by family
    groups = defaultdict(list)
    for r in rows:
        groups[("tw", r.get("tw"), r.get("method"))].append(r)
        groups[("family", r.get("family"), r.get("method"))].append(r)
        groups[("tw_family", f"{r.get('tw')}|{r.get('family')}", r.get("method"))].append(r)

    metrics = [
        "hco_calls", "calls", "feasible_hco_calls", "feasible_plans",
        "changed_calls", "unchanged_calls", "plans_changed",
        "cache_hits", "cache_misses",
        "station_insertions", "station_replacements", "station_removals",
        "charge_quantity_changes",
        "objective_improvements", "objective_ties", "rejected_worse_plans",
        "nonempty_candidate_sets", "candidates_evaluated", "candidates_feasible",
        "eval_time_s", "cache_hit_rate", "activation_rate", "improvement_rate",
    ]
    out = []
    for (scope, key, method), items in sorted(groups.items()):
        rec = {"scope": scope, "key": key, "method": method, "n": len(items)}
        for m in metrics:
            vals = [x.get(m) for x in items if x.get(m) is not None]
            rec[f"mean_{m}"] = mean(vals)
        out.append(rec)
    _write_csv(dirs["analysis"] / "hco_activation_analysis.csv", out)


def charging_indicators(rows, station_rows, dirs) -> None:
    # From station visits
    groups = defaultdict(list)
    for r in station_rows:
        key = (r.get("instance"), r.get("tw"), r.get("method"), r.get("seed"))
        groups[key].append(r)
    out = []
    for (inst, tw, method, seed), visits in groups.items():
        if not visits:
            continue
        n = len(visits)
        e = [float(v.get("charged_energy_es") or 0.0) for v in visits]
        p = [float(v.get("expected_price") or 0.0) for v in visits]
        w = [float(v.get("expected_waiting") or 0.0) for v in visits]
        rt = [float(v.get("charging_rate") or 0.0) for v in visits]
        e_sum = sum(e) or 1e-12
        out.append({
            "instance": inst, "tw": tw, "method": method, "seed": seed,
            "n_visits": n,
            "P_used_visit_avg": mean(p),
            "W_used_visit_avg": mean(w),
            "r_used_visit_avg": mean(rt),
            "P_used_energy_weighted": sum(pi * ei for pi, ei in zip(p, e)) / e_sum,
            "W_used_energy_weighted": sum(wi * ei for wi, ei in zip(w, e)) / e_sum,
            "r_used_energy_weighted": sum(ri * ei for ri, ei in zip(rt, e)) / e_sum,
        })
    # Merge solution-level ratios
    sol = {(r["instance"], r["tw"], r["method"], r["seed"]): r for r in rows}
    for rec in out:
        s = sol.get((rec["instance"], rec["tw"], rec["method"], rec["seed"]))
        if s:
            rec["Cch_over_Ech"] = s.get("Cch_over_Ech")
            rec["Twait_over_Nch"] = s.get("Twait_over_Nch")
            rec["Tch_over_Ech"] = s.get("Tch_over_Ech")
    _write_csv(dirs["processed"] / "charging_indicators.csv", out)


def _save_png(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _forest(ax, records, metrics, xlabel):
    n = len(metrics)
    ys = list(range(n - 1, -1, -1))
    for y, metric in zip(ys, metrics):
        rec = next((r for r in records if r["metric"] == metric), None)
        if not rec or rec["mean_paired_pct"] is None:
            continue
        m, lo, hi = rec["mean_paired_pct"], rec["ci95_lo"], rec["ci95_hi"]
        color = "#4c78a8"
        if hi is not None and hi < 0:
            color = "#4c78a8"
        elif lo is not None and lo > 0:
            color = "#c45c26"
        else:
            color = "#6b7280"
        xerr = None if lo is None or hi is None else [[m - lo], [hi - m]]
        ax.errorbar(m, y, xerr=xerr, fmt="o", color=color, ecolor=color, capsize=3)
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(metrics)
    ax.set_xlabel(xlabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figures(rows, pair_tables, dirs) -> None:
    metrics_charge = ["charged_energy", "charging_time", "charging_expense", "charging_visits", "station_waiting"]
    # Figure A: FR vs PR
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, tw in zip(axes, TW_VARIANTS):
        recs = [r for r in pair_tables if r["baseline"] == METHOD_ALNS_FR and r["treatment"] == METHOD_ALNS_PR and r["tw"] == tw]
        _forest(ax, recs, metrics_charge, r"$100(M_{PR}-M_{FR})/M_{FR}$ (%)")
        ax.set_title("Original TW" if tw == "original" else "Relaxed TW")
    fig.tight_layout()
    _save_png(fig, dirs["figures"] / "figA_partial_recharging_effect.png")

    # Figure B: fixed-route detour vs het
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)
    mets = ["charging_expense", "station_waiting", "charging_time", "charging_visits"]
    for ax, tw in zip(axes, TW_VARIANTS):
        recs = [r for r in pair_tables if r["baseline"] == METHOD_FIXED_DETOUR and r["treatment"] == METHOD_FIXED_HET and r["tw"] == tw]
        _forest(ax, recs, mets, r"$100(M_{Het}-M_{Det})/M_{Det}$ (%)")
        ax.set_title("Original TW" if tw == "original" else "Relaxed TW")
    fig.tight_layout()
    _save_png(fig, dirs["figures"] / "figB_fixed_route_heterogeneity.png")

    # Figure D: PhaseI -> PhaseII Het
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, tw in zip(axes, TW_VARIANTS):
        recs = [r for r in pair_tables if r["baseline"] == METHOD_PHASE1_PR15 and r["treatment"] == METHOD_PHASE2_HET and r["tw"] == tw]
        _forest(ax, recs, ["distance", "charging_expense", "station_waiting", "charging_time", "charging_visits"],
                r"$100(M_{P2}-M_{P1})/M_{P1}$ (%)")
        ax.set_title("Original TW" if tw == "original" else "Relaxed TW")
    fig.tight_layout()
    _save_png(fig, dirs["figures"] / "figD_phase1_to_phase2.png")

    # Figure E: PhaseII-NoHCO vs Heterogeneous (main HCO ablation)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, tw in zip(axes, TW_VARIANTS):
        recs = [r for r in pair_tables if r["baseline"] == METHOD_PHASE2_NOHCO and r["treatment"] == METHOD_PHASE2_HET and r["tw"] == tw]
        _forest(ax, recs, ["distance", "charging_expense", "station_waiting", "charging_time", "charging_visits"],
                r"$100(M_{HCO}-M_{NoHCO})/M_{NoHCO}$ (%)")
        ax.set_title("Original TW" if tw == "original" else "Relaxed TW")
    fig.tight_layout()
    _save_png(fig, dirs["figures"] / "figE_nohco_vs_hco.png")

    # Supplement: PhaseII detour vs het
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), sharey=True)
    for ax, tw in zip(axes, TW_VARIANTS):
        recs = [r for r in pair_tables if r["baseline"] == METHOD_PHASE2_DETOUR and r["treatment"] == METHOD_PHASE2_HET and r["tw"] == tw]
        _forest(ax, recs, ["distance", "charging_expense", "station_waiting", "charging_time", "charging_visits"],
                r"$100(M_{Het}-M_{Det})/M_{Det}$ (%)")
        ax.set_title("Original TW" if tw == "original" else "Relaxed TW")
    fig.tight_layout()
    _save_png(fig, dirs["figures"] / "figS1_phase2_detour_vs_heterogeneous.png")
    # Remove old dual-E naming if present
    legacy = dirs["figures"] / "figE_phase2_detour_vs_heterogeneous.png"
    if legacy.is_file():
        legacy.unlink()


def _rq_line(pair_rows, metric: str) -> str:
    bits = []
    for tw in TW_VARIANTS:
        r = next((x for x in pair_rows if x["tw"] == tw and x["metric"] == metric), None)
        if not r or r.get("mean_paired_pct") is None:
            bits.append(f"{tw}: n/a")
        else:
            lo = r.get("ci95_lo")
            hi = r.get("ci95_hi")
            bits.append(
                f"{tw}: mean %={r['mean_paired_pct']:.2f} "
                f"CI=[{lo},{hi}] W/T/L={r['wins']}/{r['ties']}/{r['losses']}"
            )
    return "; ".join(bits)


def write_research_questions(pair_tables, indicator_rows, dirs) -> None:
    fr_pr = [r for r in pair_tables if r["baseline"] == METHOD_ALNS_FR and r["treatment"] == METHOD_ALNS_PR]
    fix = [r for r in pair_tables if r["baseline"] == METHOD_FIXED_DETOUR and r["treatment"] == METHOD_FIXED_HET]
    p2 = [r for r in pair_tables if r["baseline"] == METHOD_PHASE2_DETOUR and r["treatment"] == METHOD_PHASE2_HET]
    nohco = [r for r in pair_tables if r["baseline"] == METHOD_PHASE2_NOHCO and r["treatment"] == METHOD_PHASE2_HET]
    p1p2 = [r for r in pair_tables if r["baseline"] == METHOD_PHASE1_PR15 and r["treatment"] == METHOD_PHASE2_HET]

    lines = [
        "# Research questions (data-driven, post-HCO-fix)",
        "",
        "Negative mean paired % = reduction. Results are reported as computed; no wording is softened.",
        "",
        "## RQ1. Partial charging (ALNS-FR → ALNS-PR)",
        f"- charged_energy: {_rq_line(fr_pr, 'charged_energy')}",
        f"- charging_time: {_rq_line(fr_pr, 'charging_time')}",
        f"- charging_expense: {_rq_line(fr_pr, 'charging_expense')}",
        "",
        "## RQ2. Heterogeneous information on fixed routes (Detour → Heterogeneous)",
        f"- charging_expense: {_rq_line(fix, 'charging_expense')}",
        f"- station_waiting: {_rq_line(fix, 'station_waiting')}",
        f"- charging_time: {_rq_line(fix, 'charging_time')}",
        f"- charging_visits: {_rq_line(fix, 'charging_visits')}",
        "",
        "## RQ3. Is HCO useful beyond matched Phase-II search? (PhaseII-NoHCO → Heterogeneous)",
        f"- charging_expense: {_rq_line(nohco, 'charging_expense')}",
        f"- station_waiting: {_rq_line(nohco, 'station_waiting')}",
        f"- distance: {_rq_line(nohco, 'distance')}",
        "",
        "## RQ4. Does heterogeneity improve Phase II? (Detour → Heterogeneous)",
        f"- charging_expense: {_rq_line(p2, 'charging_expense')}",
        f"- station_waiting: {_rq_line(p2, 'station_waiting')}",
        f"- distance: {_rq_line(p2, 'distance')}",
        "",
        "## RQ5. What does Phase II add to the shared starting point? (X_PR_15 → PhaseII-Heterogeneous)",
        f"- charging_expense: {_rq_line(p1p2, 'charging_expense')}",
        f"- station_waiting: {_rq_line(p1p2, 'station_waiting')}",
        f"- distance: {_rq_line(p1p2, 'distance')}",
        "",
        "## RQ6. What stations does HCO actually choose?",
        "See `outputs/tables/table_station_selection.csv` and `outputs/processed/charging_indicators.csv`.",
        "",
        "## RQ7. When does HCO activate?",
        "See `outputs/raw/hco_activity.csv` and `outputs/analysis/hco_activation_analysis.csv`.",
        "activation_rate = changed_calls / feasible_hco_calls; improvement_rate = objective_improvements / feasible_hco_calls.",
        "",
        "## RQ6 family note",
        "See `outputs/tables/table_family_results.csv`.",
        "",
    ]
    # Station attribute note
    for tw in TW_VARIANTS:
        det = [r for r in indicator_rows if r.get("tw") == tw and r.get("method") == METHOD_FIXED_DETOUR]
        het = [r for r in indicator_rows if r.get("tw") == tw and r.get("method") == METHOD_FIXED_HET]
        if det and het:
            lines.append(
                f"- Fixed-route {tw}: mean P_used Detour={mean([r['P_used_visit_avg'] for r in det]):.4f}, "
                f"Het={mean([r['P_used_visit_avg'] for r in het]):.4f}; "
                f"W_used Detour={mean([r['W_used_visit_avg'] for r in det]):.4f}, "
                f"Het={mean([r['W_used_visit_avg'] for r in het]):.4f}; "
                f"r_used Detour={mean([r['r_used_visit_avg'] for r in det]):.4f}, "
                f"Het={mean([r['r_used_visit_avg'] for r in het]):.4f}."
            )
    (dirs["analysis"] / "research_questions.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_report(rows, errs, dirs) -> None:
    n_ok = sum(1 for r in rows if not r.get("error") and r.get("feasible"))
    n_fail = sum(1 for r in rows if r.get("error"))
    cfg = experiment_config()
    frozen = {}
    fp = dirs["config"] / "frozen_experiment_config.json"
    if fp.is_file():
        frozen = json.loads(fp.read_text(encoding="utf-8"))
    text = f"""# Final experiment report — corrected Hanif ALNS-HCO study

## Code changes (final HCO fix)
- Explicit charging plans `(s, e_s)`; no `None → full recharge` inside HCO.
- Full sequential battery propagation through all prior charging stops.
- **HCO-Detour is fully geometry-only:** candidate ranking by `(detour, station_id)` and
  non-worsening acceptance by `(D, N_ch, station-id sequence)` — never uses
  bar-P / bar-W / r / C_ch / T_wait / T_ch.
- Heterogeneous / Phase-II charging criterion: lex `(C_ch, T_wait, T_ch, D)` at `K=K*`.
- Phase-II repair shortlists insertions, then HCO-evaluates each (or NoHCO without oracle).
- Matched control: `PhaseII-NoHCO` (same Phase-II operators as Het, no oracle).
- Corrected HCO activity from explicit input→output plan diffs.
- Cache keys: `(customers, mode, k)`; Detour and Heterogeneous never share entries.

## Frozen parameters
Embedded below from `outputs/config/frozen_experiment_config.json` at compile time
(single source of truth — do not mix with older report snapshots).

- **ε = {frozen.get("phase2_distance_tolerance", cfg.get("phase2_distance_tolerance"))}**:
  conservative tie choice among fully feasible development-subset candidates
  `{{0, 0.01, 0.02, 0.05}}` (all identical on the 12-case subset). **Not** claimed superior.
- **Time allocation 15/15**: equal split of the common 30 s budget.
  Development sensitivity for 10/20, 15/15, 20/10 also tied at mean ΔCch = 0;
  **not** claimed optimal. See `outputs/analysis/time_allocation_sensitivity.csv`.

```json
{json.dumps(frozen or cfg, indent=2)}
```

## Validation
- Failed jobs: {n_fail}
- Feasible successful rows: {n_ok}
- Validation errors listed: {len(errs)}
- See `outputs/validation/validation_report.txt` (includes `pytest tests/test_hco_final_fix.py`).

## Required tables
- `outputs/tables/table_overall_methods.csv`
- `outputs/tables/table_partial_recharging.csv`
- `outputs/tables/table_fixed_route_heterogeneity.csv`
- `outputs/tables/table_phase2_nohco_vs_hco.csv`
- `outputs/tables/table_phase2_detour_vs_heterogeneous.csv`
- `outputs/tables/table_phase1_to_phase2.csv`
- `outputs/tables/table_station_selection.csv`
- `outputs/tables/table_family_results.csv`

## Required figures (PNG only)
- `outputs/figures/figA_partial_recharging_effect.png`
- `outputs/figures/figB_fixed_route_heterogeneity.png`
- `outputs/figures/figC_station_characteristics.png`
- `outputs/figures/figD_phase1_to_phase2.png`
- `outputs/figures/figE_nohco_vs_hco.png`
- `outputs/figures/figS1_phase2_detour_vs_heterogeneous.png` (supplement)
- `outputs/figures/figF_family_heterogeneous_effect.png`

## Research questions
See `outputs/analysis/research_questions.md`. Report negative results as negative.
"""
    (dirs["root"] / "final_experiment_report.md").write_text(text, encoding="utf-8")


def compile_clean_study() -> None:
    dirs = hanif_dirs()
    rows = _load_csv(dirs["raw"] / "solution_results.csv")
    if not rows:
        rows = _load_csv(dirs["raw"] / "master_results.csv")
    station_rows = _load_csv(dirs["raw"] / "station_visits.csv")

    (dirs["statistics"] / "statistical_method.md").write_text(
        r"""# Statistical method

1. For stochastic methods, average the five seeds per (instance, TW, method).
2. Paired comparisons use the 60 instances within each TW regime that have both methods.
3. Relative change for metric M is the **mean of instance-level** percentages
   \(100(M_B - M_A)/M_A\). Negative = reduction.
4. 95% CI: paired bootstrap of that mean percentage (seed=0).
5. Wins/ties/losses: lower is better, tolerance 1e-6.
6. Wilcoxon signed-rank on absolute paired differences.
7. Aggregate ratio-of-means percentages are **not** mixed into the mean-paired-% columns.
""",
        encoding="utf-8",
    )

    errs = validate(rows, station_rows, dirs)
    charging_indicators(rows, station_rows, dirs)
    summarize_hco_activity(dirs)
    indicators = _load_csv(dirs["processed"] / "charging_indicators.csv")

    metrics = [
        "k", "distance", "duration", "charging_visits", "charged_energy",
        "station_waiting", "charging_time", "charging_expense",
    ]
    pair_all = []
    for tw in TW_VARIANTS:
        pair_all += _paired_table(rows, METHOD_ALNS_FR, METHOD_ALNS_PR, tw, metrics)
        pair_all += _paired_table(rows, METHOD_FIXED_DETOUR, METHOD_FIXED_HET, tw, metrics)
        pair_all += _paired_table(rows, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET, tw, metrics)
        pair_all += _paired_table(rows, METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, tw, metrics)
        pair_all += _paired_table(rows, METHOD_PHASE1_PR15, METHOD_PHASE2_HET, tw, metrics)
        pair_all += _paired_table(rows, METHOD_PHASE1_PR15, METHOD_PHASE2_PLAIN, tw, metrics)
        pair_all += _paired_table(rows, METHOD_PHASE2_PLAIN, METHOD_PHASE2_HET, tw, metrics)

    _write_csv(dirs["statistics"] / "paired_comparisons.csv", pair_all)
    _write_csv(dirs["tables"] / "table_partial_recharging.csv",
               [r for r in pair_all if r["baseline"] == METHOD_ALNS_FR])
    _write_csv(dirs["tables"] / "table_fixed_route_heterogeneity.csv",
               [r for r in pair_all if r["baseline"] == METHOD_FIXED_DETOUR])
    _write_csv(dirs["tables"] / "table_phase2_detour_vs_heterogeneous.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE2_DETOUR and r["treatment"] == METHOD_PHASE2_HET])
    _write_csv(dirs["tables"] / "table_phase2_nohco_vs_hco.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE2_NOHCO and r["treatment"] == METHOD_PHASE2_HET])
    _write_csv(dirs["tables"] / "table_phase1_to_phase2.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE1_PR15 and r["treatment"] == METHOD_PHASE2_HET])
    _write_csv(dirs["tables"] / "table_phase1_to_phase2_plain.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE1_PR15 and r["treatment"] == METHOD_PHASE2_PLAIN])
    _write_csv(dirs["tables"] / "table_phase2_plain_vs_heterogeneous.csv",
               [r for r in pair_all if r["baseline"] == METHOD_PHASE2_PLAIN and r["treatment"] == METHOD_PHASE2_HET])

    # Overall methods
    overall = []
    for tw in TW_VARIANTS:
        for method in list(ROUTING_METHODS) + [
            METHOD_PHASE1_PR15, METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
            METHOD_PHASE2_NOHCO, METHOD_PHASE2_PLAIN, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
        ]:
            inst = _instance_means(rows, method, tw)
            if not inst:
                continue
            overall.append({
                "tw": tw, "method": method, "n_instances": len(inst),
                **{f"mean_{m}": mean([v[m] for v in inst.values()]) for m in metrics + ["runtime_s"]},
            })
    _write_csv(dirs["tables"] / "table_overall_methods.csv", overall)

    # Station selection
    st_tab = []
    for tw in TW_VARIANTS:
        for method in (
            METHOD_FIXED_DETOUR, METHOD_FIXED_HET,
            METHOD_PHASE2_NOHCO, METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET,
        ):
            sub = [r for r in indicators if r.get("tw") == tw and r.get("method") == method]
            if not sub:
                continue
            st_tab.append({
                "tw": tw, "method": method, "n": len(sub),
                "mean_P_used": mean([r["P_used_visit_avg"] for r in sub]),
                "mean_W_used": mean([r["W_used_visit_avg"] for r in sub]),
                "mean_r_used": mean([r["r_used_visit_avg"] for r in sub]),
                "mean_P_energy_w": mean([r["P_used_energy_weighted"] for r in sub]),
                "mean_W_energy_w": mean([r["W_used_energy_weighted"] for r in sub]),
                "mean_r_energy_w": mean([r["r_used_energy_weighted"] for r in sub]),
            })
    _write_csv(dirs["tables"] / "table_station_selection.csv", st_tab)

    # Family table
    fam_rows = []
    for tw in TW_VARIANTS:
        for fam in FAMILIES:
            for baseline, treatment, tag in (
                (METHOD_FIXED_DETOUR, METHOD_FIXED_HET, "fixed_route"),
                (METHOD_PHASE2_DETOUR, METHOD_PHASE2_HET, "phase2_detour_het"),
                (METHOD_PHASE2_NOHCO, METHOD_PHASE2_HET, "phase2_nohco_hco"),
                (METHOD_PHASE1_PR15, METHOD_PHASE2_HET, "phase1_to_phase2"),
            ):
                a = _instance_means(rows, baseline, tw)
                b = _instance_means(rows, treatment, tw)
                keys = [k for k in set(a) & set(b) if a[k].get("family") == fam]
                rec = {"tw": tw, "family": fam, "comparison": tag, "n": len(keys)}
                for m in ("charging_expense", "station_waiting", "charging_time", "charged_energy", "distance", "k", "charging_visits"):
                    pcts = [pct_change(b[k][m], a[k][m]) for k in keys]
                    rec[m] = mean([p for p in pcts if p is not None])
                fam_rows.append(rec)
    _write_csv(dirs["tables"] / "table_family_results.csv", fam_rows)

    if rows:
        make_figures(rows, pair_all, dirs)
        # Figure C station characteristics
        if st_tab:
            fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.4))
            labels = ["P_used", "W_used", "r_used"]
            keys = ["mean_P_used", "mean_W_used", "mean_r_used"]
            for ax, lab, key in zip(axes, labels, keys):
                for tw, hatch in zip(TW_VARIANTS, ("", "//")):
                    d = next((r for r in st_tab if r["tw"] == tw and r["method"] == METHOD_FIXED_DETOUR), None)
                    h = next((r for r in st_tab if r["tw"] == tw and r["method"] == METHOD_FIXED_HET), None)
                    if not d or not h:
                        continue
                    ax.bar([0 + (0 if tw == "original" else 0.1)], [d[key]], width=0.35, color="#6b7280",
                           label=f"Detour {tw}" if lab == "P_used" else None)
                    ax.bar([1 + (0 if tw == "original" else 0.1)], [h[key]], width=0.35, color="#c45c26",
                           label=f"Het {tw}" if lab == "P_used" else None)
                ax.set_xticks([0, 1])
                ax.set_xticklabels(["Detour", "Het"])
                ax.set_title(lab)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            fig.tight_layout()
            _save_png(fig, dirs["figures"] / "figC_station_characteristics.png")

        # Figure F family heatmap for fixed-route charging expense
        import numpy as np
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4))
        for ax, tw in zip(axes, TW_VARIANTS):
            mat = []
            for fam in FAMILIES:
                row = []
                for m in ("charging_expense", "station_waiting", "charging_visits"):
                    rec = next((r for r in fam_rows if r["tw"] == tw and r["family"] == fam and r["comparison"] == "fixed_route"), None)
                    row.append(0.0 if not rec or rec[m] is None else float(rec[m]))
                mat.append(row)
            arr = np.array(mat, dtype=float)
            vmax = max(8.0, float(np.max(np.abs(arr))) if arr.size else 8.0)
            im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(3))
            ax.set_xticklabels([r"$C_{ch}$", r"$T_{wait}$", r"$N_{ch}$"])
            ax.set_yticks(range(len(FAMILIES)))
            ax.set_yticklabels(FAMILIES)
            ax.set_title("Original TW" if tw == "original" else "Relaxed TW")
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
        _save_png(fig, dirs["figures"] / "figF_family_heterogeneous_effect.png")

    write_research_questions(pair_all, indicators, dirs)
    write_final_report(rows, errs, dirs)

    readme = """# Clean Hanif ALNS-HCO revision outputs

## Reproduce

```bash
python scripts/select_phase2_epsilon.py
python scripts/run_clean_hanif_study.py --workers 8 --fresh
```

## Method names

| Name | Meaning |
|---|---|
| GI / VNS / SA | Routing baselines |
| ALNS-FR | Full recharge ALNS, 30 s |
| ALNS-PR | Partial recharge ALNS, 30 s |
| PhaseI-PR-15 | Shared Phase-I solution X_PR_15 |
| FixedRoute-HCO-Detour | HCO on frozen customer routes; detour-only selection |
| FixedRoute-HCO-Heterogeneous | HCO on same routes; detour+price+wait+rate |
| PhaseII-Detour | 15 s charging-aware ALNS + HCO-Detour from shared Phase I |
| PhaseII-Heterogeneous | Same with HCO-Heterogeneous (final ALNS-HCO) |

## Locations

- Raw: `outputs/raw/solution_results.csv`, `station_visits.csv`, `route_results.csv`
- Shared Phase I: `outputs/shared_phase1/`
- Config: `outputs/config/frozen_experiment_config.json`
- Stats: `outputs/statistics/`
- Tables: `outputs/tables/`
- Figures: `outputs/figures/` (PNG only)
- Validation: `outputs/validation/validation_report.txt`
- RQs: `outputs/analysis/research_questions.md`
- Report: `outputs/final_experiment_report.md`

Prior manuscript package: `outputs/legacy_pre_revision/` and `archive/pre_clean_revision_*`.
"""
    (dirs["root"] / "README.md").write_text(readme, encoding="utf-8")
