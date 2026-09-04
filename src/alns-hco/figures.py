"""Manuscript figures for ALNS-HCO on the Hanif benchmark.

Each figure answers one question. Reads only the saved master CSV.
Figure 1 is the author's framework drawing and is never overwritten.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyBboxPatch, RegularPolygon

from src.common.config import (
    FAMILIES, METHOD_ALNS_FR, METHOD_ALNS_HCO, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO,
    TW_VARIANTS,
)
from src.common.paths import hanif_dirs
from src.common.stats import bootstrap_mean_ci, mean, pct_change

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "mathtext.fontset": "dejavusans",
})

C_NAVY = "#1f4e79"
C_ORANGE = "#c45c26"
C_BLUE = "#4c78a8"
C_GREEN = "#3d7a5a"
C_RED = "#b22222"
C_GRAY = "#6b7280"
ZERO = "#111111"

ABLATION_METHODS = (METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO, METHOD_ALNS_HCO)
ABLATION_LABELS = ("ALNS-FR", "ALNS-PR", "ALNS-PR+HCO", "ALNS-HCO")
ABLATION_METRICS = (
    ("charged_energy", r"$E_{\mathrm{ch}}$", C_NAVY, "o"),
    ("charging_time", r"$T_{\mathrm{ch}}$", C_BLUE, "s"),
    ("charging_expense", r"$C_{\mathrm{ch}}$", C_ORANGE, "D"),
    ("station_waiting", r"$T_{\mathrm{wait}}$", C_GREEN, "^"),
)

FOREST_METRICS = (
    ("k", r"$K$"),
    ("distance", r"$D$"),
    ("duration", r"$T$"),
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("charged_energy", r"$E_{\mathrm{ch}}$"),
    ("station_waiting", r"$T_{\mathrm{wait}}$"),
    ("charging_time", r"$T_{\mathrm{ch}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)

# Main Figure 3: clean FR → PR only (not a four-stage sequential ablation).
FR_PR_METRICS = (
    ("distance", r"$D$"),
    ("duration", r"$T$"),
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("charged_energy", r"$E_{\mathrm{ch}}$"),
    ("station_waiting", r"$T_{\mathrm{wait}}$"),
    ("charging_time", r"$T_{\mathrm{ch}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)

FAMILY_METRICS = (
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("station_waiting", r"$T_{\mathrm{wait}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)

# Dashboard figures from the previous generator. Never regenerate these.
ARCHIVE_STEMS = (
    "fig02_paired_improvements",
    "fig03_charging_mechanisms",
    "fig03_charging_cost_scatter",
    "fig03_charging_ablation",  # replaced by FR→PR forest as main Fig. 3
    "fig04_charging_mechanisms",
    "fig04_family_heatmap",
    "fig05_charging_expense_scatter",
    "fig05_family_heatmap",
    "fig06_runtime_quality",
    "figS01_runtime_distribution",
    "figS03_runtime_distribution",
    "fig06_omitted_route_soc",
)

AUTHOR_FIG01 = (
    "Figure 1 is the author's manually prepared ALNS-HCO framework drawing. "
    "The generator does not create or overwrite it."
)


def _tw_label(tw: str) -> str:
    return "Original TW" if tw == "original" else "Relaxed TW"


def _purge_vector_sidecars(dirs) -> None:
    """Keep PNG only under figures/ and figures/supplement/."""
    for root in (dirs["paper_figures"], dirs["paper_supplement"]):
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".pdf", ".svg"}:
                # Never touch an author Fig. 1 if the user placed vector originals elsewhere;
                # manuscript package is PNG-only.
                path.unlink()


def _archive_old(dirs) -> None:
    dest = dirs["paper_figures"] / "archive"
    dest.mkdir(parents=True, exist_ok=True)
    suffixes = (".png", ".svg", ".pdf", ".csv", ".md")
    search = [dirs["paper_figures"], dirs["paper_supplement"], dirs["paper_data"]]
    for stem in ARCHIVE_STEMS:
        for root in search:
            for suf in suffixes:
                path = root / f"{stem}{suf}"
                if path.is_file():
                    target = dest / path.name
                    if target.exists():
                        target.unlink()
                    shutil.move(str(path), str(target))
    # Never move or delete the author Fig. 1. Only remove agent-generated CSV notes.
    csv01 = dirs["paper_data"] / "fig01_alns_hco_framework.csv"
    if csv01.is_file():
        target = dest / "agent_fig01_alns_hco_framework.csv"
        if target.exists():
            target.unlink()
        shutil.move(str(csv01), str(target))
    note = dest / "README.md"
    note.write_text(
        "# Archived figures\n\n"
        "Dashboard-style plots and the superseded four-stage main Fig. 3 "
        "(`fig03_charging_ablation`) live here. The four-stage view, if kept, "
        "is regenerated under `figures/supplement/`.\n\n"
        "Author Figure 1 (`fig01_alns_hco_framework.png`) is never archived or overwritten.\n",
        encoding="utf-8",
    )


def _write_fig01_note(dirs) -> None:
    text = (
        "# Figure 1 — author framework\n\n"
        + AUTHOR_FIG01
        + "\n\nPlace the author PNG as `outputs/figures/fig01_alns_hco_framework.png`.\n"
        "The generator will not create, move, or replace that file.\n"
        "Do not use the old candidate `figures_candidates_png/fig01_two_phase_framework.png`.\n"
    )
    (dirs["paper_figures"] / "README_FIG01.md").write_text(text, encoding="utf-8")
    (dirs["paper"] / "README_FIG01.md").write_text(text, encoding="utf-8")


def _ci_color(lo, hi) -> str:
    if lo is not None and hi is not None and hi < 0:
        return C_BLUE
    if lo is not None and hi is not None and lo > 0:
        return C_ORANGE
    return C_GRAY


def _draw_forest(axes, records, metrics, xlabel: str) -> None:
    n = len(metrics)
    ys = list(range(n - 1, -1, -1))
    for ax, tw in zip(axes, TW_VARIANTS):
        sub = [r for r in records if r["tw"] == tw]
        for y, rec in zip(ys, sub):
            m, lo, hi = rec["mean_pct"], rec["ci95_lo"], rec["ci95_hi"]
            if m is None:
                continue
            color = _ci_color(lo, hi)
            xerr = None if lo is None or hi is None else [[m - lo], [hi - m]]
            ax.errorbar(m, y, xerr=xerr, fmt="o", color=color, ecolor=color,
                        elinewidth=1.15, capsize=3.2, ms=5.5, zorder=3)
        ax.axvline(0.0, color=ZERO, lw=1.05, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _m, lab in metrics])
        ax.set_title(_tw_label(tw))
        ax.set_xlabel(xlabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-0.55, n - 0.45)


def fig02_hco_decision(dirs) -> None:
    """Conceptual HCO insert/partial-charge/replay illustration. Not experimental."""
    from src.alns_hco.paper import _save_fig, _write_csv

    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.55))
    fig.suptitle("Illustrative example  (not an experimental result)", fontsize=9, color=C_GRAY, y=0.98)

    # (a) battery deficit
    ax = axes[0]
    ax.set_xlim(-0.4, 4.2)
    ax.set_ylim(-1.35, 1.55)
    ax.axis("off")
    ax.set_title("(a) Battery deficit", fontsize=10, pad=8)
    ax.add_patch(Circle((0.4, 0.25), 0.18, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.add_patch(Circle((3.6, 0.25), 0.18, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.text(0.4, -0.22, r"customer $i$", ha="center", va="top", fontsize=8)
    ax.text(3.6, -0.22, r"customer $i{+}1$", ha="center", va="top", fontsize=8)
    ax.annotate(
        "", xy=(3.35, 0.25), xytext=(0.65, 0.25),
        arrowprops=dict(arrowstyle="-|>", linestyle=(0, (5, 3)), color=C_RED, lw=1.4, mutation_scale=11),
    )
    ax.text(2.0, 0.62, "direct arc infeasible", ha="center", fontsize=8, color=C_RED)
    ax.text(2.0, -0.72, r"remaining SoC $<$ energy $i\to i{+}1$", ha="center", fontsize=8)
    ax.text(2.0, -1.05, "a charging stop is required", ha="center", fontsize=8, color=C_GRAY)

    # (b) three candidates
    ax = axes[1]
    ax.set_xlim(-0.5, 4.5)
    ax.set_ylim(-1.55, 1.85)
    ax.axis("off")
    ax.set_title("(b) Restricted candidates", fontsize=10, pad=8)
    ax.add_patch(Circle((0.45, 0.85), 0.16, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.add_patch(Circle((3.55, 0.85), 0.16, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.text(0.45, 1.18, r"$i$", ha="center", fontsize=8)
    ax.text(3.55, 1.18, r"$i{+}1$", ha="center", fontsize=8)
    ax.plot([0.61, 3.39], [0.85, 0.85], color="0.75", lw=0.9, ls="--")
    stations = [
        (1.15, -0.15, r"$s_1$", "long detour\n" + r"high $\bar P_s$, high $r_s$"),
        (2.00, -0.55, r"$s_2$", "short detour\n" + r"low $\bar W_s$, moderate $r_s$"),
        (2.85, -0.15, r"$s_3$", "short detour\n" + r"low $\bar P_s$, high $\bar W_s$"),
    ]
    for x, y, name, lab in stations:
        ax.add_patch(RegularPolygon((x, y), 4, radius=0.22, orientation=0.785, facecolor="white",
                                    edgecolor="black", lw=0.9, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=8, zorder=4)
        ax.plot([0.45, x], [0.85, y + 0.18], color="0.55", lw=0.7)
        ax.plot([x, 3.55], [y + 0.18, 0.85], color="0.55", lw=0.7)
        ax.text(x, y - 0.42, lab, ha="center", va="top", fontsize=6.7, color=C_GRAY, linespacing=1.25)
    ax.annotate(
        "", xy=(2.00, -0.78), xytext=(2.00, -1.22),
        arrowprops=dict(arrowstyle="-|>", color=C_ORANGE, lw=1.1, mutation_scale=9),
    )
    ax.text(2.00, 1.55, r"attributes $\bar P_s$, $\bar W_s$, $r_s$, detour", ha="center", fontsize=8)

    # (c) selection, partial charge, replay
    ax = axes[2]
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-1.45, 1.65)
    ax.axis("off")
    ax.set_title("(c) Partial charge and replay", fontsize=10, pad=8)
    ax.add_patch(Circle((0.45, 0.55), 0.16, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.add_patch(RegularPolygon((2.0, 0.55), 4, radius=0.24, orientation=0.785,
                                facecolor="#f4e6d8", edgecolor=C_ORANGE, lw=1.3, zorder=3))
    ax.add_patch(Circle((3.55, 0.55), 0.16, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.text(0.45, 0.92, r"$i$", ha="center", fontsize=8)
    ax.text(2.0, 0.55, r"$s^*$", ha="center", va="center", fontsize=8, color=C_ORANGE, fontweight="bold")
    ax.text(3.55, 0.92, r"$i{+}1$", ha="center", fontsize=8)
    ax.annotate("", xy=(1.78, 0.55), xytext=(0.61, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10))
    ax.annotate("", xy=(3.39, 0.55), xytext=(2.22, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10))
    ax.text(2.0, 1.22, r"HCO selects $s_2\equiv s^*$", ha="center", fontsize=8)
    ax.text(2.0, -0.05, r"just-enough partial charge $e_s$", ha="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((0.35, -1.18), 3.3, 0.72, boxstyle="square,pad=0",
                                facecolor="#f7f7f7", edgecolor="black", lw=0.8))
    ax.text(2.0, -0.62, "route feasibility replay", ha="center", fontsize=8, fontweight="bold")
    ax.text(2.0, -0.92, "TW, battery, and depot return", ha="center", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, dirs["paper_figures"] / "fig02_hco_decision")
    _write_csv(dirs["paper_data"] / "fig02_hco_decision.csv", [
        {"panel": "a", "content": "infeasible customer arc due to battery deficit"},
        {"panel": "b", "content": "restricted candidates s1,s2,s3 with Pbar, Wbar, r, detour"},
        {"panel": "c", "content": "HCO selects s*; just-enough partial charge e_s; feasibility replay"},
        {"note": "illustrative schematic", "content": "not computed from saved routes"},
    ])


def fig03_partial_recharging(dirs, rows: list[dict]) -> None:
    """Main Fig. 3: paired ALNS-FR → ALNS-PR forest (partial recharging effect)."""
    from src.alns_hco.paper import _instance_means, _save_fig, _write_csv

    records = []
    for tw in TW_VARIANTS:
        fr = _instance_means(rows, METHOD_ALNS_FR, tw)
        pr = _instance_means(rows, METHOD_ALNS_PR, tw)
        keys = sorted(set(fr) & set(pr))
        for metric, lab in FR_PR_METRICS:
            pcts = [pct_change(pr[k][metric], fr[k][metric]) for k in keys]
            pcts = [p for p in pcts if p is not None]
            m = mean(pcts)
            lo, hi = bootstrap_mean_ci(pcts, seed=0)
            records.append({
                "tw": tw, "metric": metric, "label": lab, "n": len(pcts),
                "mean_pct": m, "ci95_lo": lo, "ci95_hi": hi,
                "new": METHOD_ALNS_PR, "baseline": METHOD_ALNS_FR,
            })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.35), sharex=True, sharey=True)
    _draw_forest(axes, records, FR_PR_METRICS,
                 r"$100(M_{\mathrm{PR}}-M_{\mathrm{FR}})/M_{\mathrm{FR}}$ (%)")
    axes[0].text(0.02, 1.02, "negative = reduction", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=8, color=C_GRAY)
    legend = [
        Line2D([0], [0], marker="o", color=C_BLUE, lw=0, label="CI entirely < 0"),
        Line2D([0], [0], marker="o", color=C_ORANGE, lw=0, label="CI entirely > 0"),
        Line2D([0], [0], marker="o", color=C_GRAY, lw=0, label="CI crosses 0"),
    ]
    fig.legend(handles=legend, loc="upper right", frameon=False, bbox_to_anchor=(0.99, 1.02), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, dirs["paper_figures"] / "fig03_partial_recharging_effect")
    _write_csv(dirs["paper_data"] / "fig03_partial_recharging_effect.csv", records)


def figS_charging_ablation(dirs, rows: list[dict]) -> None:
    """Supplement: four-stage index plot. Not a controlled sequential ablation."""
    from src.alns_hco.paper import _instance_means, _save_fig, _write_csv

    dirs["paper_supplement"].mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), sharey=True)
    data_rows = []
    xs = list(range(len(ABLATION_METHODS)))
    for ax, tw in zip(axes, TW_VARIANTS):
        base = _instance_means(rows, METHOD_ALNS_FR, tw)
        series = {method: _instance_means(rows, method, tw) for method in ABLATION_METHODS}
        keys = sorted(set(base) & set.intersection(*(set(series[m]) for m in ABLATION_METHODS)))
        for metric, lab, color, marker in ABLATION_METRICS:
            fr_mean = mean([base[k][metric] for k in keys])
            idx = []
            for method in ABLATION_METHODS:
                m_mean = mean([series[method][k][metric] for k in keys])
                val = None if fr_mean in (None, 0) or m_mean is None else 100.0 * m_mean / fr_mean
                idx.append(100.0 if val is None else val)
                data_rows.append({
                    "tw": tw, "method": method, "metric": metric,
                    "mean_value": m_mean, "alns_fr_mean": fr_mean,
                    "index_vs_alns_fr": val, "n": len(keys),
                })
            ax.plot(xs, idx, color=color, marker=marker, ms=6.5, lw=1.35, label=lab, zorder=3)
        ax.axhline(100.0, color=ZERO, lw=0.85, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(["ALNS-FR", "ALNS-PR", "ALNS-PR+HCO", "ALNS-HCO"], rotation=18, ha="right")
        ax.set_title(_tw_label(tw))
        ax.set_ylabel("Index  (ALNS-FR = 100)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(68, 106)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.text(
        0.5, -0.02,
        "Cross-run means only. ALNS-PR+HCO is a separately run 30 s search plus HCO, "
        "not a successive state of the stored ALNS-PR rows. Prefer Table 4 Panel B for the controlled HCO test.",
        ha="center", va="top", fontsize=7.2, color=C_GRAY,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    _save_fig(fig, dirs["paper_supplement"] / "figS01_charging_ablation_cross_run")
    _write_csv(dirs["paper_data"] / "figS01_charging_ablation_cross_run.csv", data_rows)


def fig04_forest(dirs, rows: list[dict]) -> None:
    from src.alns_hco.paper import _paired, _save_fig, _write_csv

    records = []
    for tw in TW_VARIANTS:
        pr, hco, keys = _paired(rows, tw)
        for metric, lab in FOREST_METRICS:
            pcts = [pct_change(hco[k][metric], pr[k][metric]) for k in keys]
            pcts = [p for p in pcts if p is not None]
            m = mean(pcts)
            lo, hi = bootstrap_mean_ci(pcts, seed=0)
            records.append({
                "tw": tw, "metric": metric, "label": lab, "n": len(pcts),
                "mean_pct": m, "ci95_lo": lo, "ci95_hi": hi,
            })

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.55), sharex=True, sharey=True)
    n = len(FOREST_METRICS)
    ys = list(range(n - 1, -1, -1))
    for ax, tw in zip(axes, TW_VARIANTS):
        sub = [r for r in records if r["tw"] == tw]
        for y, rec in zip(ys, sub):
            m, lo, hi = rec["mean_pct"], rec["ci95_lo"], rec["ci95_hi"]
            if m is None:
                continue
            if hi is not None and hi < 0:
                color = C_BLUE
            elif lo is not None and lo > 0:
                color = C_ORANGE
            else:
                color = C_GRAY
            xerr = None
            if lo is not None and hi is not None:
                xerr = [[m - lo], [hi - m]]
            ax.errorbar(m, y, xerr=xerr, fmt="o", color=color, ecolor=color,
                        elinewidth=1.15, capsize=3.2, ms=5.5, zorder=3)
        ax.axvline(0.0, color=ZERO, lw=1.05, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _m, lab in FOREST_METRICS])
        ax.set_title(_tw_label(tw))
        ax.set_xlabel(r"$100(M_{\mathrm{HCO}}-M_{\mathrm{PR}})/M_{\mathrm{PR}}$ (%)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(-0.6, n - 0.4)
    axes[0].text(0.02, 1.02, "negative = reduction", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=8, color=C_GRAY)
    legend = [
        Line2D([0], [0], marker="o", color=C_BLUE, lw=0, label="CI entirely < 0"),
        Line2D([0], [0], marker="o", color=C_ORANGE, lw=0, label="CI entirely > 0"),
        Line2D([0], [0], marker="o", color=C_GRAY, lw=0, label="CI crosses 0"),
    ]
    fig.legend(handles=legend, loc="upper right", frameon=False, bbox_to_anchor=(0.99, 1.02), ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_fig(fig, dirs["paper_figures"] / "fig04_paired_effects")
    _write_csv(dirs["paper_data"] / "fig04_paired_effects.csv", records)


def fig05_family(dirs, rows: list[dict]) -> None:
    from src.alns_hco.paper import _paired, _save_fig, _write_csv

    mats, data_rows = [], []
    for tw in TW_VARIANTS:
        pr, hco, keys = _paired(rows, tw)
        mat = []
        for fam in FAMILIES:
            row = []
            fk = [k for k in keys if k[1] == fam]
            for metric, _lab in FAMILY_METRICS:
                pcts = [pct_change(hco[k][metric], pr[k][metric]) for k in fk]
                val = mean([p for p in pcts if p is not None])
                row.append(0.0 if val is None else val)
                data_rows.append({"tw": tw, "family": fam, "metric": metric, "mean_pct": val, "n": len(fk)})
            mat.append(row)
        mats.append(mat)
    vmax = max(abs(v) for mat in mats for row in mat for v in row)
    vmax = max(8.0, min(22.0, 1.05 * vmax))

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.55), constrained_layout=True)
    im = None
    for ax, tw, mat in zip(axes, TW_VARIANTS, mats):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(FAMILY_METRICS)))
        ax.set_xticklabels([lab for _m, lab in FAMILY_METRICS])
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_title(_tw_label(tw))
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.55 * vmax else "black")
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("mean paired % vs ALNS-PR\nblue = reduction")
    _save_fig(fig, dirs["paper_figures"] / "fig05_family_charging")
    _write_csv(dirs["paper_data"] / "fig05_family_charging.csv", data_rows)


def write_manifest(dirs) -> None:
    text = """# Figure manifest (Hanif manuscript)

Source of truth: `outputs/raw/master_results.csv`
Generator: `scripts/generate_hanif_figures.py`
Drawing code: `src/alns-hco/figures.py`

Each figure answers one question. Detailed numbers remain in Tables 2–5.
All manuscript figures are **PNG only** (no PDF/SVG sidecars).
Schneider, CARE-FI, runtime, and dataset-description plots are not produced.
The Hanif dataset paper already describes station-price/waiting distributions
and the Original/Relaxed TW structure; those plots are not repeated here.

## Main manuscript

| Fig | File | Question | Source |
|---|---|---|---|
| 1 | author file `figures/fig01_alns_hco_framework.png` | How is ALNS-HCO organised? | **Author drawing. Not generated. Do not use the candidate framework PNG.** |
| 2 | `figures/fig02_hco_decision.png` | How does HCO choose a station and a partial charge? | Conceptual schematic (not experimental) |
| 3 | `figures/fig03_partial_recharging_effect.png` | How much of the charging improvement comes from partial recharging alone? | Paired ALNS-FR → ALNS-PR forest + 95% CI |
| 4 | `figures/fig04_paired_effects.png` | How does full ALNS-HCO compare with a 30 s ALNS-PR? | Paired instance means + bootstrap 95% CI. The $K$ gap is the 15 s vs 30 s Phase-I budget, not HCO changing fleet size |
| 5 | `figures/fig05_family_charging.png` | Which families gain or lose on charging metrics? | Same pairing, families C1–RC2; columns $N_{ch}$, $T_{wait}$, $C_{ch}$ only |

Figures 2–5 are PNG only. Numeric extracts: `processed/<stem>.csv`.

## Supplement

| Fig | File | Question | Note |
|---|---|---|---|
| S1 | `figures/supplement/figS01_charging_ablation_cross_run.png` | Cross-run FR / PR / PR+HCO / HCO index profile | **Not** a controlled sequential ablation; prefer Table 4 Panel B for within-run HCO |

Candidate mechanism / WTL / stability plots remain in `figures_candidates_png/` for optional supplement selection.

## Not used in the manuscript

Dashboard figures and the superseded main four-stage ablation live in `figures/archive/`.
Runtime is uninformative at a matched 30 s budget.

No convergence plots (no iteration traces). No experimental route/SoC figure
(no saved route dumps). Figure 2 is labelled as an illustration for that reason.

## Regeneration

```
python scripts/generate_hanif_figures.py
```

This command never creates or overwrites Figure 1.
"""
    (dirs["root"] / "figure_manifest.md").write_text(text, encoding="utf-8")
    (dirs["paper"] / "figure_manifest.md").write_text(text, encoding="utf-8")


def generate_all_figures(dirs=None, rows=None, master_path: Path | None = None) -> None:
    from src.alns_hco.paper import _captions, _load

    dirs = dirs or hanif_dirs()
    master_path = Path(master_path) if master_path else dirs["raw"] / "master_results.csv"
    if rows is None:
        rows = _load(master_path)
    _archive_old(dirs)
    _purge_vector_sidecars(dirs)
    _write_fig01_note(dirs)
    fig02_hco_decision(dirs)
    if rows:
        fig03_partial_recharging(dirs, rows)
        fig04_forest(dirs, rows)
        fig05_family(dirs, rows)
        figS_charging_ablation(dirs, rows)
    _purge_vector_sidecars(dirs)
    write_manifest(dirs)
    n_inst = len({r["instance"] for r in (rows or []) if r.get("tw") == "original"})
    _captions(dirs, n_inst)
