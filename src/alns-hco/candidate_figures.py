"""Fifteen PNG-only candidate figures for the Hanif ALNS-HCO paper.

Reads outputs/raw/master_results.csv only. Does not run the solver.
PNG files go to outputs/figures_candidates_png/ and nowhere else.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, RegularPolygon

from src.alns_hco.paper import _instance_means, _load, _write_csv
from src.common.config import (
    FAMILIES, METHOD_ALNS_FR, METHOD_ALNS_HCO, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO,
    METHOD_GI, METHOD_SA, METHOD_VNS, TIE_TOL, TW_VARIANTS,
)
from src.common.paths import HANIF_OUT, hanif_dirs
from src.common.stats import bootstrap_mean_ci, mean, pct_change, stdev, wins_ties_losses

DPI = 350
PNG_DIR = HANIF_OUT / "figures_candidates_png"
CSV_DIR = HANIF_OUT / "processed" / "candidates"
MANIFEST = HANIF_OUT / "figure_candidates_manifest.md"

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 120,
    "savefig.dpi": DPI,
    "axes.grid": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.linewidth": 0.7,
    "mathtext.fontset": "dejavusans",
})

C_NAVY = "#1f4e79"
C_ORANGE = "#c45c26"
C_BLUE = "#4c78a8"
C_GREEN = "#3d7a5a"
C_RED = "#b22222"
C_GRAY = "#6b7280"
ZERO = "#111111"

FAM_MARKERS = {"C1": "o", "C2": "s", "R1": "^", "R2": "v", "RC1": "D", "RC2": "P"}

FR_PR_METRICS = (
    ("distance", r"$D$"),
    ("duration", r"$T$"),
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("charged_energy", r"$E_{\mathrm{ch}}$"),
    ("station_waiting", r"$T_{\mathrm{wait}}$"),
    ("charging_time", r"$T_{\mathrm{ch}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)
HCO_PR_METRICS = (
    ("k", r"$K$"),
    ("distance", r"$D$"),
    ("duration", r"$T$"),
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("charged_energy", r"$E_{\mathrm{ch}}$"),
    ("station_waiting", r"$T_{\mathrm{wait}}$"),
    ("charging_time", r"$T_{\mathrm{ch}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)
HEAT_HCO = (
    ("k", r"$K$"),
    ("distance", r"$D$"),
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("charged_energy", r"$E_{\mathrm{ch}}$"),
    ("station_waiting", r"$T_{\mathrm{wait}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)
HEAT_PR = (
    ("distance", r"$D$"),
    ("duration", r"$T$"),
    ("charging_visits", r"$N_{\mathrm{ch}}$"),
    ("charged_energy", r"$E_{\mathrm{ch}}$"),
    ("charging_time", r"$T_{\mathrm{ch}}$"),
    ("charging_expense", r"$C_{\mathrm{ch}}$"),
)
ABLATION_METHODS = (METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_PR_HCO, METHOD_ALNS_HCO)
ABLATION_METRICS = (
    ("charged_energy", r"$E_{\mathrm{ch}}$", C_NAVY, "o"),
    ("charging_time", r"$T_{\mathrm{ch}}$", C_BLUE, "s"),
    ("charging_expense", r"$C_{\mathrm{ch}}$", C_ORANGE, "D"),
    ("station_waiting", r"$T_{\mathrm{wait}}$", C_GREEN, "^"),
)


def _tw_label(tw: str) -> str:
    return "Original TW" if tw == "original" else "Relaxed TW"


def _paired(rows, method_new: str, method_old: str, tw: str):
    new = _instance_means(rows, method_new, tw)
    old = _instance_means(rows, method_old, tw)
    keys = sorted(set(new) & set(old))
    return old, new, keys


def _save_png(fig, stem: str) -> Path:
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    path = PNG_DIR / f"{stem}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    return path


def _style(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ci_color(lo, hi) -> str:
    if lo is not None and hi is not None and hi < 0:
        return C_BLUE
    if lo is not None and hi is not None and lo > 0:
        return C_ORANGE
    return C_GRAY


def _forest_records(rows, method_new, method_old, metrics):
    out = []
    for tw in TW_VARIANTS:
        old, new, keys = _paired(rows, method_new, method_old, tw)
        for metric, lab in metrics:
            pcts = [pct_change(new[k][metric], old[k][metric]) for k in keys]
            pcts = [p for p in pcts if p is not None]
            m = mean(pcts)
            lo, hi = bootstrap_mean_ci(pcts, seed=0)
            out.append({
                "tw": tw, "metric": metric, "label": lab, "n": len(pcts),
                "mean_pct": m, "ci95_lo": lo, "ci95_hi": hi,
                "new": method_new, "baseline": method_old,
            })
    return out


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
                        elinewidth=1.15, capsize=3.0, ms=5.4, zorder=3)
        ax.axvline(0.0, color=ZERO, lw=1.05, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels([lab for _m, lab in metrics])
        ax.set_title(_tw_label(tw))
        ax.set_xlabel(xlabel)
        _style(ax)
        ax.set_ylim(-0.55, n - 0.45)


def _heatmap(axes, mats, metrics, vmax: float):
    im = None
    for ax, tw, mat in zip(axes, TW_VARIANTS, mats):
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(metrics)))
        ax.set_xticklabels([lab for _m, lab in metrics])
        ax.set_yticks(range(len(FAMILIES)))
        ax.set_yticklabels(FAMILIES)
        ax.set_title(_tw_label(tw))
        for i, row in enumerate(mat):
            for j, v in enumerate(row):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(v) > 0.55 * vmax else "black")
    return im


def fig01_framework() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.55))
    ax.set_xlim(0, 16.2)
    ax.set_ylim(0, 7.6)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    def box(x, y, w, h, title, body="", fc="white", lw=1.05):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="square,pad=0",
                                    facecolor=fc, edgecolor="black", linewidth=lw))
        if body:
            ax.text(x + w / 2, y + h - 0.18, title, ha="center", va="top",
                    fontsize=9, fontweight="bold")
            ax.text(x + w / 2, y + 0.16, body, ha="center", va="bottom",
                    fontsize=8, linespacing=1.3)
        else:
            ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                    fontsize=9, fontweight="bold")

    def arrow(x0, y0, x1, y1):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=11, lw=1.05, color="black",
                                     shrinkA=0, shrinkB=0))

    box(0.25, 4.85, 2.55, 1.95, "EVRPTW instance", "Hanif, 1000 customers")
    box(3.45, 4.85, 5.15, 1.95, "Phase I: ALNS-PR",
        "partial recharging\nlexicographic (unserved, $K$, $D$)", fc="#f7f7f7")
    box(9.25, 4.85, 3.55, 1.95, r"Incumbent $X^{\mathrm{PR}}$", r"$K^*=K(X^{\mathrm{PR}})$")
    arrow(2.80, 5.82, 3.45, 5.82)
    arrow(8.60, 5.82, 9.25, 5.82)

    ax.add_patch(FancyBboxPatch((0.25, 0.35), 12.15, 3.95, boxstyle="square,pad=0",
                                facecolor="#f4f4f4", edgecolor="black", linewidth=1.15))
    ax.text(6.32, 4.08, r"Phase II: fixed-fleet ALNS-HCO, $K=K^*$ throughout",
            ha="center", va="top", fontsize=9, fontweight="bold")
    box(0.50, 0.60, 6.55, 3.05, "HCO (inside Phase II)",
        r"candidates scored by $\bar P_s$, $\bar W_s$, $r_s$, detour"
        "\npartial charge $e_s$; insert / replace / remove"
        "\nroute feasibility replay", fc="white")
    box(7.30, 1.70, 4.75, 1.75, "Accept / reject",
        r"$K=K^*$ required" + "\n" + r"SA score $(D,\,C_{\mathrm{ch}})$", fc="white")
    box(13.00, 1.70, 2.90, 1.75, r"Output $X^*$", "ALNS-HCO solution", lw=1.25)
    arrow(11.02, 4.85, 11.02, 4.30)
    ax.plot([6.32, 11.02], [4.30, 4.30], color="black", lw=1.05)
    arrow(6.32, 4.30, 6.32, 3.65)
    arrow(12.05, 2.58, 13.00, 2.58)
    ax.text(11.20, 4.52, r"$K=K^*$", ha="left", va="center", fontsize=8)
    _save_png(fig, "fig01_two_phase_framework")
    _write_csv(CSV_DIR / "fig01_two_phase_framework.csv", [
        {"step": 1, "block": "EVRPTW instance"},
        {"step": 2, "block": "Phase I ALNS-PR", "output": "X_PR, K*"},
        {"step": 3, "block": "Phase II fixed-fleet ALNS-HCO", "constraint": "K=K*"},
        {"step": 4, "block": "X*"},
    ])


def fig02_hco_decision() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.50))
    fig.suptitle("Illustrative HCO decision  (not an experimental route)",
                 fontsize=9, color=C_GRAY, y=0.98)

    ax = axes[0]
    ax.set_xlim(-0.35, 4.15)
    ax.set_ylim(-1.25, 1.45)
    ax.axis("off")
    ax.set_title("(a) Battery deficit", fontsize=10, pad=6)
    ax.add_patch(Circle((0.45, 0.30), 0.17, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.add_patch(Circle((3.55, 0.30), 0.17, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.text(0.45, -0.15, r"$c_i$", ha="center", fontsize=9)
    ax.text(3.55, -0.15, r"$c_j$", ha="center", fontsize=9)
    ax.annotate("", xy=(3.32, 0.30), xytext=(0.68, 0.30),
                arrowprops=dict(arrowstyle="-|>", linestyle=(0, (5, 3)),
                                color=C_RED, lw=1.4, mutation_scale=11))
    ax.text(2.0, 0.72, r"$c_i \rightarrow ? \rightarrow c_j$", ha="center", fontsize=9)
    ax.text(2.0, -0.62, "direct arc infeasible", ha="center", fontsize=8, color=C_RED)
    ax.text(2.0, -0.95, r"remaining SoC $<$ energy $c_i\to c_j$", ha="center", fontsize=8)

    ax = axes[1]
    ax.set_xlim(-0.4, 4.4)
    ax.set_ylim(-1.55, 1.75)
    ax.axis("off")
    ax.set_title("(b) Restricted candidates", fontsize=10, pad=6)
    ax.add_patch(Circle((0.45, 0.90), 0.15, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.add_patch(Circle((3.55, 0.90), 0.15, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.text(0.45, 1.20, r"$c_i$", ha="center", fontsize=8)
    ax.text(3.55, 1.20, r"$c_j$", ha="center", fontsize=8)
    ax.plot([0.60, 3.40], [0.90, 0.90], color="0.75", lw=0.8, ls="--")
    stations = [
        (1.15, -0.05, r"$s_1$", "long detour\n" + r"high $\bar P_s$, high $r_s$"),
        (2.00, -0.48, r"$s_2$", "short detour\n" + r"low $\bar W_s$, moderate $r_s$"),
        (2.85, -0.05, r"$s_3$", "short detour\n" + r"low $\bar P_s$, high $\bar W_s$"),
    ]
    for x, y, name, lab in stations:
        ax.add_patch(RegularPolygon((x, y), 4, radius=0.21, orientation=0.785,
                                    facecolor="white", edgecolor="black", lw=0.9, zorder=3))
        ax.text(x, y, name, ha="center", va="center", fontsize=8, zorder=4)
        ax.plot([0.45, x], [0.90, y + 0.16], color="0.55", lw=0.7)
        ax.plot([x, 3.55], [y + 0.16, 0.90], color="0.55", lw=0.7)
        ax.text(x, y - 0.38, lab, ha="center", va="top", fontsize=6.6, color=C_GRAY, linespacing=1.2)
    ax.text(2.0, 1.52, r"$\bar P_s$, $\bar W_s$, $r_s$, detour", ha="center", fontsize=8)

    ax = axes[2]
    ax.set_xlim(-0.35, 4.35)
    ax.set_ylim(-1.40, 1.55)
    ax.axis("off")
    ax.set_title("(c) Partial charge and replay", fontsize=10, pad=6)
    ax.add_patch(Circle((0.45, 0.55), 0.15, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.add_patch(RegularPolygon((2.0, 0.55), 4, radius=0.23, orientation=0.785,
                                facecolor="#f4e6d8", edgecolor=C_ORANGE, lw=1.25, zorder=3))
    ax.add_patch(Circle((3.55, 0.55), 0.15, facecolor=C_NAVY, edgecolor="black", lw=0.8, zorder=3))
    ax.text(0.45, 0.90, r"$c_i$", ha="center", fontsize=8)
    ax.text(2.0, 0.55, r"$s^*$", ha="center", va="center", fontsize=8, color=C_ORANGE, fontweight="bold")
    ax.text(3.55, 0.90, r"$c_j$", ha="center", fontsize=8)
    ax.annotate("", xy=(1.78, 0.55), xytext=(0.60, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10))
    ax.annotate("", xy=(3.40, 0.55), xytext=(2.22, 0.55),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=10))
    ax.text(2.0, 1.22, r"HCO selects $s_2\equiv s^*$", ha="center", fontsize=8)
    ax.text(2.0, 0.05, r"just-enough partial charge $e_s$", ha="center", fontsize=8)
    ax.add_patch(FancyBboxPatch((0.35, -1.15), 3.30, 0.70, boxstyle="square,pad=0",
                                facecolor="#f7f7f7", edgecolor="black", lw=0.8))
    ax.text(2.0, -0.60, "feasibility replay", ha="center", fontsize=8, fontweight="bold")
    ax.text(2.0, -0.90, "TW, battery, depot return", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_png(fig, "fig02_hco_decision")
    _write_csv(CSV_DIR / "fig02_hco_decision.csv", [
        {"panel": "a", "content": "infeasible customer arc c_i to c_j"},
        {"panel": "b", "content": "candidates s1,s2,s3 with Pbar, Wbar, r, detour"},
        {"panel": "c", "content": "select s*, partial charge e_s, feasibility replay"},
        {"note": "illustrative schematic; no saved Hanif routes"},
    ])


def fig03_partial_recharge(rows) -> None:
    recs = _forest_records(rows, METHOD_ALNS_PR, METHOD_ALNS_FR, FR_PR_METRICS)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.35), sharex=True, sharey=True)
    _draw_forest(axes, recs, FR_PR_METRICS,
                 r"$100(M_{\mathrm{PR}}-M_{\mathrm{FR}})/M_{\mathrm{FR}}$ (%)")
    axes[0].text(0.02, 1.02, "negative = reduction", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=8, color=C_GRAY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_png(fig, "fig03_partial_recharging_effect")
    _write_csv(CSV_DIR / "fig03_partial_recharging_effect.csv", recs)


def fig04_ablation(rows) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.80), sharey=True)
    data_rows = []
    xs = list(range(len(ABLATION_METHODS)))
    for ax, tw in zip(axes, TW_VARIANTS):
        series = {m: _instance_means(rows, m, tw) for m in ABLATION_METHODS}
        keys = sorted(set.intersection(*(set(series[m]) for m in ABLATION_METHODS)))
        for metric, lab, color, marker in ABLATION_METRICS:
            fr_mean = mean([series[METHOD_ALNS_FR][k][metric] for k in keys])
            idx = []
            for method in ABLATION_METHODS:
                m_mean = mean([series[method][k][metric] for k in keys])
                val = None if not fr_mean or m_mean is None else 100.0 * m_mean / fr_mean
                idx.append(100.0 if val is None else val)
                data_rows.append({
                    "tw": tw, "method": method, "metric": metric,
                    "mean_value": m_mean, "index_vs_alns_fr": val, "n": len(keys),
                })
            ax.plot(xs, idx, color=color, marker=marker, ms=6.4, lw=1.35, label=lab, zorder=3)
        ax.axhline(100.0, color=ZERO, lw=0.85, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(["ALNS-FR", "ALNS-PR", "ALNS-PR+HCO", "ALNS-HCO"],
                           rotation=18, ha="right")
        ax.set_title(_tw_label(tw))
        ax.set_ylabel("Index  (ALNS-FR = 100)")
        _style(ax)
        ax.set_ylim(68, 106)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    _save_png(fig, "fig04_charging_ablation")
    _write_csv(CSV_DIR / "fig04_charging_ablation.csv", data_rows)


def fig05_hco_vs_pr(rows) -> None:
    recs = _forest_records(rows, METHOD_ALNS_HCO, METHOD_ALNS_PR, HCO_PR_METRICS)
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.55), sharex=True, sharey=True)
    _draw_forest(axes, recs, HCO_PR_METRICS,
                 r"$100(M_{\mathrm{HCO}}-M_{\mathrm{PR}})/M_{\mathrm{PR}}$ (%)")
    axes[0].text(0.02, 1.02, "negative = reduction", transform=axes[0].transAxes,
                 ha="left", va="bottom", fontsize=8, color=C_GRAY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _save_png(fig, "fig05_alns_hco_paired_effect")
    _write_csv(CSV_DIR / "fig05_alns_hco_paired_effect.csv", recs)


def fig06_win_tie_loss(rows) -> None:
    data_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.55), sharex=True)
    for ax, tw in zip(axes, TW_VARIANTS):
        old, new, keys = _paired(rows, METHOD_ALNS_HCO, METHOD_ALNS_PR, tw)
        wins, ties, losses, labels = [], [], [], []
        for metric, lab in reversed(HCO_PR_METRICS):
            a = [new[k][metric] for k in keys]
            b = [old[k][metric] for k in keys]
            w, t, l = wins_ties_losses(a, b, tol=TIE_TOL)
            wins.append(w)
            ties.append(t)
            losses.append(l)
            labels.append(lab)
            data_rows.append({"tw": tw, "metric": metric, "wins": w, "ties": t,
                              "losses": l, "n": len(keys)})
        y = range(len(labels))
        ax.barh(y, wins, color=C_BLUE, height=0.72, label="HCO better")
        ax.barh(y, ties, left=wins, color="#d1d5db", height=0.72, label="tie")
        left2 = [w + t for w, t in zip(wins, ties)]
        ax.barh(y, losses, left=left2, color=C_ORANGE, height=0.72, label="PR better")
        ax.set_yticks(list(y), labels)
        ax.set_title(_tw_label(tw))
        ax.set_xlabel("Paired instances (n = 60)")
        ax.set_xlim(0, 60)
        _style(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save_png(fig, "fig06_win_tie_loss")
    _write_csv(CSV_DIR / "fig06_win_tie_loss.csv", data_rows)


def _family_matrix(rows, method_new, method_old, metrics):
    mats, recs = [], []
    for tw in TW_VARIANTS:
        old, new, keys = _paired(rows, method_new, method_old, tw)
        mat = []
        for fam in FAMILIES:
            row = []
            fk = [k for k in keys if k[1] == fam]
            for metric, _lab in metrics:
                pcts = [pct_change(new[k][metric], old[k][metric]) for k in fk]
                val = mean([p for p in pcts if p is not None])
                row.append(0.0 if val is None else val)
                recs.append({"tw": tw, "family": fam, "metric": metric,
                             "mean_pct": val, "n": len(fk)})
            mat.append(row)
        mats.append(mat)
    vmax = max(abs(v) for mat in mats for row in mat for v in row)
    vmax = max(8.0, 1.05 * vmax)
    return mats, recs, vmax


def fig07_family_hco(rows) -> None:
    mats, recs, vmax = _family_matrix(rows, METHOD_ALNS_HCO, METHOD_ALNS_PR, HEAT_HCO)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), constrained_layout=True)
    im = _heatmap(axes, mats, HEAT_HCO, vmax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label(r"mean paired %  $100(M_{\mathrm{HCO}}-M_{\mathrm{PR}})/M_{\mathrm{PR}}$"
                   + "\nblue = reduction")
    _save_png(fig, "fig07_family_hco_effect")
    _write_csv(CSV_DIR / "fig07_family_hco_effect.csv", recs)


def fig08_family_partial(rows) -> None:
    mats, recs, vmax = _family_matrix(rows, METHOD_ALNS_PR, METHOD_ALNS_FR, HEAT_PR)
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.55), constrained_layout=True)
    im = _heatmap(axes, mats, HEAT_PR, vmax)
    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cbar.set_label(r"mean paired %  $100(M_{\mathrm{PR}}-M_{\mathrm{FR}})/M_{\mathrm{FR}}$"
                   + "\nblue = reduction")
    _save_png(fig, "fig08_family_partial_charging")
    _write_csv(CSV_DIR / "fig08_family_partial_charging.csv", recs)


def fig09_fleet_distance(rows) -> None:
    methods = (METHOD_GI, METHOD_VNS, METHOD_SA, METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_HCO)
    offsets = {
        METHOD_GI: (6, 8), METHOD_VNS: (6, -10), METHOD_SA: (6, 8),
        METHOD_ALNS_FR: (6, -12), METHOD_ALNS_PR: (6, 10), METHOD_ALNS_HCO: (6, 8),
    }
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85))
    recs = []
    for ax, tw in zip(axes, TW_VARIANTS):
        for method in methods:
            inst = _instance_means(rows, method, tw)
            k = [v["k"] for v in inst.values()]
            d = [v["distance"] for v in inst.values()]
            mk, md = mean(k), mean(d)
            ek, ed = stdev(k), stdev(d)
            recs.append({"tw": tw, "method": method, "mean_k": mk, "mean_d": md,
                         "sd_k": ek, "sd_d": ed, "n": len(inst)})
            if mk is None or md is None:
                continue
            ax.scatter([mk], [md], s=42, color=C_NAVY, zorder=3)
            dx, dy = offsets.get(method, (6, 6))
            ax.annotate(method, (mk, md), textcoords="offset points", xytext=(dx, dy),
                        fontsize=7.5, color=C_NAVY)
        ax.set_title(_tw_label(tw))
        ax.set_xlabel(r"Mean fleet size $K$")
        ax.set_ylabel(r"Mean distance $D$")
        _style(ax)
    fig.tight_layout()
    _save_png(fig, "fig09_fleet_distance_tradeoff")
    _write_csv(CSV_DIR / "fig09_fleet_distance_tradeoff.csv", recs)


def fig10_expense_waiting(rows) -> None:
    methods = ABLATION_METHODS
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.75))
    recs = []
    for ax, tw in zip(axes, TW_VARIANTS):
        xs, ys = [], []
        for method in methods:
            inst = _instance_means(rows, method, tw)
            c = mean([v["charging_expense"] for v in inst.values()])
            w = mean([v["station_waiting"] for v in inst.values()])
            recs.append({"tw": tw, "method": method, "mean_c_ch": c, "mean_t_wait": w, "n": len(inst)})
            xs.append(c)
            ys.append(w)
        label_off = {
            METHOD_ALNS_FR: (6, 6),
            METHOD_ALNS_PR: (6, 8),
            METHOD_ALNS_PR_HCO: (6, -12),
            METHOD_ALNS_HCO: (6, 6),
        }
        ax.plot(xs, ys, color=C_GRAY, lw=1.0, zorder=1)
        for method, x, y in zip(methods, xs, ys):
            ax.scatter([x], [y], s=42, color=C_NAVY, zorder=3)
            dx, dy = label_off.get(method, (6, 5))
            ax.annotate(method, (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=7.5)
        ax.set_title(_tw_label(tw))
        ax.set_xlabel(r"Mean charging expense $C_{\mathrm{ch}}$")
        ax.set_ylabel(r"Mean station waiting $T_{\mathrm{wait}}$")
        _style(ax)
    fig.tight_layout()
    _save_png(fig, "fig10_expense_waiting_tradeoff")
    _write_csv(CSV_DIR / "fig10_expense_waiting_tradeoff.csv", recs)


def _instance_deltas(rows, x_metric, y_metric):
    recs = []
    for tw in TW_VARIANTS:
        old, new, keys = _paired(rows, METHOD_ALNS_HCO, METHOD_ALNS_PR, tw)
        for k in keys:
            dx = pct_change(new[k][x_metric], old[k][x_metric])
            dy = pct_change(new[k][y_metric], old[k][y_metric])
            if dx is None or dy is None:
                continue
            recs.append({
                "tw": tw, "instance": k[0], "family": k[1],
                "dx": dx, "dy": dy, "x_metric": x_metric, "y_metric": y_metric,
            })
    return recs


def _delta_scatter(recs, xlabel, ylabel, stem, family_markers=False) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.85))
    for ax, tw in zip(axes, TW_VARIANTS):
        sub = [r for r in recs if r["tw"] == tw]
        if family_markers:
            for fam, mk in FAM_MARKERS.items():
                pts = [r for r in sub if r["family"] == fam]
                ax.scatter([r["dx"] for r in pts], [r["dy"] for r in pts],
                           marker=mk, s=22, facecolors="none", edgecolors=C_NAVY,
                           linewidths=0.8, alpha=0.85, label=fam)
        else:
            ax.scatter([r["dx"] for r in sub], [r["dy"] for r in sub],
                       s=18, color=C_NAVY, alpha=0.7, edgecolors="none")
        ax.axvline(0.0, color=ZERO, lw=0.95, zorder=1)
        ax.axhline(0.0, color=ZERO, lw=0.95, zorder=1)
        ax.set_title(_tw_label(tw))
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        _style(ax)
    if family_markers:
        fig.legend(loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.02), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    else:
        fig.tight_layout()
    _save_png(fig, stem)


def fig11_expense_waiting_instance(rows) -> None:
    recs = _instance_deltas(rows, "station_waiting", "charging_expense")
    _delta_scatter(
        recs,
        r"$\Delta T_{\mathrm{wait}}$  (% vs ALNS-PR)",
        r"$\Delta C_{\mathrm{ch}}$  (% vs ALNS-PR)",
        "fig11_instance_expense_waiting",
    )
    _write_csv(CSV_DIR / "fig11_instance_expense_waiting.csv", recs)


def fig12_energy_expense(rows) -> None:
    recs = _instance_deltas(rows, "charged_energy", "charging_expense")
    _delta_scatter(
        recs,
        r"$\Delta E_{\mathrm{ch}}$  (% vs ALNS-PR)",
        r"$\Delta C_{\mathrm{ch}}$  (% vs ALNS-PR)",
        "fig12_energy_expense_mechanism",
        family_markers=True,
    )
    _write_csv(CSV_DIR / "fig12_energy_expense_mechanism.csv", recs)


def fig13_visits_waiting(rows) -> None:
    recs = _instance_deltas(rows, "charging_visits", "station_waiting")
    _delta_scatter(
        recs,
        r"$\Delta N_{\mathrm{ch}}$  (% vs ALNS-PR)",
        r"$\Delta T_{\mathrm{wait}}$  (% vs ALNS-PR)",
        "fig13_visits_waiting_mechanism",
    )
    _write_csv(CSV_DIR / "fig13_visits_waiting_mechanism.csv", recs)


def fig14_time_per_energy(rows) -> None:
    recs = []
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.70), sharey=True)
    xs = [0, 1]
    for ax, tw in zip(axes, TW_VARIANTS):
        old, new, keys = _paired(rows, METHOD_ALNS_HCO, METHOD_ALNS_PR, tw)
        for fam in FAMILIES:
            fk = [k for k in keys if k[1] == fam]

            def ratio(src, fk=fk):
                vals = []
                for k in fk:
                    e, t = src[k]["charged_energy"], src[k]["charging_time"]
                    if e is None or t is None or abs(e) < 1e-12:
                        continue
                    vals.append(t / e)
                return mean(vals)

            a, b = ratio(old), ratio(new)
            recs.append({"tw": tw, "family": fam, "alns_pr": a, "alns_hco": b, "n": len(fk)})
            if a is None or b is None:
                continue
            ax.plot(xs, [a, b], color=C_GRAY, lw=0.9, zorder=1)
            ax.scatter([0], [a], s=28, color=C_NAVY, zorder=3)
            ax.scatter([1], [b], s=28, color=C_ORANGE, zorder=3)
            ax.annotate(fam, (1.04, b), fontsize=7.5, va="center", color=C_GRAY)
        ax.set_xticks(xs, ["ALNS-PR", "ALNS-HCO"])
        ax.set_title(_tw_label(tw))
        ax.set_ylabel(r"$T_{\mathrm{ch}}/E_{\mathrm{ch}}$")
        _style(ax)
    fig.tight_layout()
    _save_png(fig, "fig14_charging_time_per_energy")
    _write_csv(CSV_DIR / "fig14_charging_time_per_energy.csv", recs)


def _cv(vals):
    m = mean(vals)
    s = stdev(vals)
    if m is None or s is None or abs(m) < 1e-12 or len(vals) < 2:
        return None
    return 100.0 * s / abs(m)


def fig15_seed_stability(rows) -> None:
    methods = (METHOD_VNS, METHOD_SA, METHOD_ALNS_FR, METHOD_ALNS_PR, METHOD_ALNS_HCO)
    recs = []
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.85), sharey=True)
    for ax, metric, ylab in zip(
        axes, ("distance", "charging_expense"),
        (r"CV of $D$ (%)", r"CV of $C_{\mathrm{ch}}$ (%)"),
    ):
        series, labels, positions = [], [], []
        pos = 1.0
        colors = []
        for method in methods:
            for tw, color in (("original", C_NAVY), ("relaxed", C_ORANGE)):
                groups = {}
                for r in rows:
                    if r.get("method") != method or r.get("tw") != tw or not r.get("feasible"):
                        continue
                    if r.get(metric) is None:
                        continue
                    groups.setdefault(r["instance"], []).append(r[metric])
                cvs = [c for c in (_cv(vs) for vs in groups.values()) if c is not None]
                for inst, vs in groups.items():
                    recs.append({
                        "method": method, "tw": tw, "metric": metric,
                        "instance": inst, "cv_pct": _cv(vs), "n_seeds": len(vs),
                    })
                series.append(cvs)
                labels.append(f"{method}\n{_tw_label(tw).split()[0]}")
                positions.append(pos)
                colors.append(color)
                pos += 1.0
            pos += 0.35
        bp = ax.boxplot(
            series, positions=positions, widths=0.62, showfliers=False,
            medianprops=dict(color="black", linewidth=1.05),
            boxprops=dict(linewidth=0.8), whiskerprops=dict(linewidth=0.8),
            capprops=dict(linewidth=0.8), patch_artist=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor("white")
            patch.set_edgecolor(color)
        ax.set_xticks(positions, labels, fontsize=6.6)
        ax.set_ylabel(ylab)
        _style(ax)
    fig.tight_layout()
    _save_png(fig, "fig15_seed_stability")
    _write_csv(CSV_DIR / "fig15_seed_stability.csv", recs)


MANIFEST_ROWS = [
    ("fig01_two_phase_framework.png",
     "How is ALNS-HCO organised as a two-phase algorithm?",
     "schematic", "architecture", "none", "Main"),
    ("fig02_hco_decision.png",
     "What is distinctive about an HCO charging decision?",
     "schematic", "Pbar_s, Wbar_s, r_s, detour, e_s", "illustrative", "Main"),
    ("fig03_partial_recharging_effect.png",
     "How much of the charging improvement comes from partial recharging alone?",
     "master_results.csv", "D, T, N_ch, E_ch, T_wait, T_ch, C_ch",
     "ALNS-PR vs ALNS-FR", "Main"),
    ("fig04_charging_ablation.png",
     "At which algorithmic stage do charging gains appear?",
     "master_results.csv", "E_ch, T_ch, C_ch, T_wait",
     "ALNS-FR = 100; FR to PR to PR+HCO to HCO", "Main"),
    ("fig05_alns_hco_paired_effect.png",
     "What does Phase-II ALNS-HCO change relative to ALNS-PR?",
     "master_results.csv", "K, D, T, N_ch, E_ch, T_wait, T_ch, C_ch",
     "ALNS-HCO vs ALNS-PR", "Main"),
    ("fig06_win_tie_loss.png",
     "Are the Phase-II effects consistent across the 60 instances?",
     "master_results.csv", "K, D, T, N_ch, E_ch, T_wait, T_ch, C_ch",
     "HCO better / tie / PR better", "Main"),
    ("fig07_family_hco_effect.png",
     "Which Hanif families benefit from charging-aware search?",
     "master_results.csv", "K, D, N_ch, E_ch, T_wait, C_ch",
     "ALNS-HCO vs ALNS-PR by family", "Main or Supplement"),
    ("fig08_family_partial_charging.png",
     "Is the partial-recharging benefit uniform across families?",
     "master_results.csv", "D, T, N_ch, E_ch, T_ch, C_ch",
     "ALNS-PR vs ALNS-FR by family", "Supplement"),
    ("fig09_fleet_distance_tradeoff.png",
     "How do the methods sit on the fleet-distance frontier?",
     "master_results.csv", "K, D",
     "GI, VNS, SA, ALNS-FR, ALNS-PR, ALNS-HCO", "Supplement"),
    ("fig10_expense_waiting_tradeoff.png",
     "Does lower station waiting imply lower charging expense?",
     "master_results.csv", "C_ch, T_wait",
     "ALNS-FR to PR to PR+HCO to HCO", "Supplement"),
    ("fig11_instance_expense_waiting.png",
     "On which instances does HCO reduce both waiting and expense?",
     "master_results.csv", "dC_ch, dT_wait",
     "ALNS-HCO vs ALNS-PR, instance level", "Supplement"),
    ("fig12_energy_expense_mechanism.png",
     "Do expense changes track charged energy, or station selection?",
     "master_results.csv", "dE_ch, dC_ch",
     "ALNS-HCO vs ALNS-PR, instance level", "Supplement"),
    ("fig13_visits_waiting_mechanism.png",
     "Does waiting reduction come mainly from fewer charging visits?",
     "master_results.csv", "dN_ch, dT_wait",
     "ALNS-HCO vs ALNS-PR, instance level", "Supplement"),
    ("fig14_charging_time_per_energy.png",
     "Does HCO change the effective charging time per unit energy?",
     "master_results.csv", "T_ch / E_ch",
     "family means, ALNS-PR vs ALNS-HCO", "Supplement"),
    ("fig15_seed_stability.png",
     "How repeatable are the stochastic methods across the five seeds?",
     "master_results.csv", "CV of D and C_ch",
     "VNS, SA, ALNS-FR, ALNS-PR, ALNS-HCO", "Supplement"),
]


def write_manifest() -> None:
    lines = [
        "# Candidate figure manifest (Hanif ALNS-HCO)",
        "",
        "PNG directory: `outputs/figures_candidates_png/` (PNG only).",
        "Numerical extracts: `outputs/processed/candidates/`.",
        "Source of truth: `outputs/raw/master_results.csv`.",
        "Generator: `scripts/generate_hanif_candidate_figures.py`.",
        "Drawing code: `src/alns-hco/candidate_figures.py`.",
        "",
        "No solver rerun. No Schneider/CARE-FI figures. No dataset-description plots.",
        "Figure 2 is a methodological illustration; no saved Hanif routes exist.",
        "Percentage changes are always $100(\\mathrm{new}-\\mathrm{baseline})/\\mathrm{baseline}$; negative = reduction.",
        "",
        "| Fig | File | Research question | Data | Metrics | Comparison | Recommendation |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (fn, q, data, metrics, cmp, rec) in enumerate(MANIFEST_ROWS, start=1):
        lines.append(f"| {i} | `{fn}` | {q} | {data} | {metrics} | {cmp} | {rec} |")
    lines += [
        "",
        "## Regeneration",
        "",
        "```",
        "python scripts/generate_hanif_candidate_figures.py",
        "```",
        "",
    ]
    MANIFEST.write_text("\n".join(lines), encoding="utf-8")


def generate() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    dirs = hanif_dirs()
    rows = _load(dirs["raw"] / "master_results.csv")
    fig01_framework()
    fig02_hco_decision()
    fig03_partial_recharge(rows)
    fig04_ablation(rows)
    fig05_hco_vs_pr(rows)
    fig06_win_tie_loss(rows)
    fig07_family_hco(rows)
    fig08_family_partial(rows)
    fig09_fleet_distance(rows)
    fig10_expense_waiting(rows)
    fig11_expense_waiting_instance(rows)
    fig12_energy_expense(rows)
    fig13_visits_waiting(rows)
    fig14_time_per_energy(rows)
    fig15_seed_stability(rows)
    write_manifest()
    allowed = {row[0] for row in MANIFEST_ROWS}
    for path in PNG_DIR.iterdir():
        if path.is_file() and path.name not in allowed:
            path.unlink()

