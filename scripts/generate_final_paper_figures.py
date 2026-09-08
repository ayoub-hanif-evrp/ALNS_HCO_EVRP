"""Regenerate publication-ready paper figures from frozen CSV tables only.

Does not run any solver. Reads:
  outputs/tables/table_partial_recharging.csv
  outputs/tables/table_fixed_route_heterogeneity.csv
  outputs/tables/table_phase2_nohco_vs_hco.csv
  outputs/tables/table_family_results.csv
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = OUT / "figures"
TABLES = OUT / "tables"

# Paper math labels (matplotlib mathtext)
LABEL = {
    "distance": r"$D$",
    "charged_energy": r"$E_{\mathrm{ch}}$",
    "charging_expense": r"$C_{\mathrm{ch}}$",
    "station_waiting": r"$T_{\mathrm{wait}}$",
    "charging_time": r"$T_{\mathrm{ch}}$",
    "charging_visits": r"$N_{\mathrm{ch}}$",
}

TW_TITLE = {"original": "Original TW", "relaxed": "Relaxed TW"}
XLABEL = "Mean paired change (%)"

STYLE = {
    "dpi": 300,
    "figsize_forest": (8.6, 3.6),
    "figsize_family": (8.6, 3.8),
    "marker": "o",
    "markersize": 6.5,
    "elinewidth": 1.35,
    "capsize": 0.0,
    "color": "#1f4e79",
    "zero_color": "#666666",
    "grid_alpha": 0.25,
    "font_size": 10,
    "title_size": 11,
    "margin": 0.08,  # fraction of span beyond CI extrema
}


def _load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("mean_paired_pct", "ci95_lo", "ci95_hi", "charging_expense", "n"):
            if k in r and r[k] not in (None, ""):
                try:
                    r[k] = float(r[k])
                except ValueError:
                    pass
    return rows


def _pick(rows: list[dict], tw: str, metrics: list[str]) -> list[dict]:
    out = []
    for m in metrics:
        rec = next((r for r in rows if r.get("tw") == tw and r.get("metric") == m), None)
        if rec is None:
            raise SystemExit(f"missing metric={m} tw={tw} in source table")
        out.append(rec)
    return out


def _common_xlim(panel_records: list[list[dict]], suggested: tuple[float, float] | None = None) -> tuple[float, float]:
    vals = []
    for recs in panel_records:
        for r in recs:
            for k in ("ci95_lo", "mean_paired_pct", "ci95_hi"):
                v = r.get(k)
                if v is not None and v == v:
                    vals.append(float(v))
    if not vals:
        return suggested or (-5.0, 5.0)
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, 1.0)
    margin = STYLE["margin"] * span
    x0, x1 = lo - margin, hi + margin
    # Always include zero
    x0 = min(x0, -0.5)
    x1 = max(x1, 0.5)
    if suggested is not None:
        # Expand to at least the suggested window if data fit near it, else keep data-driven
        s0, s1 = suggested
        # Prefer suggested if it covers data (+tiny slack); else use data-driven
        if s0 <= lo and s1 >= hi:
            return s0, s1
        return x0, x1
    return x0, x1


def _forest_panel(ax, records: list[dict], xlim: tuple[float, float], title: str) -> None:
    n = len(records)
    ys = list(range(n - 1, -1, -1))
    means, los, his, labels = [], [], [], []
    for r in records:
        m = float(r["mean_paired_pct"])
        lo = float(r["ci95_lo"]) if r.get("ci95_lo") is not None else m
        hi = float(r["ci95_hi"]) if r.get("ci95_hi") is not None else m
        means.append(m)
        los.append(m - lo)
        his.append(hi - m)
        labels.append(LABEL[r["metric"]])

    ax.errorbar(
        means, ys,
        xerr=[los, his],
        fmt=STYLE["marker"],
        color=STYLE["color"],
        markersize=STYLE["markersize"],
        elinewidth=STYLE["elinewidth"],
        capsize=STYLE["capsize"],
        zorder=3,
    )
    ax.axvline(0.0, color=STYLE["zero_color"], linewidth=1.0, linestyle="-", zorder=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlim(*xlim)
    ax.set_xlabel(XLABEL)
    ax.set_title(title, fontsize=STYLE["title_size"])
    ax.grid(True, axis="x", alpha=STYLE["grid_alpha"], linestyle=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _verify_against_source(records: list[dict], source_rows: list[dict], tw: str) -> None:
    for r in records:
        src = next(x for x in source_rows if x["tw"] == tw and x["metric"] == r["metric"])
        for k in ("mean_paired_pct", "ci95_lo", "ci95_hi"):
            a, b = float(r[k]), float(src[k])
            if abs(a - b) > 1e-9:
                raise SystemExit(f"CI/mean mismatch {tw} {r['metric']} {k}: plot={a} csv={b}")


def fig_partial_recharging() -> Path:
    src = _load(TABLES / "table_partial_recharging.csv")
    metrics = ["charged_energy", "charging_time", "charging_expense", "charging_visits", "station_waiting"]
    orig = _pick(src, "original", metrics)
    rel = _pick(src, "relaxed", metrics)
    _verify_against_source(orig, src, "original")
    _verify_against_source(rel, src, "relaxed")
    xlim = _common_xlim([orig, rel], suggested=(-40.0, 5.0))

    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_forest"], sharey=True)
    _forest_panel(axes[0], orig, xlim, TW_TITLE["original"])
    _forest_panel(axes[1], rel, xlim, TW_TITLE["relaxed"])
    fig.tight_layout()
    path = FIG / "figA_partial_recharging_effect.png"
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path} xlim={xlim}")
    return path


def fig_fixed_route() -> Path:
    src = _load(TABLES / "table_fixed_route_heterogeneity.csv")
    metrics = ["charging_expense", "station_waiting", "charging_time", "charging_visits"]
    orig = _pick(src, "original", metrics)
    rel = _pick(src, "relaxed", metrics)
    _verify_against_source(orig, src, "original")
    _verify_against_source(rel, src, "relaxed")
    xlim = _common_xlim([orig, rel], suggested=(-3.5, 2.0))

    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_forest"], sharey=True)
    _forest_panel(axes[0], orig, xlim, TW_TITLE["original"])
    _forest_panel(axes[1], rel, xlim, TW_TITLE["relaxed"])
    fig.tight_layout()
    path = FIG / "figB_fixed_route_heterogeneity.png"
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path} xlim={xlim}")
    return path


def fig_nohco_vs_hco() -> Path:
    src = _load(TABLES / "table_phase2_nohco_vs_hco.csv")
    # Core four (omit N_ch: near-zero / low signal)
    metrics = ["distance", "charging_expense", "station_waiting", "charging_time"]
    orig = _pick(src, "original", metrics)
    rel = _pick(src, "relaxed", metrics)
    _verify_against_source(orig, src, "original")
    _verify_against_source(rel, src, "relaxed")
    xlim = _common_xlim([orig, rel], suggested=(-8.0, 2.0))

    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_forest"], sharey=True)
    _forest_panel(axes[0], orig, xlim, TW_TITLE["original"])
    _forest_panel(axes[1], rel, xlim, TW_TITLE["relaxed"])
    fig.tight_layout()
    path = FIG / "figE_nohco_vs_hco.png"
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path} xlim={xlim}")
    return path


def fig_family_phase2_hco() -> Path:
    src = _load(TABLES / "table_family_results.csv")
    families = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
    panels = {}
    for tw in ("original", "relaxed"):
        vals = []
        for fam in families:
            rec = next(
                (
                    r for r in src
                    if r.get("tw") == tw and r.get("family") == fam and r.get("comparison") == "phase2_nohco_hco"
                ),
                None,
            )
            if rec is None:
                raise SystemExit(f"missing family row {tw} {fam} phase2_nohco_hco")
            vals.append(float(rec["charging_expense"]))
        panels[tw] = vals

    # No family-level CIs in the frozen table — means only (do not invent CIs).
    all_vals = panels["original"] + panels["relaxed"]
    lo, hi = min(all_vals), max(all_vals)
    span = max(hi - lo, 1.0)
    margin = STYLE["margin"] * span
    xlim = (min(lo - margin, -25.0), max(hi + margin, 5.0))
    # Prefer suggested window if it covers data
    if -25.0 <= lo and 5.0 >= hi:
        xlim = (-25.0, 5.0)

    fig, axes = plt.subplots(1, 2, figsize=STYLE["figsize_family"], sharey=True)
    ys = list(range(len(families) - 1, -1, -1))
    for ax, tw in zip(axes, ("original", "relaxed")):
        xs = panels[tw]
        ax.scatter(xs, ys, s=45, color=STYLE["color"], zorder=3)
        for x, y in zip(xs, ys):
            ax.hlines(y, 0.0, x, color=STYLE["color"], linewidth=1.2, alpha=0.55, zorder=2)
        ax.axvline(0.0, color=STYLE["zero_color"], linewidth=1.0, zorder=1)
        ax.set_yticks(ys)
        ax.set_yticklabels(families)
        ax.set_xlim(*xlim)
        ax.set_xlabel(XLABEL)
        ax.set_title(TW_TITLE[tw], fontsize=STYLE["title_size"])
        ax.grid(True, axis="x", alpha=STYLE["grid_alpha"], linestyle=":")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    # Verify printed means against CSV
    for tw in ("original", "relaxed"):
        for fam, x in zip(families, panels[tw]):
            rec = next(
                r for r in src
                if r["tw"] == tw and r["family"] == fam and r["comparison"] == "phase2_nohco_hco"
            )
            if abs(float(rec["charging_expense"]) - x) > 1e-9:
                raise SystemExit("family mean mismatch")
    fig.tight_layout()
    path = FIG / "figF_family_phase2_hco_effect.png"
    fig.savefig(path, dpi=STYLE["dpi"], bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path} xlim={xlim}")
    print("  Original C_ch by family:", dict(zip(families, panels["original"])))
    print("  Relaxed  C_ch by family:", dict(zip(families, panels["relaxed"])))
    return path


def relocate_old_family_heatmap() -> None:
    old = FIG / "figF_family_heterogeneous_effect.png"
    if not old.is_file():
        print("no old figF heatmap to move")
        return
    supp = FIG / "supplement"
    supp.mkdir(parents=True, exist_ok=True)
    dest = supp / "figS2_fixed_route_family_heterogeneity.png"
    shutil.move(str(old), str(dest))
    print(f"moved {old.name} -> {dest}")


def write_manifest() -> None:
    text = r"""# Final paper figure manifesto

Generated from frozen CSV tables only (no solver rerun).
Sign convention: mean paired percentage \(100(B-A)/A\); negative means \(B\) is lower than \(A\).

Figure 1 — methodology figure — manual, not generated by code
Figure 2 — figA_partial_recharging_effect.png
Figure 3 — figB_fixed_route_heterogeneity.png
Figure 4 — figE_nohco_vs_hco.png
Figure 5 — figF_family_phase2_hco_effect.png

## Source CSV / table for each generated figure

| Paper figure | File | Comparison | Metrics | Source CSV |
|---|---|---|---|---|
| Figure 2 | `figA_partial_recharging_effect.png` | ALNS-FR → ALNS-PR | \(E_{ch}, T_{ch}, C_{ch}, N_{ch}, T_{wait}\) | `outputs/tables/table_partial_recharging.csv` |
| Figure 3 | `figB_fixed_route_heterogeneity.png` | FixedRoute-HCO-Detour → FixedRoute-HCO-Heterogeneous | \(C_{ch}, T_{wait}, T_{ch}, N_{ch}\) | `outputs/tables/table_fixed_route_heterogeneity.csv` |
| Figure 4 | `figE_nohco_vs_hco.png` | PhaseII-NoHCO → PhaseII-Heterogeneous | \(D, C_{ch}, T_{wait}, T_{ch}\) | `outputs/tables/table_phase2_nohco_vs_hco.csv` |
| Figure 5 | `figF_family_phase2_hco_effect.png` | PhaseII-NoHCO → PhaseII-Heterogeneous by family | \(C_{ch}\) | `outputs/tables/table_family_results.csv` (`comparison=phase2_nohco_hco`) |

## Captions (paper)

- **Figure 2.** Effect of Partial Recharging on Routing and Charging Performance
- **Figure 3.** Fixed-Route Effect of Heterogeneous Charging Information
- **Figure 4.** Effect of HCO Relative to the Matched Phase-II NoHCO Search
- **Figure 5.** Family-Level Effect of HCO in the Matched Phase-II Search

## Supplementary / diagnostic (not main paper)

- `figC_station_characteristics.png`
- `figD_phase1_to_phase2.png`
- `figS1_phase2_detour_vs_heterogeneous.png`
- `supplement/figS2_fixed_route_family_heterogeneity.png` (former main figF heatmap)

## Style notes

- Identical x-axis limits across Original / Relaxed panels within each figure
- Vertical zero reference line
- X-axis label: Mean paired change (%)
- Paper math notation for metrics; no code-style column names
- Family Figure 5: means only (no family-level CIs in the frozen table)
- Shared xlims used: Fig2 [-40, 5], Fig3 [-3.5, 2.0], Fig4 [-8, 2], Fig5 [-25, 5]
"""
    path = FIG / "final_figure_manifest.md"
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path}")


def main() -> int:
    plt.rcParams.update({
        "font.size": STYLE["font_size"],
        "axes.labelsize": STYLE["font_size"],
        "axes.titlesize": STYLE["title_size"],
        "figure.dpi": 120,
        "savefig.dpi": STYLE["dpi"],
    })
    FIG.mkdir(parents=True, exist_ok=True)
    relocate_old_family_heatmap()
    fig_partial_recharging()
    fig_fixed_route()
    fig_nohco_vs_hco()
    fig_family_phase2_hco()
    write_manifest()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
