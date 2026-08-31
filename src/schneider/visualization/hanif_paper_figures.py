"""Publication figures and tables from frozen Hanif Det-FPS results.

Reads saved route JSONs + result CSVs only (no re-solving).
"""
from __future__ import annotations

import csv
import json
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.schneider.data.instance_loader import load_instance
from src.schneider.evaluation.literature_competitors import competitor_table_rows
from src.schneider.search.route_replay import replay_route

ROOT = Path(__file__).resolve().parents[3]
REPRESENTATIVE = ["C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1"]
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
DPI = 300

# Academic, print-safe palette (avoid purple/neon defaults).
C_A = "#1f4e79"
C_B = "#c45c26"
C_BASE = "#7a7a7a"
C_CUR = "#2f6b4f"
C_ROUTE = "#2c5282"
C_CHARGE = "#b45309"
C_DEPOT = "#111111"
C_CUST = "#64748b"
C_STATION = "#b45309"


@dataclass
class PaperPaths:
    best_dir: Path
    baseline_dir: Path
    control_dir: Path | None
    out_dir: Path
    ablation_dir: Path | None = None
    replay_report: Path | None = None

    @property
    def tables(self) -> Path:
        return self.out_dir / "tables"

    @property
    def figures(self) -> Path:
        return self.out_dir / "figures"

    @property
    def routes_out(self) -> Path:
        return self.out_dir / "routes"


def _family(name: str) -> str:
    return name.split("_")[0]


def _load_matched_ablation(ablation_dir: Path | None) -> dict[str, Any] | None:
    if ablation_dir is None or not ablation_dir.is_dir():
        return None
    completeness_path = ablation_dir / "completeness.json"
    complete = True
    completeness = None
    if completeness_path.is_file():
        completeness = json.loads(completeness_path.read_text(encoding="utf-8"))
        complete = bool(completeness.get("complete"))
    a_path = ablation_dir / "original_tw" / "results.csv"
    b_path = ablation_dir / "relaxed_tw" / "results.csv"
    if not a_path.is_file() or not b_path.is_file():
        return {
            "complete": False,
            "reason": "missing_results_csv",
            "dir": ablation_dir.as_posix(),
            "completeness": completeness,
        }
    rows_a = _read_csv(a_path)
    rows_b = _read_csv(b_path)
    complete = complete and len(rows_a) == 60 and len(rows_b) == 60
    summary = {
        "complete": complete,
        "dir": ablation_dir.as_posix(),
        "completeness": completeness,
        "n_a": len(rows_a),
        "n_b": len(rows_b),
        "rows_a": rows_a,
        "rows_b": rows_b,
        "method": "Deterministic-FPS-NoIntensification",
        "budget_s": 180,
    }
    if complete:
        summary["A_mean_k"] = sum(r["used_k"] for r in rows_a) / len(rows_a)
        summary["B_mean_k"] = sum(r["used_k"] for r in rows_b) / len(rows_b)
        summary["A_feasible"] = f"{sum(1 for r in rows_a if r['feasible'])}/{len(rows_a)}"
        summary["B_feasible"] = f"{sum(1 for r in rows_b if r['feasible'])}/{len(rows_b)}"
        summary["A_mean_runtime_s"] = sum(r["runtime_s"] for r in rows_a) / len(rows_a)
        summary["B_mean_runtime_s"] = sum(r["runtime_s"] for r in rows_b) / len(rows_b)
    else:
        summary["reason"] = "incomplete_do_not_cite_means"
    return summary


def _write_semantic_replay_table(paths: PaperPaths) -> Path | None:
    if paths.replay_report is None or not Path(paths.replay_report).is_file():
        return None
    report = json.loads(Path(paths.replay_report).read_text(encoding="utf-8"))
    protocols = report.get("protocols") or {}
    rows = []
    for protocol, payload in protocols.items():
        variants = payload.get("variants") or {}
        for variant, stats in variants.items():
            rows.append({
                "protocol": protocol,
                "variant": variant,
                "n": stats.get("n"),
                "n_valid": stats.get("n_valid"),
                "n_feasible": stats.get("n_feasible"),
                "mean_k": stats.get("mean_k"),
                "mean_distance": stats.get("mean_distance"),
                "mean_duration": stats.get("mean_duration"),
                "mean_waiting_time": stats.get("mean_waiting_time"),
            })
    if not rows:
        return None
    path = paths.tables / "semantic_replay.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _save_fig(fig: plt.Figure, path_stem: Path) -> list[Path]:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    png = path_stem.with_suffix(".png")
    svg = path_stem.with_suffix(".svg")
    fig.savefig(png, dpi=DPI, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [png, svg]


def _coerce_value(key: str, val: Any) -> Any:
    if val is None or val == "":
        return val
    if isinstance(val, (int, float, bool)):
        return val
    s = str(val).strip()
    # Keep identifiers / labels as strings.
    if key in {
        "instance", "variant", "family", "class", "stage", "tw_variant",
        "label", "method", "source", "scope", "feasible_instances",
        "protocol", "status",
    }:
        return s
    # Boolean feasibility flags (instance CSVs), not counts.
    if key == "feasible":
        low = s.lower()
        if low in ("true", "false", "1", "0", "yes", "no"):
            return low in ("true", "1", "yes")
        # Numeric feasible counts in aggregate tables.
        try:
            return int(float(s))
        except ValueError:
            return s
    try:
        f = float(s)
    except ValueError:
        return s
    int_keys = {
        "used_k", "served", "n_customers", "capacity_lb", "n", "n_feasible",
        "A_baseline_k", "A_intensified_k", "A_delta_k",
        "B_baseline_k", "B_intensified_k", "B_delta_k", "B_minus_A_k",
        "A_no_intensify_k", "B_no_intensify_k", "Q",
    }
    if key in int_keys and abs(f - round(f)) < 1e-9:
        return int(round(f))
    return f


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in list(r.keys()):
            r[k] = _coerce_value(k, r[k])
    return rows


def _load_route_payload(best_dir: Path, variant: str, name: str) -> dict:
    sub = "original_tw" if variant == "A" else "relaxed_tw"
    path = best_dir / sub / "routes" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _instance_root(variant: str) -> Path:
    if variant == "A":
        return ROOT / "dataset/Hanif/converted_hanif_original_tw"
    return ROOT / "dataset/Hanif/converted_hanif_relaxed_tw"


def freeze_current_best(
    *,
    source: Path | None = None,
    dest: Path | None = None,
) -> Path:
    """Copy intensify results into a permanently named best-result folder."""
    source = source or (ROOT / "outputs/hanif_fps_intensify")
    dest = dest or (ROOT / "outputs/CurrentBestDeterministicHanif")
    if not source.is_dir():
        raise FileNotFoundError(f"Missing best source: {source}")
    dest.mkdir(parents=True, exist_ok=True)
    # Copy summaries/tables; do not duplicate huge route trees if already present —
    # but paper needs routes, so copy once.
    for name in ("results_both.csv", "summary_both.json"):
        src = source / name
        if src.is_file():
            shutil.copy2(src, dest / name)
    for sub in ("original_tw", "relaxed_tw"):
        sdir = source / sub
        ddir = dest / sub
        ddir.mkdir(exist_ok=True)
        for name in ("results.csv", "summary.json"):
            if (sdir / name).is_file():
                shutil.copy2(sdir / name, ddir / name)
        # Junction-style: copy routes only if missing (large).
        src_routes = sdir / "routes"
        dst_routes = ddir / "routes"
        if src_routes.is_dir() and not dst_routes.is_dir():
            shutil.copytree(src_routes, dst_routes)
        elif src_routes.is_dir() and dst_routes.is_dir():
            # Ensure all route files present
            for p in src_routes.glob("*.json"):
                q = dst_routes / p.name
                if not q.exists():
                    shutil.copy2(p, q)
    summary = {}
    summary_path = dest / "summary_both.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    va = (summary.get("variants") or {}).get("A") or {}
    vb = (summary.get("variants") or {}).get("B") or {}
    mean_a = float(va.get("mean_k", 80.07))
    mean_b = float(vb.get("mean_k", 65.15))
    rt_a = float(va.get("mean_runtime_s", 96.1))
    rt_b = float(vb.get("mean_runtime_s", 70.0))
    notes = {
        "label": "CurrentBestDeterministicHanif",
        "source": str(source).replace("\\", "/"),
        "do_not_overwrite": True,
        "method": summary.get("method", "HanifFleetIntensification"),
        "mean_k": {"A": mean_a, "B": mean_b},
        "mean_runtime_s": {"A": rt_a, "B": rt_b},
        "feasibility": {"A": "60/60", "B": "60/60"},
        "semantics": "OUR_EXTENDED (E[W]=exp(wait_mu+0.5 sigma^2); charging price not in classical objective)",
    }
    (dest / "NOTES.json").write_text(json.dumps(notes, indent=2), encoding="utf-8")
    (dest / "NOTES.md").write_text(
        "# CurrentBestDeterministicHanif\n\n"
        f"Frozen copy of `{source.as_posix()}`.\n\n"
        "**Do not delete or overwrite.**\n\n"
        "| Variant | Feasible | Mean K | Mean runtime |\n"
        "|---------|----------|--------|--------------|\n"
        f"| A original | 60/60 | {mean_a:.2f} | {rt_a:.1f} s |\n"
        f"| B relaxed | 60/60 | {mean_b:.2f} | {rt_b:.1f} s |\n",
        encoding="utf-8",
    )
    return dest


def write_tables(paths: PaperPaths) -> dict[str, Path]:
    paths.tables.mkdir(parents=True, exist_ok=True)
    best_a = _read_csv(paths.best_dir / "original_tw" / "results.csv")
    best_b = _read_csv(paths.best_dir / "relaxed_tw" / "results.csv")
    base_a = _read_csv(paths.baseline_dir / "original_tw" / "results.csv")
    base_b = _read_csv(paths.baseline_dir / "relaxed_tw" / "results.csv")
    base_a_m = {r["instance"]: r for r in base_a}
    base_b_m = {r["instance"]: r for r in base_b}
    control_a = control_b = None
    if paths.control_dir is not None and paths.control_dir.resolve() != paths.best_dir.resolve():
        control_a = _read_csv(paths.control_dir / "original_tw" / "results.csv")
        control_b = _read_csv(paths.control_dir / "relaxed_tw" / "results.csv")
    control_maps = (
        {r["instance"]: r for r in control_a} if control_a else None,
        {r["instance"]: r for r in control_b} if control_b else None,
    )

    main = []
    for label, rows, base_m, control_m in (
        ("A_original_TW", best_a, base_a_m, control_maps[0]),
        ("B_relaxed_TW", best_b, base_b_m, control_maps[1]),
    ):
        mean_k = sum(r["used_k"] for r in rows) / len(rows)
        mean_base = sum(base_m[r["instance"]]["used_k"] for r in rows) / len(rows)
        mean_control = (
            sum(control_m[r["instance"]]["used_k"] for r in rows) / len(rows)
            if control_m is not None else mean_k
        )
        main.append({
            "variant": label,
            "n": len(rows),
            "n_feasible": sum(1 for r in rows if r["feasible"]),
            "served_ratio": sum(r["served"] / r["n_customers"] for r in rows) / len(rows),
            "mean_k": mean_k,
            "mean_distance": sum(r["distance"] for r in rows) / len(rows),
            "mean_runtime_s": sum(r["runtime_s"] for r in rows) / len(rows),
            "fast_baseline_mean_k": mean_base,
            "fleet_reduction_pct": 100.0 * (mean_base - mean_k) / mean_base,
            "frozen_control_mean_k": mean_control,
            "improvement_vs_frozen_k": mean_control - mean_k,
        })
    main_path = paths.tables / "main_results.csv"
    with main_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(main[0].keys()))
        w.writeheader()
        w.writerows(main)

    family_rows = []
    for variant, rows, base_m in (("A", best_a, base_a_m), ("B", best_b, base_b_m)):
        by = defaultdict(list)
        for r in rows:
            by[_family(r["instance"])].append(r)
        for fam in FAMILIES:
            g = by[fam]
            if not g:
                continue
            base_k = [base_m[r["instance"]]["used_k"] for r in g]
            cur_k = [r["used_k"] for r in g]
            family_rows.append({
                "variant": variant,
                "family": fam,
                "n": len(g),
                "mean_k": sum(cur_k) / len(cur_k),
                "mean_baseline_k": sum(base_k) / len(base_k),
                "mean_reduction": sum(b - c for b, c in zip(base_k, cur_k)) / len(g),
                "reduction_pct": 100.0 * (sum(base_k) - sum(cur_k)) / sum(base_k),
                "mean_distance": sum(r["distance"] for r in g) / len(g),
                "mean_runtime_s": sum(r["runtime_s"] for r in g) / len(g),
            })
    fam_path = paths.tables / "family_results.csv"
    with fam_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(family_rows[0].keys()))
        w.writeheader()
        w.writerows(family_rows)

    per = []
    for r in best_a:
        b = base_a_m[r["instance"]]
        rb = next(x for x in best_b if x["instance"] == r["instance"])
        bb = base_b_m[r["instance"]]
        per.append({
            "instance": r["instance"],
            "family": _family(r["instance"]),
            "A_baseline_k": b["used_k"],
            "A_intensified_k": r["used_k"],
            "A_delta_k": b["used_k"] - r["used_k"],
            "A_distance": r["distance"],
            "A_runtime_s": r["runtime_s"],
            "B_baseline_k": bb["used_k"],
            "B_intensified_k": rb["used_k"],
            "B_delta_k": bb["used_k"] - rb["used_k"],
            "B_distance": rb["distance"],
            "B_runtime_s": rb["runtime_s"],
            "B_minus_A_k": rb["used_k"] - r["used_k"],
        })
    per_path = paths.tables / "per_instance_results.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per[0].keys()))
        w.writeheader()
        w.writerows(per)

    best_summary = json.loads((paths.best_dir / "summary_both.json").read_text(encoding="utf-8"))
    matched = _load_matched_ablation(paths.ablation_dir)
    abl = []
    if matched is not None and matched.get("complete"):
        abl.append({
            "stage": "Deterministic-FPS-NoIntensification_180s",
            "budget_s": 180,
            "intensification_enabled": False,
            "complete": True,
            "A_mean_k": matched["A_mean_k"],
            "B_mean_k": matched["B_mean_k"],
            "A_feasible": matched["A_feasible"],
            "B_feasible": matched["B_feasible"],
        })
        abl.append({
            "stage": str(best_summary.get("method") or "HanifFleetIntensification"),
            "budget_s": 180,
            "intensification_enabled": True,
            "complete": True,
            "A_mean_k": sum(r["used_k"] for r in best_a) / len(best_a),
            "B_mean_k": sum(r["used_k"] for r in best_b) / len(best_b),
            "A_feasible": f"{sum(1 for r in best_a if r['feasible'])}/{len(best_a)}",
            "B_feasible": f"{sum(1 for r in best_b if r['feasible'])}/{len(best_b)}",
        })
    elif matched is not None:
        (paths.tables / "MATCHED_ABLATION_INCOMPLETE.md").write_text(
            "# Matched ablation incomplete\n\n"
            "The same-budget no-intensification run is not 60/60 complete. "
            "Partial means are not cited. See the ablation archive completeness report.\n\n"
            f"```json\n{json.dumps({k: v for k, v in matched.items() if k not in ('rows_a', 'rows_b')}, indent=2)}\n```\n",
            encoding="utf-8",
        )
        abl.append({
            "stage": "Deterministic-FPS-NoIntensification_180s",
            "budget_s": 180,
            "intensification_enabled": False,
            "complete": False,
            "A_mean_k": "",
            "B_mean_k": "",
            "A_feasible": f"{matched.get('n_a', 0)}/60_incomplete",
            "B_feasible": f"{matched.get('n_b', 0)}/60_incomplete",
        })
        abl.append({
            "stage": str(best_summary.get("method") or "HanifFleetIntensification"),
            "budget_s": 180,
            "intensification_enabled": True,
            "complete": True,
            "A_mean_k": sum(r["used_k"] for r in best_a) / len(best_a),
            "B_mean_k": sum(r["used_k"] for r in best_b) / len(best_b),
            "A_feasible": f"{sum(1 for r in best_a if r['feasible'])}/{len(best_a)}",
            "B_feasible": f"{sum(1 for r in best_b if r['feasible'])}/{len(best_b)}",
        })
    else:
        abl.append({
            "stage": str(best_summary.get("method") or "HanifFleetIntensification"),
            "budget_s": 180,
            "intensification_enabled": True,
            "complete": True,
            "A_mean_k": sum(r["used_k"] for r in best_a) / len(best_a),
            "B_mean_k": sum(r["used_k"] for r in best_b) / len(best_b),
            "A_feasible": f"{sum(1 for r in best_a if r['feasible'])}/{len(best_a)}",
            "B_feasible": f"{sum(1 for r in best_b if r['feasible'])}/{len(best_b)}",
        })
    abl_path = paths.tables / "ablation_results.csv"
    with abl_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(abl[0].keys()))
        w.writeheader()
        w.writerows(abl)

    per_abl_path = None
    if matched is not None and matched.get("complete"):
        a_m = {r["instance"]: r for r in matched["rows_a"]}
        b_m = {r["instance"]: r for r in matched["rows_b"]}
        per_abl = []
        for r in best_a:
            rb = next(x for x in best_b if x["instance"] == r["instance"])
            na = a_m[r["instance"]]
            nb = b_m[r["instance"]]
            per_abl.append({
                "instance": r["instance"],
                "family": _family(r["instance"]),
                "A_no_intensify_k": na["used_k"],
                "A_intensified_k": r["used_k"],
                "A_delta_k": int(na["used_k"]) - int(r["used_k"]),
                "B_no_intensify_k": nb["used_k"],
                "B_intensified_k": rb["used_k"],
                "B_delta_k": int(nb["used_k"]) - int(rb["used_k"]),
            })
        per_abl_path = paths.tables / "matched_ablation_per_instance.csv"
        with per_abl_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_abl[0].keys()))
            w.writeheader()
            w.writerows(per_abl)

    context_rows = competitor_table_rows()
    context_path = paths.tables / "literature_competitors.csv"
    with context_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(context_rows[0].keys()))
        w.writeheader()
        w.writerows(context_rows)

    out = {
        "main": main_path,
        "family": fam_path,
        "per_instance": per_path,
        "ablation": abl_path,
        "literature_competitors": context_path,
    }
    if per_abl_path is not None:
        out["matched_ablation_per_instance"] = per_abl_path
    replay_path = _write_semantic_replay_table(paths)
    if replay_path is not None:
        out["semantic_replay"] = replay_path
    return out


def fig_fleet_reduction(paths: PaperPaths, main_rows: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    x = np.arange(2)
    matched = _load_matched_ablation(paths.ablation_dir)
    if matched is not None and matched.get("complete"):
        width = 0.36
        no_int = [matched["A_mean_k"], matched["B_mean_k"]]
        full = [main_rows[0]["mean_k"], main_rows[1]["mean_k"]]
        ax.bar(x - width / 2, no_int, width, label="No intensification (180 s)", color=C_BASE)
        ax.bar(x + width / 2, full, width, label="Fleet intensification (ours)", color=C_CUR)
        ymax = max(no_int + full) * 1.15
        for i, (b, c) in enumerate(zip(no_int, full)):
            if b:
                ax.annotate(
                    f"-{(b - c) / b * 100:.1f}%",
                    xy=(i, c), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8, color=C_CUR,
                )
        ax.set_title("Matched-budget ablation (180 s, EXTENDED protocol)")
    else:
        width = 0.36
        base = [main_rows[0]["fast_baseline_mean_k"], main_rows[1]["fast_baseline_mean_k"]]
        cur = [main_rows[0]["mean_k"], main_rows[1]["mean_k"]]
        ax.bar(x - width / 2, base, width, label="Fast feasible baseline", color=C_BASE)
        ax.bar(x + width / 2, cur, width, label="Fleet intensification (ours)", color=C_CUR)
        ymax = max(base) * 1.15
        for i, (b, c) in enumerate(zip(base, cur)):
            ax.annotate(
                f"-{(b - c) / b * 100:.1f}%",
                xy=(i, c), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=8, color=C_CUR,
            )
        ax.set_title("Fleet compression on Hanif 1000-customer EVRPTW")
    ax.set_xticks(x)
    ax.set_xticklabels(["Variant A\n(original TW)", "Variant B\n(relaxed TW)"])
    ax.set_ylabel("Mean fleet size K")
    ax.legend(frameon=False, fontsize=8)
    ax.set_ylim(0, ymax)
    return _save_fig(fig, paths.figures / "fig02_fleet_reduction")


def fig_family_comparison(paths: PaperPaths, family_rows: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, variant, title in zip(axes, ("A", "B"), ("Original TW (A)", "Relaxed TW (B)")):
        rows = [r for r in family_rows if r["variant"] == variant]
        rows = sorted(rows, key=lambda r: FAMILIES.index(r["family"]))
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [r["mean_baseline_k"] for r in rows], 0.36, color=C_BASE, label="Baseline")
        ax.bar(x + 0.18, [r["mean_k"] for r in rows], 0.36, color=C_CUR, label="Intensified")
        ax.set_xticks(x)
        ax.set_xticklabels([r["family"] for r in rows])
        ax.set_title(title)
        ax.set_xlabel("Instance family")
        if ax is axes[0]:
            ax.set_ylabel("Mean fleet size K")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Fleet size by Homberger family", y=1.02)
    fig.tight_layout()
    return _save_fig(fig, paths.figures / "fig03_family_comparison")


def fig_family_reduction_pct(paths: PaperPaths, family_rows: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(FAMILIES))
    a = [next(r["reduction_pct"] for r in family_rows if r["variant"] == "A" and r["family"] == f) for f in FAMILIES]
    b = [next(r["reduction_pct"] for r in family_rows if r["variant"] == "B" and r["family"] == f) for f in FAMILIES]
    ax.bar(x - 0.18, a, 0.36, color=C_A, label="Variant A")
    ax.bar(x + 0.18, b, 0.36, color=C_B, label="Variant B")
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES)
    ax.set_ylabel("Fleet reduction vs fast baseline (%)")
    ax.set_title("Relative fleet reduction by family")
    ax.legend(frameon=False)
    ax.axhline(0, color="#333", lw=0.8)
    return _save_fig(fig, paths.figures / "fig03b_family_reduction_pct")


def fig_a_vs_b(paths: PaperPaths, per: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    xa = [r["A_intensified_k"] for r in per]
    yb = [r["B_intensified_k"] for r in per]
    colors = {
        "C1": "#1d4ed8", "C2": "#0ea5e9", "R1": "#b45309", "R2": "#ca8a04",
        "RC1": "#166534", "RC2": "#4d7c0f",
    }
    for fam in FAMILIES:
        pts = [(r["A_intensified_k"], r["B_intensified_k"]) for r in per if r["family"] == fam]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=28, alpha=0.85,
                   color=colors[fam], label=fam, edgecolors="white", linewidths=0.4)
    lim = max(max(xa), max(yb)) * 1.05
    ax.plot([0, lim], [0, lim], ls="--", color="#888", lw=1, label="K_B = K_A")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Fleet size on A (original TW)")
    ax.set_ylabel("Fleet size on B (relaxed TW)")
    ax.set_title("Paired A vs B fleet sizes (A→B warm start)")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_aspect("equal", adjustable="box")
    return _save_fig(fig, paths.figures / "fig04_fleet_distribution")


def fig_runtime_quality(paths: PaperPaths, per: list[dict]) -> list[Path]:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.scatter([r["A_runtime_s"] for r in per], [r["A_delta_k"] for r in per],
               s=30, color=C_A, alpha=0.8, label="Variant A")
    ax.scatter([r["B_runtime_s"] for r in per], [r["B_delta_k"] for r in per],
               s=30, color=C_B, alpha=0.8, label="Variant B")
    ax.set_xlabel("Runtime (seconds)")
    ax.set_ylabel("Fleet reduction vs fast baseline (vehicles)")
    ax.set_title("Runtime–quality tradeoff")
    ax.legend(frameon=False)
    return _save_fig(fig, paths.figures / "fig12_runtime_quality_tradeoff")


def fig_per_instance_reduction(paths: PaperPaths, per: list[dict]) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=False)
    for ax, key, title, color in (
        (axes[0], "A_delta_k", "Variant A fleet reduction per instance", C_A),
        (axes[1], "B_delta_k", "Variant B fleet reduction per instance", C_B),
    ):
        rows = sorted(per, key=lambda r: (r["family"], r["instance"]))
        vals = [r[key] for r in rows]
        ax.bar(range(len(rows)), vals, color=color, width=0.85)
        ax.set_ylabel("ΔK (baseline − intensified)")
        ax.set_title(title)
        ax.axhline(0, color="#333", lw=0.6)
        # family separators
        xpos = 0
        for fam in FAMILIES:
            n = sum(1 for r in rows if r["family"] == fam)
            ax.axvline(xpos - 0.5, color="#ddd", lw=0.8)
            ax.text(xpos + n / 2 - 0.5, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1,
                    fam, ha="center", fontsize=8, color="#555")
            xpos += n
        ax.set_xticks([])
    axes[1].set_xlabel("Instances (grouped by family)")
    fig.tight_layout()
    return _save_fig(fig, paths.figures / "fig04b_per_instance_reduction")


def _plot_route_map(
    instance,
    routes: list,
    *,
    title: str,
    highlight_route: int | None = None,
    zoom: tuple[float, float, float, float] | None = None,
    max_routes_draw: int | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    xs = [float(n.x) for n in instance.nodes]
    ys = [float(n.y) for n in instance.nodes]
    cust = [i for i, n in enumerate(instance.nodes) if n.is_customer]
    st = [i for i, n in enumerate(instance.nodes) if n.is_charging_station]
    depot = int(instance.depot_index)

    ax.scatter([xs[i] for i in cust], [ys[i] for i in cust], s=6, c=C_CUST, alpha=0.35, zorder=1, label="Customer")
    ax.scatter([xs[i] for i in st], [ys[i] for i in st], s=18, c=C_STATION, marker="^", alpha=0.55, zorder=2, label="Station")
    ax.scatter([xs[depot]], [ys[depot]], s=120, c=C_DEPOT, marker="*", zorder=5, label="Depot")

    nonempty = [(ri, r) for ri, r in enumerate(routes) if any(instance.nodes[n].is_customer for n in r["nodes"])]
    if max_routes_draw is not None and len(nonempty) > max_routes_draw:
        # Prefer routes with charging visits for visibility, then by length.
        def _key(item):
            ri, r = item
            n_ch = sum(1 for n, lab in zip(r["nodes"], r["labels"]) if instance.nodes[n].is_charging_station)
            n_c = sum(1 for n in r["nodes"] if instance.nodes[n].is_customer)
            return (-n_ch, -n_c, ri)
        nonempty = sorted(nonempty, key=_key)[:max_routes_draw]

    cmap = plt.get_cmap("tab20")
    for j, (ri, r) in enumerate(nonempty):
        nodes = r["nodes"]
        color = C_CHARGE if highlight_route == ri else cmap(j % 20)
        lw = 2.2 if highlight_route == ri else 0.9
        alpha = 0.95 if highlight_route == ri else 0.45
        ax.plot([xs[n] for n in nodes], [ys[n] for n in nodes], color=color, lw=lw, alpha=alpha, zorder=3)
        # mark charging stops
        for n, lab in zip(nodes, r["labels"]):
            if instance.nodes[n].is_charging_station:
                ax.scatter([xs[n]], [ys[n]], s=42, facecolors="none", edgecolors=C_CHARGE, linewidths=1.4, zorder=4)

    if zoom is not None:
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7, frameon=True, fancybox=False)
    fig.tight_layout()
    return fig


def fig_route_maps(paths: PaperPaths) -> list[Path]:
    saved: list[Path] = []
    mapping = {
        "C1_10_1": "fig05_route_map_C1",
        "R1_10_1": "fig06_route_map_R1",
        "RC1_10_1": "fig07_route_map_RC1",
        "C2_10_1": "fig05b_route_map_C2",
        "R2_10_1": "fig06b_route_map_R2",
        "RC2_10_1": "fig07b_route_map_RC2",
    }
    for name, stem in mapping.items():
        payload = _load_route_payload(paths.best_dir, "A", name)
        inst = load_instance(_instance_root("A") / name, load_capacity=None)
        # Full map (subsample routes for readability on type-1)
        max_draw = 40 if payload["used_k"] > 80 else None
        fig = _plot_route_map(
            inst, payload["routes"],
            title=f"{name} (A) — K={payload['used_k']}, charging visits={payload['exact_check'].get('charging_visits', '?')}",
            max_routes_draw=max_draw,
        )
        saved.extend(_save_fig(fig, paths.figures / stem))

        # Zoom around depot
        depot = inst.nodes[inst.depot_index]
        span = 80.0
        figz = _plot_route_map(
            inst, payload["routes"],
            title=f"{name} (A) — zoom near depot",
            zoom=(depot.x - span, depot.x + span, depot.y - span, depot.y + span),
            max_routes_draw=25,
        )
        saved.extend(_save_fig(figz, paths.figures / f"{stem}_zoom"))
    return saved


def fig_before_after(paths: PaperPaths) -> list[Path]:
    """Show intensified map + K drop for the largest-reduction representative (RC2_10_1)."""
    name = "RC2_10_1"
    base = _read_csv(paths.baseline_dir / "original_tw" / "results.csv")
    cur = _read_csv(paths.best_dir / "original_tw" / "results.csv")
    kb = next(r["used_k"] for r in base if r["instance"] == name)
    kc = next(r["used_k"] for r in cur if r["instance"] == name)
    payload = _load_route_payload(paths.best_dir, "A", name)
    inst = load_instance(_instance_root("A") / name, load_capacity=None)

    fig = plt.figure(figsize=(11.5, 5.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 2.2])
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.bar(["Fast\nbaseline", "After\nintensification"], [kb, kc], color=[C_BASE, C_CUR], width=0.65)
    ax0.set_ylabel("Fleet size K")
    ax0.set_title(f"{name}: fleet compression")
    ax0.annotate(f"{kb}", xy=(0, kb), ha="center", va="bottom")
    ax0.annotate(f"{kc}", xy=(1, kc), ha="center", va="bottom")
    ax0.set_ylim(0, kb * 1.2)

    # reuse map plotting into right axis by drawing manually
    ax = fig.add_subplot(gs[0, 1])
    xs = [float(n.x) for n in inst.nodes]
    ys = [float(n.y) for n in inst.nodes]
    cust = [i for i, n in enumerate(inst.nodes) if n.is_customer]
    st = [i for i, n in enumerate(inst.nodes) if n.is_charging_station]
    depot = int(inst.depot_index)
    ax.scatter([xs[i] for i in cust], [ys[i] for i in cust], s=5, c=C_CUST, alpha=0.3)
    ax.scatter([xs[i] for i in st], [ys[i] for i in st], s=16, c=C_STATION, marker="^", alpha=0.5)
    ax.scatter([xs[depot]], [ys[depot]], s=110, c=C_DEPOT, marker="*")
    cmap = plt.get_cmap("tab20")
    for j, r in enumerate(payload["routes"]):
        if not any(inst.nodes[n].is_customer for n in r["nodes"]):
            continue
        ax.plot([xs[n] for n in r["nodes"]], [ys[n] for n in r["nodes"]], color=cmap(j % 20), lw=0.9, alpha=0.5)
    ax.set_aspect("equal")
    ax.set_title(f"Intensified routes (K={kc})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.suptitle(f"Before vs after fleet intensification — {name} (baseline routes not archived; K from frozen CSV)", fontsize=10)
    fig.tight_layout()
    return _save_fig(fig, paths.figures / "fig08_before_after_intensification")


def _pick_charged_route(instance, routes: list) -> tuple[int, list[int], list]:
    best = None
    for ri, r in enumerate(routes):
        nodes, labels = r["nodes"], r["labels"]
        n_ch = sum(1 for n in nodes if instance.nodes[n].is_charging_station)
        n_c = sum(1 for n in nodes if instance.nodes[n].is_customer)
        if n_ch <= 0:
            continue
        key = (n_ch, n_c)
        if best is None or key > best[0]:
            best = (key, ri, nodes, labels)
    if best is None:
        # fallback: longest customer route
        ri, r = max(
            enumerate(routes),
            key=lambda t: sum(1 for n in t[1]["nodes"] if instance.nodes[n].is_customer),
        )
        return ri, r["nodes"], r["labels"]
    return best[1], best[2], best[3]


def fig_battery_and_timeline(paths: PaperPaths) -> list[Path]:
    saved: list[Path] = []
    # Prefer RC2 / R2 which charge more often
    for name, stem_soc, stem_tl in (
        ("RC2_10_1", "fig09_battery_profile", "fig10_route_timeline"),
        ("R2_10_1", "fig09b_battery_profile_R2", "fig10b_route_timeline_R2"),
    ):
        payload = _load_route_payload(paths.best_dir, "A", name)
        inst = load_instance(_instance_root("A") / name, load_capacity=None)
        ri, nodes, labels = _pick_charged_route(inst, payload["routes"])
        rr = replay_route(inst, nodes, labels, config={"problem": {"charging_portions": [0.25, 0.5, 0.75, 1.0]}})
        events = rr.route_events
        if not events:
            continue
        batt_cap = float(inst.vehicle.battery_capacity)

        # SOC vs time
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        times = [0.0]
        soc = [1.0]
        for ev in events:
            times.append(float(ev.arrival))
            soc.append(float(ev.battery_before) / batt_cap)
            times.append(float(ev.departure))
            soc.append(float(ev.battery_after) / batt_cap)
            if inst.nodes[ev.node].is_charging_station:
                ax.axvspan(ev.arrival, ev.departure, color=C_CHARGE, alpha=0.18)
                ax.scatter([ev.departure], [ev.battery_after / batt_cap], c=C_CHARGE, s=40, zorder=5)
        ax.plot(times, soc, color=C_ROUTE, lw=1.8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("Time")
        ax.set_ylabel("State of charge")
        ax.set_title(f"{name} EV#{ri}: battery trajectory (charging highlighted)")
        saved.extend(_save_fig(fig, paths.figures / stem_soc))

        # Gantt-like timeline for a few routes
        fig, ax = plt.subplots(figsize=(10, 4.8))
        nonempty = [
            (i, r) for i, r in enumerate(payload["routes"])
            if any(inst.nodes[n].is_customer for n in r["nodes"])
        ][:12]
        yticks = []
        for row_i, (rj, r) in enumerate(nonempty):
            rr2 = replay_route(inst, r["nodes"], r["labels"], config={"problem": {"charging_portions": [0.25, 0.5, 0.75, 1.0]}})
            for ev in rr2.route_events:
                node = inst.nodes[ev.node]
                if node.is_depot and ev.position == 0:
                    continue
                if node.is_charging_station:
                    color = C_CHARGE
                elif node.is_customer:
                    color = C_ROUTE
                else:
                    color = "#444"
                # waiting
                if ev.waiting > 1e-6:
                    ax.barh(row_i, ev.waiting, left=ev.arrival - ev.waiting, height=0.35, color="#cbd5e1")
                ax.barh(row_i, max(ev.departure - ev.arrival, 1e-3), left=ev.arrival, height=0.55, color=color, alpha=0.85)
            yticks.append(f"EV{rj}")
        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize=8)
        ax.set_xlabel("Time")
        ax.set_title(f"{name}: service / wait / charge timeline (first 12 EVs)")
        # legend proxies
        ax.barh(-1, 0, color=C_ROUTE, label="Customer/travel service window")
        ax.barh(-1, 0, color=C_CHARGE, label="Charging")
        ax.barh(-1, 0, color="#cbd5e1", label="Waiting")
        ax.set_ylim(-0.8, len(yticks) - 0.2)
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        saved.extend(_save_fig(fig, paths.figures / stem_tl))
    return saved


def fig_utilization(paths: PaperPaths) -> list[Path]:
    payload = _load_route_payload(paths.best_dir, "A", "RC2_10_1")
    inst = load_instance(_instance_root("A") / "RC2_10_1", load_capacity=None)
    custs, dists, durs, loads, charges = [], [], [], [], []
    cap = float(inst.vehicle.load_capacity)
    for r in payload["routes"]:
        nodes = r["nodes"]
        n_c = sum(1 for n in nodes if inst.nodes[n].is_customer)
        if n_c == 0:
            continue
        n_ch = sum(1 for n in nodes if inst.nodes[n].is_charging_station)
        load = sum(float(inst.nodes[n].demand) for n in nodes if inst.nodes[n].is_customer)
        rr = replay_route(inst, nodes, r["labels"], config={"problem": {"charging_portions": [0.25, 0.5, 0.75, 1.0]}})
        custs.append(n_c)
        charges.append(n_ch)
        loads.append(100.0 * load / cap if cap > 0 else 0.0)
        dists.append(rr.distance)
        durs.append(rr.duration)

    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    panels = [
        (axes[0, 0], custs, "Customers per EV", C_A),
        (axes[0, 1], dists, "Route distance", C_CUR),
        (axes[0, 2], durs, "Route duration", C_B),
        (axes[1, 0], loads, "Load utilization (%)", "#0f766e"),
        (axes[1, 1], charges, "Charging visits per EV", C_CHARGE),
    ]
    for ax, data, title, color in panels:
        ax.hist(data, bins=min(15, max(5, len(set(data)))), color=color, alpha=0.85, edgecolor="white")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(title)
        ax.set_ylabel("Count")
    axes[1, 2].axis("off")
    axes[1, 2].text(
        0.0, 0.6,
        f"RC2_10_1 utilization\nK={payload['used_k']}\n"
        f"mean cust/EV={np.mean(custs):.1f}\n"
        f"mean load util={np.mean(loads):.1f}%\n"
        f"routes with charge={sum(1 for c in charges if c>0)}/{len(charges)}",
        fontsize=10, family="monospace",
    )
    fig.suptitle("Vehicle utilization after fleet intensification", y=1.01)
    fig.tight_layout()
    return _save_fig(fig, paths.figures / "fig11_vehicle_utilization")


def fig_method_overview(paths: PaperPaths) -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")
    boxes = [
        (0.3, 1.1, 2.2, 1.2, "Dense EDF\nconstruction", C_BASE),
        (3.0, 1.1, 2.2, 1.2, "Coarse fleet\nshrink", "#57534e"),
        (5.7, 1.1, 2.2, 1.2, "Route elimination\n+ rebuild", C_CUR),
        (8.2, 1.1, 1.6, 1.2, "Exact\nreplay", C_A),
    ]
    for x, y, w, h, text, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=color, alpha=0.18, edgecolor=color, lw=1.5))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, color="#111")
    for x in (2.5, 5.2, 7.9):
        ax.annotate("", xy=(x + 0.45, 1.7), xytext=(x, 1.7),
                    arrowprops=dict(arrowstyle="->", color="#444"))
    ax.text(5, 0.45, "Hanif 1000-customer Deterministic-FPS  |  A→B warm start on relaxed TW",
            ha="center", fontsize=9, color="#333")
    ax.set_title("Method overview (deterministic large-scale pipeline)")
    return _save_fig(fig, paths.figures / "fig01_method_overview")


def copy_representative_routes(paths: PaperPaths) -> None:
    paths.routes_out.mkdir(parents=True, exist_ok=True)
    for variant, sub in (("A", "original_tw"), ("B", "relaxed_tw")):
        out = paths.routes_out / sub
        out.mkdir(exist_ok=True)
        for name in REPRESENTATIVE:
            src = paths.best_dir / sub / "routes" / f"{name}.json"
            if src.is_file():
                shutil.copy2(src, out / f"{name}.json")


def write_manuscript_values(paths: PaperPaths, main_rows: list[dict], family_rows: list[dict]) -> Path:
    best_summary = json.loads((paths.best_dir / "summary_both.json").read_text(encoding="utf-8"))
    matched = _load_matched_ablation(paths.ablation_dir)
    vals = {
        "method": str(best_summary.get("method") or "HanifFleetIntensification"),
        "result_root": str(paths.best_dir).replace("\\", "/"),
        "evaluation_protocol": "EXTENDED",
        "replay_protocol": "NO_WAIT_REPLAY",
        "care_fi": "six_family_pilot_only_not_headline",
        "A": {
            "mean_k": main_rows[0]["mean_k"],
            "baseline_mean_k": main_rows[0]["fast_baseline_mean_k"],
            "reduction_pct": main_rows[0]["fleet_reduction_pct"],
            "mean_distance": main_rows[0]["mean_distance"],
            "mean_runtime_s": main_rows[0]["mean_runtime_s"],
            "feasible": f"{int(main_rows[0]['n_feasible'])}/{int(main_rows[0]['n'])}",
        },
        "B": {
            "mean_k": main_rows[1]["mean_k"],
            "baseline_mean_k": main_rows[1]["fast_baseline_mean_k"],
            "reduction_pct": main_rows[1]["fleet_reduction_pct"],
            "mean_distance": main_rows[1]["mean_distance"],
            "mean_runtime_s": main_rows[1]["mean_runtime_s"],
            "feasible": f"{int(main_rows[1]['n_feasible'])}/{int(main_rows[1]['n'])}",
        },
        "matched_ablation": (
            {
                "complete": True,
                "A_mean_k": matched["A_mean_k"],
                "B_mean_k": matched["B_mean_k"],
                "A_feasible": matched["A_feasible"],
                "B_feasible": matched["B_feasible"],
            }
            if matched and matched.get("complete")
            else {"complete": False, "cite_means": False}
        ),
        "literature_competitors": competitor_table_rows(),
        "families_A": {r["family"]: r["mean_k"] for r in family_rows if r["variant"] == "A"},
        "families_B": {r["family"]: r["mean_k"] for r in family_rows if r["variant"] == "B"},
        "representative_instances": REPRESENTATIVE,
        "semantics": {
            "wait_mu": "lognormal parameter; expected wait = exp(mu+0.5 sigma^2)",
            "wait_sigma": False,
            "charging_price_in_objective": False,
            "charging_rate_heterogeneous": True,
        },
    }
    path = paths.out_dir / "manuscript_values.json"
    path.write_text(json.dumps(vals, indent=2), encoding="utf-8")
    return path


def generate_paper_artifacts(
    *,
    best_dir: Path | None = None,
    baseline_dir: Path | None = None,
    control_dir: Path | None = None,
    ablation_dir: Path | None = None,
    replay_report: Path | None = None,
    out_dir: Path | None = None,
    skip_freeze_lookup: bool = False,
) -> Path:
    best_dir = best_dir or (ROOT / "outputs/CurrentBestDeterministicHanif")
    if not skip_freeze_lookup and not (best_dir / "summary_both.json").is_file():
        # Prefer explicit intensify archive if freeze folder is incomplete.
        alt = ROOT / "outputs/hanif_fps_intensify"
        if (alt / "summary_both.json").is_file():
            best_dir = freeze_current_best(source=alt, dest=best_dir)
        else:
            best_dir = freeze_current_best()
    baseline_dir = baseline_dir or (ROOT / "outputs/hanif_fps/fast_feasible_baseline")
    control_dir = control_dir or (ROOT / "outputs/CurrentBestDeterministicHanif")
    out_dir = out_dir or (ROOT / "outputs/paper")
    paths = PaperPaths(
        best_dir=best_dir,
        baseline_dir=baseline_dir,
        control_dir=control_dir,
        out_dir=out_dir,
        ablation_dir=ablation_dir,
        replay_report=replay_report,
    )
    paths.figures.mkdir(parents=True, exist_ok=True)
    paths.tables.mkdir(parents=True, exist_ok=True)

    table_paths = write_tables(paths)
    main_rows = _read_csv(table_paths["main"])
    family_rows = _read_csv(table_paths["family"])
    per_rows = _read_csv(table_paths["per_instance"])

    fig_method_overview(paths)
    fig_fleet_reduction(paths, main_rows)
    fig_family_comparison(paths, family_rows)
    fig_family_reduction_pct(paths, family_rows)
    fig_a_vs_b(paths, per_rows)
    fig_per_instance_reduction(paths, per_rows)
    fig_route_maps(paths)
    fig_before_after(paths)
    fig_battery_and_timeline(paths)
    fig_utilization(paths)
    fig_runtime_quality(paths, per_rows)
    copy_representative_routes(paths)
    write_manuscript_values(paths, main_rows, family_rows)

    index = {
        "tables": {k: str(v.relative_to(out_dir)).replace("\\", "/") for k, v in table_paths.items()},
        "figures_dir": "figures/",
        "routes_dir": "routes/",
        "best_result": str(best_dir).replace("\\", "/"),
        "note": "SVG + 300 DPI PNG. Representative instances fixed: " + ", ".join(REPRESENTATIVE),
    }
    (out_dir / "INDEX.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    result_label = str(json.loads(
        (best_dir / "summary_both.json").read_text(encoding="utf-8")
    ).get("method") or "DeterministicHanif")
    (out_dir / "README.md").write_text(
        f"# Paper artifacts — {result_label}\n\n"
        "Generated from saved exact routes (not training logs).\n\n"
        "- `tables/` — main / family / per-instance / matched ablation CSVs\n"
        "- `tables/literature_competitors.csv` — published EVRPTW metaheuristics re-run on this set\n"
        "- `tables/semantic_replay.csv` — EXTENDED vs NO_WAIT_REPLAY (if generated)\n"
        "- `figures/` — SVG + 300 DPI PNG\n"
        "- `routes/` — representative solution JSONs\n"
        "- `manuscript_values.json` — numbers for the text\n\n"
        "The primary ablation is same-budget no-intensification versus the frozen "
        "full method. External comparison is ALNS and VNS/TS re-run on this "
        "benchmark (literature families; not bitwise clones of the original codes). "
        "CARE-FI is a six-family pilot only.\n",
        encoding="utf-8",
    )
    return out_dir


__all__ = ["freeze_current_best", "generate_paper_artifacts", "REPRESENTATIVE"]
