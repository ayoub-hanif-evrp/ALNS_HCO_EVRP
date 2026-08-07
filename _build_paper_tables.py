"""Build paper-facing Expert / BC / RL holdout tables from eval JSON summaries."""
from __future__ import annotations

import csv
import json
from pathlib import Path


TYPE2_HOLDOUTS = ["c208_21", "r211_21", "rc208_21"]
FULL_HOLDOUTS = [
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
]


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _row_from_summary(label: str, summary: dict) -> dict:
    return {
        "method": label,
        "n": int(summary.get("n") or summary.get("num_instances") or 0),
        "served": float(summary.get("mean_served") or summary.get("mean_served_ratio") or 0.0),
        "feas": float(summary.get("feasibility_rate") or 0.0),
        "J": float(summary.get("mean_objective") or 0.0),
    }


def _write_holdout_table(
    run_dir: Path,
    *,
    rows: list[dict],
    stem: str,
    title: str,
    footnote: str,
) -> Path:
    out = run_dir / "evaluation" / f"{stem}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "n", "served", "feas", "J"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    md = run_dir / "evaluation" / f"{stem}.md"
    lines = [
        f"# {title}",
        "",
        "| Method | Served | Feas | Mean J |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['method']} | {r['served']:.3f} | {r['feas']:.3f} | {r['J']:.1f} |"
        )
    lines.append("")
    lines.append(footnote)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def _rows_from_summaries_dir(root: Path) -> list[dict]:
    all_path = root / "all_summaries.json"
    rows: list[dict] = []
    if all_path.is_file():
        data = json.loads(all_path.read_text(encoding="utf-8"))
        for label, summary in data.items():
            rows.append(_row_from_summary(label, summary))
        return rows
    # Fallback: per-method summary JSON next to holdout report.
    for label, fname in (
        ("Expert", "expert_summary.json"),
        ("BC", "bc_summary.json"),
        ("RL-best", "rl_best_summary.json"),
        ("RL-last", "rl_last_summary.json"),
    ):
        summary = _load(root / fname)
        if summary:
            rows.append(_row_from_summary(label, summary))
    return rows


def build_type2_table(run_dir: Path) -> Path:
    root = run_dir / "evaluation" / "holdout_type2"
    rows = _rows_from_summaries_dir(root)
    return _write_holdout_table(
        run_dir,
        rows=rows,
        stem="paper_table_type2",
        title="Type-2 holdout table (greedy decode)",
        footnote=(
            "Note: type-2 only; do **not** claim full Schneider success from these numbers."
        ),
    )


def build_full9_table(run_dir: Path) -> Path:
    """Expert / BC / RL-best on the full 9-instance Schneider holdout."""
    rows: list[dict] = []
    # Pipeline multi-checkpoint comparison (preferred).
    cmp_path = run_dir / "evaluation" / "checkpoint_comparison.json"
    if cmp_path.is_file():
        data = json.loads(cmp_path.read_text(encoding="utf-8"))
        methods = data.get("methods") or {}
        label_map = {
            "bc_best": "BC",
            "bc_last": "BC-last",
            "rl_best_feasibility": "RL-best-feas",
            "rl_best_served": "RL-best-served",
            "rl_best_objective": "RL-best-objective",
            "rl_last": "RL-last",
        }
        selected = str(data.get("selected_on_validation") or "")
        for key, v in methods.items():
            if not isinstance(v, dict):
                continue
            label = label_map.get(key, key)
            if key == selected:
                label = f"{label}*"
            rows.append({
                "method": label,
                "n": len(FULL_HOLDOUTS),
                "served": float(v.get("macro_served") or 0.0),
                "feas": float(v.get("macro_feasibility") or 0.0),
                "J": float(v.get("macro_objective") or 0.0),
            })
        # Prefer hierarchical feasibility best as the paper RL column when present.
        if any(r["method"].startswith("RL-best-feas") for r in rows):
            # Keep all rows; footnote clarifies * = selected.
            pass
    if not rows:
        candidates = [
            run_dir / "evaluation" / "holdout_full9",
            run_dir / "evaluation" / "holdout",
            run_dir / "evaluation",
        ]
        for root in candidates:
            rows = _rows_from_summaries_dir(root)
            if rows:
                break
    if not rows:
        holdout = _load(run_dir / "holdout_eval_summary.json") or {}
        report = holdout.get("holdout_report") or holdout
        methods = report.get("methods") or report.get("by_checkpoint") or {}
        if isinstance(methods, dict) and methods:
            for label, summary in methods.items():
                if isinstance(summary, dict):
                    rows.append(_row_from_summary(str(label), summary))
        elif isinstance(report, dict) and (
            "mean_objective" in report or "feasibility_rate" in report
        ):
            rows.append(_row_from_summary("RL-best", report))
    # Attach expert train demo summary as a weak prior column if holdout expert missing.
    expert_metrics = _load(run_dir / "expert" / "expert_metrics.json")
    if expert_metrics and not any(str(r["method"]).lower().startswith("expert") for r in rows):
        rows.insert(0, _row_from_summary("Expert (train demos)", expert_metrics))
    preferred = [
        "Expert", "Expert (train demos)", "BC", "BC*", "RL-best-feas", "RL-best-feas*",
        "RL-best-served", "RL-best-objective", "final_best", "best", "RL-last", "last",
    ]
    order = {k.lower(): i for i, k in enumerate(preferred)}
    rows.sort(key=lambda r: order.get(str(r["method"]).lower(), 99))
    return _write_holdout_table(
        run_dir,
        rows=rows,
        stem="paper_table_full9",
        title="Full Schneider 9-holdout table (greedy decode, best ckpt)",
        footnote=(
            f"Holdouts: {', '.join(FULL_HOLDOUTS)}. "
            "* = checkpoint selected on validation/test protocol. "
            "Use hierarchical **best** checkpoint only; shaped reward is not paper J."
        ),
    )


def build_antidrift_cycle_table(run_dir: Path) -> Path:
    log = run_dir / "logs" / "train_log.jsonl"
    from collections import defaultdict

    by = defaultdict(list)
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = int(r.get("cycle", -1))
        if 1 <= c <= 8:
            by[c].append(r)
    rows = []
    for c in sorted(by):
        xs = by[c]
        rows.append({
            "cycle": c,
            "n": len(xs),
            "served": sum(x.get("served_rate", 0) for x in xs) / len(xs),
            "feas": sum(x.get("feasibility_rate", 0) for x in xs) / len(xs),
            "J": sum(x.get("final_objective", 0) for x in xs) / len(xs),
            "reward": sum(x.get("reward", 0) for x in xs) / len(xs),
        })
    # Hierarchical best = max feas, then served, then min J
    best = None
    best_key = None
    for r in rows:
        key = (-r["feas"], -r["served"], r["J"])
        if best_key is None or key < best_key:
            best_key = key
            best = r
    last = rows[-1] if rows else None
    out = {
        "cycles": rows,
        "best_cycle": best,
        "last_cycle": last,
        "restored_best_vs_last": {
            "best_cycle": None if best is None else best["cycle"],
            "last_cycle": None if last is None else last["cycle"],
            "served_delta": None if not (best and last) else best["served"] - last["served"],
            "feas_delta": None if not (best and last) else best["feas"] - last["feas"],
            "J_delta": None if not (best and last) else best["J"] - last["J"],
        },
        "success_bar_train": {
            "served_ge_0_98": bool(best and best["served"] >= 0.98),
            "feas_ge_0_55": bool(best and best["feas"] >= 0.55),
        },
    }
    path = run_dir / "evaluation" / "antidrift_cycle_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = run_dir / "evaluation" / "antidrift_cycle_summary.md"
    lines = [
        "# Anti-drift RL cycle summary",
        "",
        "| Cycle | n | Served | Feas | Mean J | Reward |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        mark = " **" if best and r["cycle"] == best["cycle"] else ""
        lines.append(
            f"| {r['cycle']}{mark} | {r['n']} | {r['served']:.4f} | {r['feas']:.4f} | "
            f"{r['J']:.1f} | {r['reward']:.3f} |"
        )
    lines.append("")
    if best and last:
        lines.append(
            f"Best hierarchical cycle **{best['cycle']}** vs last cycle **{last['cycle']}**: "
            f"served Δ={best['served']-last['served']:+.4f}, "
            f"feas Δ={best['feas']-last['feas']:+.4f}, "
            f"J Δ={best['J']-last['J']:+.1f}."
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


def main():
    import sys

    run = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/runs/run_20260729_140606")
    if not run.is_dir():
        raise SystemExit(f"Run dir not found: {run}")
    if (run / "logs" / "train_log.jsonl").is_file():
        print("Wrote", build_antidrift_cycle_table(run))
    if (run / "evaluation" / "holdout_type2" / "all_summaries.json").is_file():
        print("Wrote", build_type2_table(run))
    full9 = build_full9_table(run)
    print("Wrote", full9)
    # Type-2 subset slice note when full-9 exists
    print("Holdouts full-9:", ", ".join(FULL_HOLDOUTS))
    print("Holdouts type-2:", ", ".join(TYPE2_HOLDOUTS))


if __name__ == "__main__":
    main()
