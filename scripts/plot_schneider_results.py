"""Publication figures for Schneider FR/PR (feasible rows only)."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import reviewer_root  # noqa: E402


def main() -> int:
    import matplotlib.pyplot as plt

    out = reviewer_root() / "schneider"
    path = out / "runs.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    rows = [r for r in csv.DictReader(path.open(encoding="utf-8")) if str(r.get("feasible")).lower() in {"true", "1"}]
    # mean distance by method × family
    buckets: dict[tuple, list] = defaultdict(list)
    for r in rows:
        if r.get("method") not in {"ALNS-FR", "ALNS-PR"}:
            continue
        if abs(float(r.get("budget_s") or 0) - 120.0) > 1e-9:
            continue
        buckets[(r["method"], r["family"])].append(float(r["distance"]))

    families = ["C1", "C2", "R2", "RC2"]  # feasible families in selected set
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    import numpy as np
    x = np.arange(len(families))
    w = 0.35
    for i, (method, color) in enumerate((("ALNS-FR", "#1f4e79"), ("ALNS-PR", "#c45911"))):
        means = []
        for fam in families:
            vals = buckets.get((method, fam)) or []
            means.append(sum(vals) / len(vals) if vals else float("nan"))
        ax.bar(x + (i - 0.5) * w, means, width=w, label=method, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel("Mean travel distance")
    ax.set_xlabel("Schneider family (feasible instances only)")
    ax.set_title("Schneider E-VRPTW: ALNS-FR vs ALNS-PR (120 s, seeds 0–2)")
    ax.legend(frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_schneider_fr_pr_distance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / "fig_schneider_fr_pr_distance.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
