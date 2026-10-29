"""Publication figures for Hanif main reused results."""
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
    import numpy as np

    out = reviewer_root() / "hanif_main_seeds012"
    rows = [r for r in csv.DictReader((out / "runs.csv").open(encoding="utf-8"))
            if str(r.get("feasible")).lower() in {"true", "1"}]
    methods = ["ALNS-FR", "ALNS-PR", "PhaseI-PR-15", "PhaseII-NoHCO", "PhaseII-Heterogeneous"]
    for tw in ("original", "relaxed"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for ax, metric, ylab in zip(
            axes, ("distance", "k"), ("Mean travel distance", r"Mean fleet size $K$"),
        ):
            xs = np.arange(len(methods))
            means = []
            for m in methods:
                vals = [float(r[metric]) for r in rows if r["method"] == m and r["tw"] == tw and r.get(metric) not in (None, "")]
                means.append(sum(vals) / len(vals) if vals else float("nan"))
            ax.bar(xs, means, color="#1f4e79")
            ax.set_xticks(xs)
            ax.set_xticklabels(methods, rotation=25, ha="right")
            ax.set_ylabel(ylab)
            ax.grid(True, axis="y", alpha=0.3)
        fig.suptitle(f"Hanif main benchmark (60 instances, seeds 0–2, TW={tw})")
        fig.tight_layout()
        fig.savefig(out / f"fig_hanif_main_{tw}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
    print("wrote figures to", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
