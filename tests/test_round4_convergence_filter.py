"""Round-4: primary 288 protocol excludes seed-2 and has cell size 24."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from src.alns_hco.reviewer_experiments import BALANCED_12
from src.common.paired_stats import paired_instance_effects


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ROOT / "outputs" / "reviewer_revision" / "fr_pr_convergence_primary288" / "runs.csv"
FALLBACK = ROOT / "outputs" / "reviewer_revision" / "fr_pr_convergence" / "runs.csv"


def _rows():
    path = PRIMARY if PRIMARY.is_file() else FALLBACK
    with path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # If using fallback before Round4 regen, filter
    out = []
    for r in rows:
        if r.get("instance") not in set(BALANCED_12):
            continue
        if int(float(r["seed"])) not in (0, 1):
            continue
        if str(r.get("feasible")).lower() not in {"true", "1"}:
            continue
        if r.get("method") not in ("ALNS-FR", "ALNS-PR"):
            continue
        out.append(r)
    return out


def test_primary_has_288_and_cell_24():
    rows = _rows()
    assert len(rows) == 288
    cells = defaultdict(int)
    for r in rows:
        cells[(r["method"], r["tw"], float(r["budget_s"]))] += 1
    assert len(cells) == 12
    assert set(cells.values()) == {24}


def test_paired_uses_exactly_two_matched_seeds():
    rows = _rows()
    eff = paired_instance_effects(
        rows, method_a="ALNS-FR", method_b="ALNS-PR", metric="distance",
        tw="original", budget_s=30.0, require_budget=True,
    )
    assert len(eff) == 12
    assert all(int(e["n_matched_seeds"]) == 2 for e in eff)
