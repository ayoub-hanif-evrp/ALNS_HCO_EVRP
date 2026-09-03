"""Regenerate Hanif manuscript Figures 2–5 from the saved master CSV.

Does not run the solver. Does not overwrite the author's Figure 1.

  python scripts/generate_hanif_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.figures import AUTHOR_FIG01, generate_all_figures  # noqa: E402
from src.alns_hco.paper import _load  # noqa: E402
from src.common.paths import hanif_dirs  # noqa: E402


def main() -> int:
    dirs = hanif_dirs()
    master = dirs["raw"] / "master_results.csv"
    if not master.is_file():
        print(f"missing {master}", file=sys.stderr)
        return 1
    rows = _load(master)
    print(f"loaded {len(rows)} rows from {master}")
    generate_all_figures(dirs, rows, master_path=master)
    print("wrote Figures 2–5 and figure_manifest.md")
    print(AUTHOR_FIG01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
