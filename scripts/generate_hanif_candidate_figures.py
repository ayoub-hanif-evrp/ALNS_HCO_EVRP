"""Generate 15 PNG-only candidate figures from the saved Hanif master CSV.

  python scripts/generate_hanif_candidate_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.candidate_figures import PNG_DIR, generate  # noqa: E402


def main() -> int:
    generate()
    pngs = sorted(p.name for p in PNG_DIR.glob("*.png"))
    extras = [p.name for p in PNG_DIR.iterdir() if p.is_file() and p.suffix.lower() != ".png"]
    print(f"wrote {len(pngs)} PNG files to {PNG_DIR}")
    for name in pngs:
        print(f"  {name}")
    if extras:
        print("non-PNG files present (should be none):", extras, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
