"""Generate paper tables/figures from frozen Hanif Det-FPS results.

Usage:
  python scripts/generate_hanif_paper_figures.py
  python scripts/generate_hanif_paper_figures.py --best outputs/hanif_fps_intensify --out outputs/paper
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.hanif_paper_figures import freeze_current_best, generate_paper_artifacts


def main() -> int:
    ap = argparse.ArgumentParser(description="Build outputs/paper from frozen Hanif results")
    ap.add_argument(
        "--best",
        default="outputs/hanif_fps_intensify",
        help="Source intensify run (copied into CurrentBestDeterministicHanif)",
    )
    ap.add_argument(
        "--freeze-dest",
        default="outputs/CurrentBestDeterministicHanif",
        help="Permanent freeze folder (never overwrite with a worse run manually)",
    )
    ap.add_argument(
        "--baseline",
        default="outputs/hanif_fps/fast_feasible_baseline",
        help="Frozen fast-feasible baseline CSVs",
    )
    ap.add_argument(
        "--control",
        default="outputs/CurrentBestDeterministicHanif",
        help="Frozen full-method control",
    )
    ap.add_argument(
        "--ablation",
        default="outputs/hanif_fps_no_intensify_180",
        help="Same-budget no-intensification archive",
    )
    ap.add_argument("--replay", default=None, help="Optional semantic replay JSON")
    ap.add_argument("--out", default="outputs/paper")
    ap.add_argument("--skip-freeze", action="store_true", help="Use --best as-is without copying")
    args = ap.parse_args()

    best = ROOT / args.best
    freeze_dest = ROOT / args.freeze_dest
    if not args.skip_freeze:
        print(f"Freezing {best} -> {freeze_dest}")
        freeze_current_best(source=best, dest=freeze_dest)
        best_for_figs = freeze_dest
    else:
        best_for_figs = best

    print(f"Generating paper artifacts -> {ROOT / args.out}")
    ablation = ROOT / args.ablation
    out = generate_paper_artifacts(
        best_dir=best_for_figs,
        baseline_dir=ROOT / args.baseline,
        control_dir=ROOT / args.control,
        ablation_dir=ablation if ablation.is_dir() else None,
        replay_report=(ROOT / args.replay) if args.replay else None,
        out_dir=ROOT / args.out,
        skip_freeze_lookup=True,
    )
    print(f"Done: {out}")
    print(f"  tables:  {out / 'tables'}")
    print(f"  figures: {out / 'figures'}")
    print(f"  routes:  {out / 'routes'}")
    print(f"  values:  {out / 'manuscript_values.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
