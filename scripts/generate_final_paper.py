"""Generate final paper tables and the six one-message figures.

Only a frozen-final run may write outputs/figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schneider.evaluation.output_paths import OUTPUTS
from src.schneider.visualization.final_paper_figures import generate


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--paper", default=None)
    ap.add_argument("--validation-only", action="store_true")
    ap.add_argument(
        "--frozen-final",
        action="store_true",
        help="Allow writing outputs/figures/. Required for manuscript figures.",
    )
    args = ap.parse_args()
    exp = Path(args.exp)
    if not exp.is_absolute():
        exp = ROOT / exp
    if args.paper:
        paper = ROOT / args.paper if not Path(args.paper).is_absolute() else Path(args.paper)
    else:
        paper = OUTPUTS if args.frozen_final else exp
    if paper.resolve() == OUTPUTS.resolve() and not args.frozen_final:
        raise SystemExit(
            "Refusing to write outputs/hanif/figures/ from a non-frozen run. "
            "Pass --frozen-final after a scientific PASS on the full 120, "
            "or pass --paper pointing at a separate experiment folder."
        )
    report = generate(exp, paper, frozen_final=bool(args.frozen_final))
    report["validation_only"] = bool(args.validation_only)
    report["frozen_final"] = bool(args.frozen_final)
    report["artifact_root"] = str(paper).replace("\\", "/")
    (exp / "paper_artifacts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
