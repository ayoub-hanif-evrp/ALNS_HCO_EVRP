"""Generate objective_v2 paper tables and the seven main-paper figures."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.visualization.objective_v2_paper import generate_all

FROZEN = ROOT / "outputs/CurrentBestDeterministicHanif"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="outputs/objective_v2")
    ap.add_argument("--figures", default="outputs/paper/figures")
    ap.add_argument("--tables", default="outputs/paper/tables")
    ap.add_argument("--scenarios", type=int, default=100)
    args = ap.parse_args()
    exp = ROOT / args.exp
    if exp.resolve() == FROZEN.resolve():
        raise SystemExit("Refusing to treat CurrentBest as the new experiment directory")
    report = generate_all(
        exp,
        figures_dir=ROOT / args.figures,
        tables_dir=ROOT / args.tables,
        n_scenarios=int(args.scenarios),
    )
    (exp / "paper_artifacts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
