"""Summarize bundled Hanif CON/HC results for the paper context table."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.hanif_baselines import context_table_rows, summarize_hanif_baseline_csv


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        default="dataset/Hanif/EVRPTW_ChargingStations_Dataset/code/baseline_solver_results.csv",
    )
    ap.add_argument("--out", default="outputs/paper_ready/tables/hanif_context.csv")
    ap.add_argument("--json-out", default="outputs/paper_ready/hanif_context.json")
    args = ap.parse_args()
    summary = summarize_hanif_baseline_csv(ROOT / args.csv)
    rows = context_table_rows(summary)
    out_csv = ROOT / args.out
    out_json = ROOT / args.json_out
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2))
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
