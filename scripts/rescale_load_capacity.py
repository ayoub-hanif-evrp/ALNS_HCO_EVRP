"""Set vehicle load capacity (C) on all converted E-VRPTW instances.

Updates evrp_instance.json, nodes.csv, and summary.json so on-disk data matches
the configured capacity (default 100).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def patch_instance(instance_dir: Path, load_capacity: float) -> None:
    json_path = instance_dir / "evrp_instance.json"
    csv_path = instance_dir / "nodes.csv"
    summary_path = instance_dir / "summary.json"

    data = json.loads(json_path.read_text(encoding="utf-8"))
    data["vehicle_parameters"]["load_capacity"] = load_capacity
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["demand_norm"] = df["demand"].astype(float) / load_capacity
        df.to_csv(csv_path, index=False)

    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["load_capacity"] = load_capacity
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rescale vehicle load capacity on all instances.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset/evrptw_instances"),
        help="Root folder of converted instances",
    )
    parser.add_argument(
        "--load-capacity",
        type=float,
        default=100.0,
        help="Target vehicle load capacity C (default: 100)",
    )
    args = parser.parse_args()

    dirs = sorted(
        d for d in args.dataset.iterdir() if d.is_dir() and (d / "evrp_instance.json").exists()
    )
    if not dirs:
        raise SystemExit(f"No instances under {args.dataset}")

    for d in dirs:
        patch_instance(d, args.load_capacity)
    print(f"Updated load_capacity={args.load_capacity} on {len(dirs)} instances.")


if __name__ == "__main__":
    main()
