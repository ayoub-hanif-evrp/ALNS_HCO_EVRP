#!/usr/bin/env python
"""One-time conversion: Hanif variant B (.txt) -> dataset/converted_hanif_data/ (fast npz/json)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.hanif_converter import convert_hanif_directory
from src.data.hanif_loader import list_hanif_txt_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Hanif EVRPTW .txt files to precomputed folders.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("dataset/Hanif/EVRPTW_ChargingStations_Dataset/variant_B_relaxed_TW"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/converted_hanif_data"),
    )
    parser.add_argument(
        "--ev-specs",
        type=Path,
        default=Path("dataset/Hanif/EVRPTW_ChargingStations_Dataset/ev_specs.json"),
    )
    parser.add_argument("--variant", type=str, default="variant_B_relaxed_TW")
    args = parser.parse_args()

    n = len(list_hanif_txt_files(args.source))
    print(f"Converting {n} instances from {args.source}")
    print(f"Output: {args.output}")
    t0 = time.time()
    paths = convert_hanif_directory(
        args.source,
        args.output,
        ev_specs_path=args.ev_specs,
        variant=args.variant,
    )
    elapsed = time.time() - t0
    manifest = {
        "source": str(args.source),
        "output": str(args.output),
        "variant": args.variant,
        "num_instances": len(paths),
        "elapsed_seconds": round(elapsed, 1),
        "instances": [p.name for p in paths],
    }
    (args.output / "conversion_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done: {len(paths)} instances in {elapsed:.1f}s")
    print(f"Manifest: {args.output / 'conversion_manifest.json'}")


if __name__ == "__main__":
    main()
