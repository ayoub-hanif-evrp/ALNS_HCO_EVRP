"""Matched baselines for the paper table: constructor, ALNS, VNS-TS.

Same 120 instances, 180 s budget, seed 0, classical fleet ranking,
corrected log-normal wait. Does not overwrite frozen archives.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.output_paths import ALNS, CONSTRUCTOR, VNS_TS

PYTHON = sys.executable
FPS = ROOT / "scripts/run_hanif_fps.py"
COMPILE = ROOT / "scripts/compile_paper_results.py"

JOBS = [
    {
        "name": "constructor",
        "out": CONSTRUCTOR,
        "extra": ["--disable-intensify", "--objective", "classical"],
    },
    {
        "name": "alns",
        "out": ALNS,
        "extra": ["--competitor", "alns", "--objective", "classical"],
    },
    {
        "name": "vns_ts",
        "out": VNS_TS,
        "extra": ["--competitor", "vns_ts", "--objective", "classical"],
    },
]


def complete(out: Path) -> bool:
    for tw in ("original_tw", "relaxed_tw"):
        csv_path = out / tw / "results.csv"
        if not csv_path.is_file():
            return False
        n = sum(1 for _ in csv_path.open(encoding="utf-8")) - 1
        if n < 60:
            return False
    return True


def main() -> int:
    workers = "8"
    budget = "180"
    for job in JOBS:
        out = job["out"]
        if complete(out):
            print(f"SKIP {job['name']} already complete at {out}", flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        cmd = [
            PYTHON, "-u", str(FPS),
            "--all", "--variant", "both",
            "--budget", budget, "--workers", workers,
            "--out", str(out),
            *job["extra"],
        ]
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))
    print("COMPILE tables", flush=True)
    subprocess.check_call([PYTHON, "-u", str(COMPILE)], cwd=str(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
