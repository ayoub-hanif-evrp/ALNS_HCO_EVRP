"""Assert Round-3 changes did not touch ALNS/HCO decision logic files vs origin/main."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROTECTED = [
    "src/alns-hco/solver.py",
    "src/alns-hco/phase1.py",
    "src/alns-hco/phase2.py",
    "src/alns-hco/hco.py",
    "src/alns-hco/destroy.py",
    "src/alns-hco/insert.py",
    "src/alns-hco/operators.py",
    "src/alns-hco/hyperparameters.py",
    "src/care-fi/benchmarks/alns.py",
    "src/care-fi/benchmarks/greedy.py",
]


def main() -> int:
    r = subprocess.run(
        ["git", "diff", "--name-only", "origin/main...HEAD"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    changed = {ln.strip().replace("\\", "/") for ln in r.stdout.splitlines() if ln.strip()}
    # also unstaged
    r2 = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    changed |= {ln.strip().replace("\\", "/") for ln in r2.stdout.splitlines() if ln.strip()}
    hits = [p for p in PROTECTED if p in changed]
    if hits:
        print("PROTECTED algorithmic files modified:", hits)
        return 1
    print("OK: no protected algorithmic files in diff vs origin/main / working tree")
    print("changed_count", len(changed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
