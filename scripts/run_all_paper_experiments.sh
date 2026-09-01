#!/usr/bin/env bash
set -euo pipefail
python scripts/run_all_paper_experiments.py --workers "${WORKERS:-8}" "$@"
python scripts/run_all_paper_experiments.py --compile-only
