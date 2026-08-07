"""Run 6-instance neural validation gate, then optional full-9 if neural wins.

Usage:
  python scripts/run_neural_validation.py
  python scripts/run_neural_validation.py --full9
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.tgfps_eval import (
    FULL9,
    VALIDATION6,
    evaluate_methods_on_instances,
    run_validation_gate,
)
from src.utils.config import load_config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full9", action="store_true", help="Run full-9 only if neural passes gate")
    ap.add_argument("--force-full9-fallback", action="store_true",
                    help="Also evaluate deployed fallback on full-9 after gate")
    ap.add_argument("--m", type=int, default=30)
    args = ap.parse_args()

    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    # Cap ALNS for tractable validation
    hs = dict(config.get("hybrid_search") or {})
    alns = dict(hs.get("alns") or {})
    alns["iterations"] = int(alns.get("iterations", 25))
    alns["time_limit_seconds"] = float(alns.get("time_limit_seconds", 12))
    hs["alns"] = alns
    config = dict(config)
    config["hybrid_search"] = hs

    val = list((config.get("dataset") or {}).get("validation_instances") or VALIDATION6)
    assert len(val) == 6, f"need 6 validation instances, got {val}"
    print("Validation instances:", val, flush=True)

    out = ROOT / "outputs" / "tgfps_neural_validation"
    gate = run_validation_gate(config, validation_instances=val, m_default=args.m, out_dir=out)
    print("Selected:", gate["selected_method"], flush=True)
    print("Answer:", gate["answer"], flush=True)
    print(json.dumps(gate.get("gates"), indent=2, default=str), flush=True)

    if args.full9:
        if not gate.get("neural_passed_gate") and not args.force_full9_fallback:
            print("Skipping full-9: neural did not pass validation gate. "
                  "Deployed fallback remains Expert+Repair+ALNS "
                  "(see outputs/baselines/expert_repair_alns_20260801).", flush=True)
            return
        selected = gate["selected_method"]
        print("Full-9 with selected method:", selected, flush=True)
        full = evaluate_methods_on_instances(
            config,
            list(FULL9),
            methods=[selected],
            m_default=40,
            out_dir=out / "holdout_full9_selected",
            include_frozen_baseline=True,
        )
        (out / "full9_selected_summary.json").write_text(
            json.dumps({"selected": selected, "summaries": full.get("summaries"), "rows": full.get("rows")}, indent=2, default=str),
            encoding="utf-8",
        )
        print(json.dumps(full.get("summaries"), indent=2), flush=True)


if __name__ == "__main__":
    main()
