"""Re-eval final checkpoints/best.pt on type-2 holdouts after early-stop."""
from __future__ import annotations

import json
from pathlib import Path

from _holdout_eval_type2 import eval_checkpoint
from src.utils.config import load_config


def main() -> None:
    run = Path("outputs/runs/run_20260729_140606")
    config = load_config(str(run / "config.yaml"))
    live = load_config("configs/train.yaml")
    config["fleet_search"] = live.get("fleet_search") or config.get("fleet_search")
    mstar = Path("outputs/demonstrations/schneider21_type2_v3_mstar.json")
    if mstar.is_file():
        config.setdefault("demonstrations", {})["mstar_cache"] = str(mstar)
    holdouts = ["c208_21", "r211_21", "rc208_21"]
    ckpt = run / "checkpoints" / "best.pt"
    out = eval_checkpoint(config, ckpt, holdouts, run / "evaluation" / "holdout_type2" / "final_best", "final_best")
    # Merge into all_summaries
    all_path = run / "evaluation" / "holdout_type2" / "all_summaries.json"
    data = json.loads(all_path.read_text(encoding="utf-8")) if all_path.is_file() else {}
    data["final_best"] = out
    all_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps({k: data[k] for k in data if k in ("final_best", "checkpoints_best", "bc_best")}, indent=2))


if __name__ == "__main__":
    main()
