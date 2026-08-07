"""Regenerate expert demos for curriculum (all classes) after type-1 smoke gate."""
from __future__ import annotations

import json
from pathlib import Path

from src.expert.generate_demonstrations import generate_demonstrations
from src.utils.config import load_config


def main() -> None:
    smoke_path = Path("outputs/expert_diagnostics/type1_smoke.json")
    force = False
    if smoke_path.is_file():
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        gate = bool(smoke.get("gate_pass") or (float(smoke.get("feasibility_rate", 0)) >= 0.5))
        print(
            f"type1 smoke served={smoke.get('mean_served')} feas={smoke.get('feasibility_rate')} gate={gate}",
            flush=True,
        )
        if not gate and not force:
            # Still refresh type-2 demos with the upgraded expert (safe teacher).
            config = load_config("configs/train.yaml")
            config["dataset"]["tw_family"] = "type2"
            config.setdefault("demonstrations", {})
            config["demonstrations"]["output"] = "outputs/demonstrations/schneider21_type2_v3_expert.pt"
            config["demonstrations"]["input"] = config["demonstrations"]["output"]
            config["demonstrations"]["eval_report"] = "outputs/demonstrations/schneider21_type2_v3_expert_eval.json"
            config["demonstrations"]["mstar_cache"] = "outputs/demonstrations/schneider21_type2_v3_mstar.json"
            print("Gate not passed — regenerating type-2 demos only.", flush=True)
            out = generate_demonstrations(config)
            print("Wrote", out, flush=True)
            return
    config = load_config("configs/train_curriculum.yaml")
    # Write episode-level demos to demonstrations.input; pipeline builds the
    # decision-level cache at demonstrations.output without re-rolling.
    demo = config.setdefault("demonstrations", {})
    episodes_out = demo.get("input") or "outputs/demonstrations/schneider21_curriculum_episodes.pt"
    demo["output"] = episodes_out
    print("Gate passed — regenerating full curriculum demos.", flush=True)
    print(f"Episode output → {episodes_out}", flush=True)
    out = generate_demonstrations(config)
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
