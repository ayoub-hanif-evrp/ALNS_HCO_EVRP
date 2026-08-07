"""Fast DAgger smoke: 2 rounds on c101C5 only."""
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import apply_policy_action
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.search.candidate_dataset import collect_candidate_decisions_from_env
from src.training.dagger import run_dagger_rounds
from src.utils.config import load_config


def main() -> None:
    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    config = dict(config)
    config["device"] = "cpu"
    config["dataset"] = dict(config["dataset"])
    config["behavior_cloning"] = dict(config.get("behavior_cloning") or {})
    out_bc = ROOT / "outputs" / "tgfps_smoke" / "bc"
    out_bc.mkdir(parents=True, exist_ok=True)
    config["behavior_cloning"]["output_dir"] = str(out_bc)
    config.setdefault("pipeline", {})["ranker_max_per_epoch"] = 32

    from src.data import training_phases as tp

    orig = tp.resolve_training_phase_split
    tp.resolve_training_phase_split = lambda cfg: (["c101C5"], [], {"training_phase": "tiny"})

    inst = load_instance(ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5")
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(config))
    weights = CostWeights.from_eval_config(config)
    env = EVRPTWEnv(inst, m=4, charging_portions=[0.25, 0.5, 0.75, 1.0], hard_time_windows=True)
    decisions = []
    for _ in range(2):
        for d in collect_candidate_decisions_from_env(
            env, expert=expert, weights=weights, top_k=5, horizon=2
        ):
            decisions.append(asdict(d))
        ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_auto_close_from_analysis(ev, analysis) is not None:
            continue
        action = expert.select_action(env, ev, analysis)
        if action is None:
            break
        apply_policy_action(env, ev, action, mask_result=getattr(analysis, "mask_result", None))
    print("seed_decisions", len(decisions), flush=True)

    try:
        result = run_dagger_rounds(
            config,
            initial_decisions=decisions,
            rounds=2,
            betas=[1.0, 0.7],
            max_train_instances=1,
            max_steps=20,
        )
    finally:
        tp.resolve_training_phase_split = orig

    assert result["n_rounds"] >= 2
    assert len(result["history"]) >= 2
    out = ROOT / "outputs" / "tgfps_smoke" / "dagger_summary.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("OK n_rounds", result["n_rounds"], "n_decisions", result["n_decisions"], flush=True)
    print("ckpt", result["best_checkpoint"], flush=True)


if __name__ == "__main__":
    main()
