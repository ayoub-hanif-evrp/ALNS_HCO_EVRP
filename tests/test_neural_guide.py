"""Unit tests for NeuralSearchGuide and fail-loud neural methods."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.search.hybrid import METHOD_NEURAL_REPAIR, solve_method
from src.search.neural_guide import NeuralCheckpointError, NeuralSearchGuide, resolve_neural_checkpoint
from src.training.gates import neural_no_regression_gate
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"
BC_CKPT = ROOT / "outputs" / "baselines" / "full9_20260731" / "checkpoints" / "bc_best_policy.pt"


def test_neural_checkpoint_missing_fails_loud(tmp_path):
    cfg = {"hybrid_search": {"neural_guidance": {"checkpoint": str(tmp_path / "missing.pt")}}}
    with pytest.raises(NeuralCheckpointError):
        resolve_neural_checkpoint(cfg, required=True)


def test_neural_gate_rejects_worse_feas():
    base = [{"feasible": True, "served_ratio": 1.0, "used_evs": 5, "distance": 10,
             "duration": 10, "charging_time": 1, "objective": 100, "runtime": 10}]
    chal = [{"feasible": False, "served_ratio": 1.0, "used_evs": 4, "distance": 5,
             "duration": 5, "charging_time": 0, "objective": 50, "runtime": 5}]
    g = neural_no_regression_gate("Det", base, "Neural", chal)
    assert g.accepted is False
    assert g.reason == "feasibility_regression"


def test_neural_gate_accepts_equal_faster():
    base = [{"feasible": True, "served_ratio": 1.0, "used_evs": 5, "distance": 10,
             "duration": 10, "charging_time": 1, "objective": 100, "runtime": 20}]
    chal = [{"feasible": True, "served_ratio": 1.0, "used_evs": 5, "distance": 10,
             "duration": 10, "charging_time": 1, "objective": 100, "runtime": 10}]
    g = neural_no_regression_gate("Det", base, "Neural", chal)
    assert g.accepted is True
    assert g.reason == "equal_quality_faster"


@pytest.mark.skipif(not BC_CKPT.is_file(), reason="frozen BC checkpoint absent")
@pytest.mark.skipif(not SMALL.is_dir(), reason="c101C5 fixture absent")
def test_neural_guide_uses_transformer_score_candidates():
    from src.data.instance_loader import load_instance
    from src.env.evrptw_env import EVRPTWEnv

    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    config = dict(config)
    config.setdefault("hybrid_search", {}).setdefault("neural_guidance", {})["checkpoint"] = str(BC_CKPT)
    guide = NeuralSearchGuide(config, checkpoint=BC_CKPT, device="cpu")
    inst = load_instance(SMALL)
    env = EVRPTWEnv(inst, m=3, charging_portions=[1.0], hard_time_windows=True)
    ev = env.select_acting_ev()
    assert ev is not None
    analysis = env.analyze_decision(ev)
    legal = [i for i, ok in enumerate(analysis.final_node_mask) if ok][:5]
    assert legal
    logits = guide.score_nodes(env, ev, legal, mask_result=analysis.mask_result)
    assert logits.shape[0] == len(legal)
    assert guide.telemetry.score_candidate_calls >= 1
    assert guide.telemetry.encode_calls >= 1
    tel = guide.telemetry.as_dict()
    assert tel["neural_participation"] is True
    assert tel["checkpoint_sha256"]


@pytest.mark.skipif(not BC_CKPT.is_file(), reason="frozen BC checkpoint absent")
@pytest.mark.skipif(not SMALL.is_dir(), reason="c101C5 fixture absent")
def test_neural_method_records_telemetry():
    from src.data.instance_loader import load_instance

    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    config = dict(config)
    hs = dict(config.get("hybrid_search") or {})
    alns = dict(hs.get("alns") or {})
    alns["iterations"] = 3
    alns["time_limit_seconds"] = 2
    hs["alns"] = alns
    hs.setdefault("neural_guidance", {})["checkpoint"] = str(BC_CKPT)
    config["hybrid_search"] = hs
    guide = NeuralSearchGuide(config, checkpoint=BC_CKPT, device="cpu")
    inst = load_instance(SMALL)
    res = solve_method(METHOD_NEURAL_REPAIR, inst, 4, config=config, guide=guide)
    assert res.method == METHOD_NEURAL_REPAIR
    assert res.telemetry.get("neural_participation") is True
    assert res.telemetry.get("neural_decisions", 0) + res.telemetry.get("score_candidate_calls", 0) > 0
    assert res.stages.get("neural_variants", 0) >= 1
