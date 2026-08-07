"""Checkpoint selection and policy score_candidates smoke."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.data.instance_loader import load_instance
from src.env.evrptw_env import EVRPTWEnv
from src.training.train_rl import build_policy
from src.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"


@pytest.mark.skipif(not SMALL.is_dir(), reason="c101C5 missing")
def test_score_candidates_bounded():
    config = load_config(str(ROOT / "configs" / "train_tgfps.yaml"))
    inst = load_instance(SMALL)
    portions = list((config.get("problem") or {}).get("charging_portions") or [1.0])
    policy = build_policy(config, portions)
    env = EVRPTWEnv(inst, m=3, charging_portions=portions)
    ev = env.select_acting_ev()
    assert ev is not None
    emb = policy.encode(inst, device="cpu")
    analysis = env.analyze_decision(ev)
    legal = [i for i, ok in enumerate(analysis.final_node_mask) if ok][:5]
    if not legal:
        pytest.skip("no legal candidates")
    logits = policy.score_candidates(env, ev, emb, legal, mask_result=analysis.mask_result)
    assert logits.shape[0] == len(legal)
    assert torch.isfinite(logits).all()
