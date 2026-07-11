"""Architecture invariant checks (Section 18)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.models.decision_state import DecisionState
from src.models.feature_specs import DYNAMIC_CANDIDATE_FEAT_DIM

DEBUG_DIR = Path("logs/architecture_assertions")


def _save_failure(record: dict) -> None:
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)
    path = DEBUG_DIR / "failures.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


def assert_decision_state(state: DecisionState, *, instance_name: str = "?") -> None:
    n = state.static_node_embeddings.shape[0]
    d = state.static_node_embeddings.shape[1]
    checks = [
        ("dynamic_feat_shape", state.dynamic_candidate_feats.shape == (n, DYNAMIC_CANDIDATE_FEAT_DIM)),
        ("candidate_emb_shape", state.candidate_embeddings.shape == (n, d)),
        ("feasible_mask_shape", state.feasible_mask.shape == (n,)),
        ("dynamic_finite", bool(torch.isfinite(state.dynamic_candidate_feats).all())),
        ("fused_finite", bool(torch.isfinite(state.candidate_embeddings).all())),
        ("decoder_finite", bool(torch.isfinite(state.decoder_context).all())),
        ("legal_actions", bool(state.feasible_mask.any())),
    ]
    for name, ok in checks:
        if not ok:
            _save_failure({"check": name, "instance": instance_name})
            raise AssertionError(f"architecture assertion failed: {name} ({instance_name})")

    if state.mask_result is not None:
        cm = state.mask_result.charge_masks
        for node, arr in cm.items():
            if arr.shape[0] != len(getattr(state.mask_result, "charge_masks", {})):
                pass
        dm = state.mask_result.depot_mode_mask
        if dm.shape[0] != 2:
            _save_failure({"check": "depot_mode_mask_shape", "instance": instance_name})
            raise AssertionError("depot_mode_mask must have length 2")
