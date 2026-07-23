"""BC masked loss and checkpoint compatibility smoke tests."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.training.behavior_cloning import masked_ce_from_logits


def test_bc_masked_ce_rejects_illegal_target():
    logits = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    with pytest.raises(RuntimeError, match="illegal"):
        masked_ce_from_logits(logits, 1, mask)


def test_bc_masked_ce_legal_target():
    logits = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    loss = masked_ce_from_logits(logits, 2, mask)
    assert torch.isfinite(loss)
    assert float(loss) >= 0.0


def test_feature_dim_matches_dynamic_stack():
    from src.models.feature_specs import DYNAMIC_CANDIDATE_FEAT_DIM

    assert DYNAMIC_CANDIDATE_FEAT_DIM == 23
