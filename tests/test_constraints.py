"""Hard time-window constraint config must reject dataset relaxation."""
from __future__ import annotations

import pytest

from src.env.constraints import ConstraintConfig
from src.training.config_validation import apply_safe_training_defaults, validate_training_config


def test_hard_original_defaults():
    cfg = apply_safe_training_defaults({})
    cc = ConstraintConfig.from_config(cfg)
    assert cc.hard_time_windows
    cc.validate_or_fail()


def test_relax_time_windows_raises():
    cfg = {"constraints": {"relax_time_windows": True}}
    with pytest.raises(ValueError, match="relax_time_windows"):
        ConstraintConfig.from_config(cfg).validate_or_fail()


def test_allow_late_service_raises_in_hard_mode():
    cfg = {"constraints": {"allow_late_service": True, "time_windows_mode": "hard_original"}}
    with pytest.raises(ValueError, match="allow_late_service"):
        ConstraintConfig.from_config(cfg).validate_or_fail()


def test_validate_training_config_rejects_relaxation():
    cfg = apply_safe_training_defaults({
        "dataset": {"train_instances": ["rc101_21"], "eval_instances": ["rc102_21"]},
        "constraints": {"relax_time_windows": True},
    })
    with pytest.raises(ValueError):
        validate_training_config(cfg)
