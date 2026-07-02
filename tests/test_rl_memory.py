"""Tests for memory-bounded Q candidate selection and RL config defaults."""
from __future__ import annotations

import numpy as np
import torch

from src.training.actor_critic import RLConfig
from src.training.rl_memory import select_q_candidates


def test_rl_config_safe_defaults():
    cfg = RLConfig.from_config({})
    assert cfg.use_counterfactual_critic is False
    assert cfg.use_pomo is False
    assert cfg.effective_pomo_k() == 1
    assert cfg.q_eval_mode == "topk"
    assert cfg.max_q_actions == 64
    assert cfg.detach_critic_inputs is True


def test_select_q_candidates_topk_includes_chosen():
    rl_cfg = RLConfig(max_q_actions=4, q_eval_mode="topk")
    mask = np.zeros(10, dtype=bool)
    mask[[1, 3, 5, 7, 9]] = True
    probs = torch.softmax(torch.randn(10), dim=0)
    chosen = 7
    cand = select_q_candidates(mask, probs, chosen, rl_cfg)
    assert chosen in cand
    assert len(cand) <= 4
    assert all(mask[i] for i in cand)


def test_q_critic_active_delayed():
    rl_cfg = RLConfig(use_counterfactual_critic=True, q_critic_enable_after_epoch=5)
    assert not rl_cfg.q_critic_active(4)
    assert rl_cfg.q_critic_active(5)
