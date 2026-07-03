"""Verify actor loss receives gradients on log_probs."""
from __future__ import annotations

import numpy as np
import torch

from src.training.actor_critic import RLConfig, actor_critic_loss
from src.training.trajectory import StepRecord, Trajectory


def test_actor_loss_gradients_flow():
    device = "cpu"
    d = 8
    traj = Trajectory()
    ctx = torch.randn(d, requires_grad=False)
    log_p = torch.tensor(-1.2, requires_grad=True)
    ent = torch.tensor(0.5)
    traj.steps.append(
        StepRecord(
            decoder_context=ctx,
            feasible_mask=np.ones(5, dtype=bool),
            node=1,
            ev_id=0,
            log_prob=log_p,
            entropy=ent,
            shaped_reward=1.0,
        )
    )
    traj.terminal_info = None

    class DummyCritic(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    rl_cfg = RLConfig(normalize_advantages=False, use_gae=False, gamma=1.0)
    out = actor_critic_loss(traj, DummyCritic(), None, rl_cfg, device=device)
    assert out.actor_loss.requires_grad
    out.actor_loss.backward()
    assert log_p.grad is not None
    assert float(log_p.grad.abs()) > 0.0
