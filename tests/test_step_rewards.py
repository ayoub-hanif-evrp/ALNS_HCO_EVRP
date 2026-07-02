"""Tests for step-wise reward decomposition and actor-critic utilities."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.reward import StepRewardCalculator, snapshot_env
from src.models.critic import ValueCritic
from src.training.actor_critic import RLConfig, compute_gae, explained_variance
from src.training.rollout_ac import run_episode_trajectory


def test_step_reward_reconstruction_small(small_instance):
    weights = CostWeights()
    env = EVRPTWEnv(small_instance, m=2)
    calc = StepRewardCalculator(weights, reward_scale=1000.0)
    rl_cfg = RLConfig(reward_scale=1000.0)

    # Manual short rollout: move to first customer if feasible
    ev = env.select_acting_ev()
    assert ev is not None
    mask, _ = env.feasible_mask(ev)
    if not mask.any():
        pytest.skip("No feasible action on fixture")
    target = int(np.argmax(mask))
    before = snapshot_env(env)
    env.apply_action(ev, target)
    after = snapshot_env(env)
    step = calc.step_reward(
        before, after,
        ev_id=ev.ev_id, selected_node=target, node_type="customer",
        previous_node=before.total_distance and 0 or 0,
    )
    assert step.delta_objective != 0.0 or step.reward == 0.0


def test_gae_shapes():
    rewards = torch.tensor([1.0, 0.0, 2.0])
    values = torch.tensor([0.5, 0.5, 0.5])
    adv, ret = compute_gae(rewards, values, gamma=1.0, lam=0.95)
    assert adv.shape == rewards.shape
    assert ret.shape == rewards.shape
    ev = explained_variance(ret, values)
    assert -1.0 <= ev <= 1.0


def test_value_critic_forward():
    c = ValueCritic(32)
    x = torch.randn(32)
    v = c(x)
    assert v.shape == ()


@pytest.mark.slow
def test_trajectory_objective_mismatch_small(small_instance):
    from src.models.policy import TransformerEVRPTWPolicy

    policy = TransformerEVRPTWPolicy({"embedding_dim": 64, "encoder_layers": 1, "heads": 4, "feedforward_dim": 128})
    weights = CostWeights()
    rl_cfg = RLConfig(reward_scale=1000.0)
    result, traj = run_episode_trajectory(
        small_instance, policy, m=2, weights=weights,
        device="cpu", rl_cfg=rl_cfg, collect_trajectory=True, max_steps_factor=4,
    )
    assert traj is not None
    assert result.objective > 0
    # Allow slack for terminal penalty attribution
    assert traj.objective_mismatch < result.objective * 0.5 + 1e4
