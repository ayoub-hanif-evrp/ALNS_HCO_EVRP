"""Reward baselines for variance reduction in REINFORCE."""
from __future__ import annotations

import torch

from src.env.cost import CostWeights, training_reward
from src.training.rollout import run_episode


class GreedyRolloutBaseline:
    """Self-critical greedy-rollout baseline (Rennie et al. / Kool et al.).

    For each sampled solution, the baseline is the reward of a *greedy* (argmax)
    rollout of the current policy on the same instance and fleet size. The
    greedy rollout uses no gradient, so the advantage

        A = reward_sample - reward_greedy

    has low variance and needs no separate critic network. Greedy decoding is
    deterministic, so it is a strong, instance-specific reference.
    """

    def __init__(
        self,
        weights,
        charging_portions,
        use_horizon_check,
        max_steps_factor,
        reward_scale: float = 1000.0,
    ):
        self.weights = weights
        self.charging_portions = charging_portions
        self.use_horizon_check = use_horizon_check
        self.max_steps_factor = max_steps_factor
        self.reward_scale = reward_scale

    @torch.no_grad()
    def reward(self, instance, policy, m: int, device) -> float:
        result = run_episode(
            instance, policy, m, self.weights, mode="greedy", device=device,
            charging_portions=self.charging_portions,
            use_horizon_check=self.use_horizon_check,
            max_steps_factor=self.max_steps_factor,
        )
        return training_reward(result.objective, self.reward_scale)


class MovingAverageBaseline:
    """Exponential moving average of the batch-mean reward.

    baseline <- beta * baseline + (1 - beta) * batch_mean_reward
    """

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value: float | None = None

    def update(self, batch_mean_reward: float) -> float:
        if self.value is None:
            self.value = batch_mean_reward
        else:
            self.value = self.beta * self.value + (1.0 - self.beta) * batch_mean_reward
        return self.value

    def get(self) -> float:
        return 0.0 if self.value is None else self.value
