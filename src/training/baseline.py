"""Reward baselines for variance reduction in REINFORCE."""
from __future__ import annotations


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
