"""Trajectory storage for actor-critic training."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.env.reward import StepInfo


@dataclass
class StepRecord:
    """One policy decision within an episode."""

    decoder_context: torch.Tensor
    feasible_mask: np.ndarray
    node: int
    ev_id: int
    log_prob: torch.Tensor
    entropy: torch.Tensor
    node_probs: torch.Tensor | None = None
    q_candidate_indices: np.ndarray | None = None
    q_candidate_probs: torch.Tensor | None = None
    reward: float = 0.0
    raw_reward: float = 0.0
    shaping_reward: float = 0.0
    shaped_reward: float = 0.0
    step_info: StepInfo | None = None
    charge_portion: float | None = None


@dataclass
class Trajectory:
    steps: list[StepRecord] = field(default_factory=list)
    terminal_info: StepInfo | None = None
    objective: float = 0.0
    reconstructed_objective: float = 0.0
    objective_mismatch: float = 0.0
    relative_objective_mismatch: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    terminal_reward: float = 0.0
    pomo_group_reward: float | None = None
    pomo_advantage: float | None = None
    training_stats: dict[str, float] = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def total_step_reward(self) -> float:
        return sum(s.shaped_reward for s in self.steps) + (
            self.terminal_info.reward if self.terminal_info else 0.0
        )

    def stack_contexts(self) -> torch.Tensor:
        return torch.stack([s.decoder_context for s in self.steps])

    def stack_log_probs(self) -> torch.Tensor:
        return torch.stack([s.log_prob for s in self.steps])

    def stack_entropies(self) -> torch.Tensor:
        return torch.stack([s.entropy for s in self.steps])

    def reward_tensor(self, device: torch.device | str) -> torch.Tensor:
        rewards = [s.shaped_reward for s in self.steps]
        if self.terminal_info is not None:
            rewards.append(self.terminal_info.reward)
        return torch.tensor(rewards, dtype=torch.float32, device=device)

    def mean_feasible_actions(self) -> float:
        if not self.steps:
            return 0.0
        return float(np.mean([np.count_nonzero(s.feasible_mask) for s in self.steps]))

    def mean_q_candidates(self) -> float:
        counts = [len(s.q_candidate_indices) for s in self.steps if s.q_candidate_indices is not None]
        return float(np.mean(counts)) if counts else 0.0
