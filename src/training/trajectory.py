"""Trajectory storage for actor-critic training."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.env.reward import StepInfo


@dataclass
class EpisodeTransition:
    """One chronological transition: policy decision or environment auto-close."""

    is_policy_action: bool
    shaped_reward: float = 0.0
    raw_reward: float = 0.0
    shaping_reward: float = 0.0
    decoder_context: torch.Tensor | None = None
    feasible_mask: np.ndarray | None = None
    node: int = -1
    ev_id: int = -1
    log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    node_probs: torch.Tensor | None = None
    q_candidate_indices: np.ndarray | None = None
    q_candidate_probs: torch.Tensor | None = None
    step_info: StepInfo | None = None
    charge_portion: float | None = None
    action_type: str = "policy"


# Backward alias
StepRecord = EpisodeTransition


@dataclass
class Trajectory:
    transitions: list[EpisodeTransition] = field(default_factory=list)
    terminal_info: StepInfo | None = None
    objective: float = 0.0
    reconstructed_objective: float = 0.0
    reconstructed_original_objective: float = 0.0
    objective_mismatch: float = 0.0
    relative_objective_mismatch: float = 0.0
    original_objective_mismatch: float = 0.0
    relative_original_objective_mismatch: float = 0.0
    reward_breakdown: dict[str, float] = field(default_factory=dict)
    terminal_reward: float = 0.0
    pomo_group_reward: float | None = None
    pomo_advantage: float | None = None
    training_stats: dict[str, float] = field(default_factory=dict)
    node_embeddings: torch.Tensor | None = None

    @property
    def steps(self) -> list[EpisodeTransition]:
        """Policy-only steps (for backward compatibility)."""
        return [t for t in self.transitions if t.is_policy_action]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def policy_transitions(self) -> list[EpisodeTransition]:
        return [t for t in self.transitions if t.is_policy_action]

    def total_step_reward(self) -> float:
        return sum(t.shaped_reward for t in self.transitions) + (
            self.terminal_info.reward if self.terminal_info else 0.0
        )

    def stack_contexts(self) -> torch.Tensor:
        pol = self.policy_transitions()
        return torch.stack([t.decoder_context for t in pol if t.decoder_context is not None])

    def stack_log_probs(self) -> torch.Tensor:
        pol = self.policy_transitions()
        return torch.stack([t.log_prob for t in pol if t.log_prob is not None])

    def stack_entropies(self) -> torch.Tensor:
        pol = self.policy_transitions()
        return torch.stack([t.entropy for t in pol if t.entropy is not None])

    def reward_sequence(self) -> list[float]:
        """Chronological shaped rewards including env transitions + terminal."""
        rewards = [t.shaped_reward for t in self.transitions]
        if self.terminal_info is not None:
            rewards.append(float(self.terminal_info.reward))
        return rewards

    def value_indices(self) -> list[int]:
        """Indices in reward_sequence that have critic value predictions."""
        return [i for i, t in enumerate(self.transitions) if t.is_policy_action]

    def reward_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.reward_sequence(), dtype=torch.float32, device=device)

    def mean_feasible_actions(self) -> float:
        pol = self.policy_transitions()
        if not pol:
            return 0.0
        return float(np.mean([np.count_nonzero(t.feasible_mask) for t in pol if t.feasible_mask is not None]))

    def mean_q_candidates(self) -> float:
        counts = [
            len(t.q_candidate_indices)
            for t in self.policy_transitions()
            if t.q_candidate_indices is not None
        ]
        return float(np.mean(counts)) if counts else 0.0
