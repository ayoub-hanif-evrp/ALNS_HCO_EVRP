"""Trajectory storage for semi-Markov decision-epoch actor-critic training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from src.env.reward import StepInfo


@dataclass
class DecisionEpochTransition:
    """One policy decision epoch (Section 1)."""

    state_features: Any = None
    action_node: int = -1
    charge_portion: float | None = None
    depot_finish: bool = False

    node_log_prob: torch.Tensor | None = None
    secondary_log_prob: torch.Tensor | None = None
    total_log_prob: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    value_prediction: torch.Tensor | None = None

    raw_reward: float = 0.0
    shaped_reward: float = 0.0
    total_reward: float = 0.0
    original_cost_delta: float = 0.0

    elapsed_time: float = 0.0
    next_state_features: Any = None
    terminal: bool = False
    termination_reason: str | None = None
    environment_events: list[dict] = field(default_factory=list)

    # Legacy / training fields
    decoder_context: torch.Tensor | None = None
    decision_state: Any = None
    feasible_mask: np.ndarray | None = None
    mask_result: object | None = None
    ev_id: int = -1
    action_type: str = "policy"
    step_info: StepInfo | None = None
    node_probs: torch.Tensor | None = None
    q_candidate_indices: np.ndarray | None = None
    q_candidate_probs: torch.Tensor | None = None

    @property
    def log_prob(self) -> torch.Tensor | None:
        return self.total_log_prob

    @property
    def aggregated_raw_reward(self) -> float:
        return self.raw_reward

    @property
    def aggregated_shaped_reward(self) -> float:
        return self.shaped_reward

    @property
    def aggregated_shaping_reward(self) -> float:
        return self.total_reward - self.raw_reward

    @property
    def environment_subtransitions(self) -> list[dict]:
        return self.environment_events

    @property
    def node(self) -> int:
        return self.action_node


DecisionTransition = DecisionEpochTransition


@dataclass
class EpisodeTransition:
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


StepRecord = EpisodeTransition


@dataclass
class Trajectory:
    transitions: list[EpisodeTransition] = field(default_factory=list)
    decision_transitions: list[DecisionEpochTransition] = field(default_factory=list)
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
    multi_sample_group_reward: float | None = None
    multi_sample_advantage: float | None = None
    training_stats: dict[str, float] = field(default_factory=dict)
    terminal_feasible: bool | None = None
    num_customers: int = 0
    node_embeddings: torch.Tensor | None = None

    @property
    def steps(self) -> list[EpisodeTransition]:
        return [t for t in self.transitions if t.is_policy_action]

    @property
    def num_steps(self) -> int:
        return len(self.decision_transitions)

    def policy_transitions(self) -> list[EpisodeTransition]:
        return [t for t in self.transitions if t.is_policy_action]

    def smdp_decisions(self) -> list[DecisionEpochTransition]:
        return self.decision_transitions

    def smdp_reward_sequence(self) -> list[float]:
        return [d.shaped_reward for d in self.decision_transitions]

    def stack_smdp_contexts(self) -> torch.Tensor:
        return torch.stack([d.decoder_context for d in self.decision_transitions if d.decoder_context is not None])

    def stack_smdp_log_probs(self) -> torch.Tensor:
        return torch.stack([d.total_log_prob for d in self.decision_transitions if d.total_log_prob is not None])

    def stack_smdp_entropies(self) -> torch.Tensor:
        return torch.stack([d.entropy for d in self.decision_transitions if d.entropy is not None])

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
        rewards = [t.shaped_reward for t in self.transitions]
        if self.terminal_info is not None:
            rewards.append(float(self.terminal_info.reward))
        return rewards

    def value_indices(self) -> list[int]:
        return [i for i, t in enumerate(self.transitions) if t.is_policy_action]

    def reward_tensor(self, device: torch.device | str) -> torch.Tensor:
        return torch.tensor(self.reward_sequence(), dtype=torch.float32, device=device)

    def mean_feasible_actions(self) -> float:
        if not self.decision_transitions:
            return 0.0
        return float(np.mean([
            np.count_nonzero(d.feasible_mask) for d in self.decision_transitions if d.feasible_mask is not None
        ]))

    def mean_q_candidates(self) -> float:
        counts = [
            len(d.q_candidate_indices)
            for d in self.decision_transitions
            if d.q_candidate_indices is not None
        ]
        return float(np.mean(counts)) if counts else 0.0
