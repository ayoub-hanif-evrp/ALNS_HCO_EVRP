"""Memory-efficient Q-critic evaluation over candidate action subsets."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.training.rl_memory import counterfactual_advantage_subset


def compute_q_on_candidates(
    q_critic: nn.Module,
    decoder_context: torch.Tensor,
    node_embeddings: torch.Tensor,
    candidate_indices: np.ndarray,
    *,
    q_eval_mode: str,
    q_chunk_size: int,
) -> torch.Tensor:
    """Return Q(s, a) for each candidate index (1D tensor length K)."""
    if candidate_indices.size == 0:
        return torch.zeros(0, device=node_embeddings.device)

    if q_eval_mode == "chunked_full":
        chunks = []
        for start in range(0, len(candidate_indices), q_chunk_size):
            chunk_idx = candidate_indices[start : start + q_chunk_size]
            emb = node_embeddings[torch.tensor(chunk_idx, device=node_embeddings.device, dtype=torch.long)]
            chunks.append(q_critic.forward_actions(decoder_context, emb))
        return torch.cat(chunks, dim=0)

    idx_t = torch.tensor(candidate_indices, device=node_embeddings.device, dtype=torch.long)
    emb = node_embeddings.index_select(0, idx_t)
    return q_critic.forward_actions(decoder_context, emb)


def q_loss_and_cf_for_record(
    q_critic: nn.Module,
    decoder_context: torch.Tensor,
    node_embeddings: torch.Tensor,
    candidate_indices: np.ndarray,
    candidate_probs: torch.Tensor,
    chosen_node: int,
    target_return: torch.Tensor,
    *,
    q_eval_mode: str,
    q_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Q regression loss + counterfactual advantage for one step."""
    q_vals = compute_q_on_candidates(
        q_critic,
        decoder_context,
        node_embeddings,
        candidate_indices,
        q_eval_mode=q_eval_mode,
        q_chunk_size=q_chunk_size,
    )
    probs = candidate_probs.to(device=q_vals.device, dtype=q_vals.dtype)
    local = np.where(candidate_indices == chosen_node)[0]
    local_chosen = int(local[0]) if len(local) else 0
    cf = counterfactual_advantage_subset(q_vals, candidate_indices, probs, local_chosen)
    q_loss = (q_vals[local_chosen] - target_return.detach()) ** 2
    return q_loss, cf
