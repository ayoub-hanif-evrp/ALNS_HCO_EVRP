"""RL memory validation, Q-candidate selection, and GPU diagnostics."""
from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import torch

from src.utils.logging_utils import get_logger

if TYPE_CHECKING:
    from src.training.actor_critic import RLConfig

LOGGER = get_logger()


def validate_rl_memory_config(config: dict, *, num_nodes: int, num_customers: int) -> None:
    """Warn or adjust when heavy RL options are combined on large instances."""
    rcfg = config.get("rl") or {}
    mcfg = config.get("model") or {}
    use_q = bool(rcfg.get("use_counterfactual_critic", False))
    use_ms = bool(rcfg.get(
        "use_multi_sample_rollout_baseline",
        rcfg.get("use_pomo", False),
    ))
    ms_k = int(rcfg.get("multi_sample_K", rcfg.get("pomo_K", 1 if not use_ms else 4)))
    if not use_ms:
        ms_k = min(ms_k, 1)
    edge = bool(mcfg.get("use_edge_aware_encoder", False))

    large = num_nodes > 1000 or num_customers >= 500
    heavy_count = sum([use_q, ms_k >= 4, edge])

    if large and heavy_count >= 3:
        LOGGER.warning(
            "Counterfactual Q critic + multi-sample rollout baseline K=%d + edge encoder on N=%d is memory intensive. "
            "Consider q_eval_mode=topk, max_q_actions<=128, lower multi_sample_K, or disabling edge encoder.",
            ms_k, num_nodes,
        )
    if large and use_q and ms_k >= 4 and edge:
        LOGGER.error(
            "Unsafe memory config: Q critic + multi-sample K>=4 + edge encoder on N=%d. "
            "Reduce multi_sample_K, disable use_edge_aware_encoder, or set use_counterfactual_critic=false.",
            num_nodes,
        )


def log_memory_plan(
    *,
    instance_name: str,
    num_nodes: int,
    num_customers: int,
    num_stations: int,
    rl_cfg: "RLConfig",
    use_edge_aware: bool,
    use_amp: bool,
    gradient_accumulation_steps: int,
) -> None:
    risk = _memory_risk(num_nodes, rl_cfg, use_edge_aware)
    LOGGER.info(
        "Memory plan | %s | N=%d (cust=%d, sta=%d) | multi-sample K=%d | Q=%s mode=%s max_q=%d chunk=%d | "
        "edge_enc=%s | amp=%s | grad_accum=%d | detach_critic=%s | risk=%s",
        instance_name,
        num_nodes,
        num_customers,
        num_stations,
        rl_cfg.effective_multi_sample_k(),
        rl_cfg.use_counterfactual_critic,
        rl_cfg.q_eval_mode,
        rl_cfg.max_q_actions,
        rl_cfg.q_chunk_size,
        use_edge_aware,
        use_amp,
        gradient_accumulation_steps,
        rl_cfg.detach_critic_inputs,
        risk,
    )


def _memory_risk(num_nodes: int, rl_cfg: "RLConfig", use_edge: bool) -> str:
    score = 0
    if num_nodes > 1000:
        score += 2
    if rl_cfg.effective_multi_sample_k() >= 4:
        score += 2
    if rl_cfg.use_counterfactual_critic and rl_cfg.q_eval_mode == "chunked_full":
        score += 2
    elif rl_cfg.use_counterfactual_critic:
        score += 1
    if use_edge and num_nodes > 800:
        score += 2
    if score >= 5:
        return "high"
    if score >= 3:
        return "medium"
    return "low"


def cuda_memory_stats(device: torch.device | str) -> dict[str, float]:
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    return {
        "cuda_mem_allocated_mb": torch.cuda.memory_allocated(device) / (1024 ** 2),
        "cuda_mem_reserved_mb": torch.cuda.memory_reserved(device) / (1024 ** 2),
        "cuda_max_mem_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024 ** 2),
    }


def reset_cuda_peak_stats(device: torch.device | str) -> None:
    if isinstance(device, str):
        device = torch.device(device)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def select_q_candidates(
    feasible_mask: np.ndarray,
    node_probs: torch.Tensor,
    chosen: int,
    rl_cfg: "RLConfig",
    rng: random.Random | None = None,
) -> np.ndarray:
    """Build candidate action set for memory-bounded counterfactual Q."""
    feas_idx = np.where(feasible_mask)[0]
    if feas_idx.size == 0:
        return np.array([chosen], dtype=np.int64)

    k = min(int(rl_cfg.max_q_actions), int(feas_idx.size))
    probs_cpu = node_probs.detach().float().cpu().numpy()

    if rl_cfg.q_eval_mode == "sampled":
        rng = rng or random.Random(0)
        weights = probs_cpu[feas_idx].clip(min=1e-12)
        weights = weights / weights.sum()
        n_sample = max(1, k - 1)
        sampled = rng.choices(list(feas_idx), weights=list(weights), k=n_sample)
        cand = list(dict.fromkeys([chosen, *sampled]))
        return np.array(cand[:k], dtype=np.int64)

    if rl_cfg.q_eval_mode == "chunked_full":
        return feas_idx.astype(np.int64)

    # topk (default): selected + top feasible by actor probability
    order = feas_idx[np.argsort(-probs_cpu[feas_idx])]
    cand = list(order[:k])
    if chosen not in cand:
        cand = [chosen] + cand[:-1]
    return np.unique(np.array(cand, dtype=np.int64))


def counterfactual_advantage_subset(
    q_values: torch.Tensor,
    candidate_indices: np.ndarray,
    candidate_probs: torch.Tensor,
    chosen_local_idx: int,
) -> torch.Tensor:
    """Approximate counterfactual baseline over candidate set only."""
    denom = candidate_probs.sum().clamp_min(1e-8)
    baseline = (candidate_probs * q_values).sum() / denom
    return q_values[chosen_local_idx] - baseline
