"""Episode rollout for the learned policy under the event-driven environment."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.utils.logging_utils import get_logger

LOGGER = get_logger()


@dataclass
class EpisodeResult:
    env: EVRPTWEnv
    objective: float
    breakdown: dict
    summary: object
    log_prob_sum: torch.Tensor
    entropy_sum: torch.Tensor
    feasible: bool
    num_decisions: int = 0
    done_reason: str = "unknown"
    fleet_size: int = 0


def run_episode(
    instance,
    policy,
    m: int,
    weights: CostWeights,
    mode: str = "sampling",
    device: torch.device | str = "cpu",
    charging_portions: list[float] | None = None,
    use_horizon_check: bool = True,
    max_steps_factor: int = 6,
    min_decisions_warn: int | None = None,
) -> EpisodeResult:
    """Run a single event-driven episode and return result + RL signals.

    The encoder is invoked once; the decoder is re-entered at every event for
    the EV with the smallest current_time.
    """
    env = EVRPTWEnv(
        instance,
        m=m,
        charging_portions=charging_portions or policy.charging_portions,
        use_horizon_check=use_horizon_check,
        max_steps_factor=max_steps_factor,
    )
    node_embeddings = policy.encode(instance, device=device)

    log_prob_sum = torch.zeros((), device=device)
    entropy_sum = torch.zeros((), device=device)
    num_decisions = 0

    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            env.done_reason = "no_acting_ev"
            break
        feasible_mask, _ = env.feasible_mask(ev)
        if not feasible_mask.any():
            env.finish_idle_ev(ev)
            continue
        action = policy.select_action(env, ev, node_embeddings, feasible_mask, mode=mode)
        log_prob_sum = log_prob_sum + action.log_prob
        entropy_sum = entropy_sum + action.entropy
        num_decisions += 1
        env.apply_action(ev, action.node, action.charge_portion)

    done_reason = env.get_done_reason()
    if env.done_reason is None and env.is_done():
        done_reason = env.get_done_reason()

    if min_decisions_warn is not None and instance.num_customers >= 100:
        if num_decisions < min_decisions_warn:
            LOGGER.warning(
                "Suspiciously few decisions (%d) on instance %s (N=%d); done_reason=%s",
                num_decisions, getattr(instance, "instance_name", "?"),
                instance.num_customers, done_reason,
            )

    j, breakdown, summary = env.score(weights)
    return EpisodeResult(
        env=env,
        objective=j,
        breakdown=breakdown,
        summary=summary,
        log_prob_sum=log_prob_sum,
        entropy_sum=entropy_sum,
        feasible=env.feasible_solution(),
        num_decisions=num_decisions,
        done_reason=done_reason,
        fleet_size=m,
    )


def run_multi_sample(
    instance,
    policy,
    m: int,
    weights: CostWeights,
    n_samples: int = 16,
    device: torch.device | str = "cpu",
    **kwargs,
) -> EpisodeResult:
    """Sample N episodes and return the best feasible one (or least-cost)."""
    best: EpisodeResult | None = None
    for _ in range(n_samples):
        result = run_episode(instance, policy, m, weights, mode="sampling", device=device, **kwargs)
        if best is None or _is_better(result, best):
            best = result
    return best


def _is_better(a: EpisodeResult, b: EpisodeResult) -> bool:
    """Prefer feasible solutions; among same feasibility prefer lower objective."""
    if a.feasible != b.feasible:
        return a.feasible
    return a.objective < b.objective
