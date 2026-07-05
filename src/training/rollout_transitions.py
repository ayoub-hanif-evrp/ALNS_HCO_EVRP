"""Helpers for recording environment transitions in actor-critic rollouts."""
from __future__ import annotations

from src.env.reward import StepInfo, StepRewardCalculator, snapshot_env
from src.training.reward_shaping import StepContext, TrainingRewardCalculator


def transition_node_type(info: dict) -> str:
    return str(info.get("type", info.get("action_type", "unknown")))


def record_transition_reward(
    *,
    env,
    before,
    info: dict,
    reward_calculator: StepRewardCalculator | TrainingRewardCalculator | None,
    train_calc: TrainingRewardCalculator | None,
    rl_cfg,
    step_infos: list,
    ctx: StepContext | None = None,
    phi_prev_holder: list,
    node_type: str | None = None,
    customer_served: bool | None = None,
) -> tuple[float, float, float, StepInfo | None]:
    """Compute objective-aligned reward for any env transition (policy or auto-close)."""
    after = snapshot_env(env)
    node_type = node_type or transition_node_type(info)
    served = customer_served if customer_served is not None else bool(info.get("customer_served", False))
    step_info = None
    raw_r = 0.0
    shaping = 0.0
    shaped_r = 0.0

    if reward_calculator is not None:
        if ctx is not None and hasattr(env, "progress_tracker"):
            ctx.steps_since_last_customer_service = env.progress_tracker.steps_since_last_customer_service
        base_kwargs = dict(
            ev_id=int(info.get("ev_id", -1)),
            selected_node=int(info.get("node", -1)),
            node_type=node_type,
            previous_node=int(info.get("previous_node", info.get("from_node", -1))),
            delta_travel_time=float(info.get("delta_travel_time", 0.0)),
            delta_energy_used=float(info.get("delta_energy_used", 0.0)),
            charged_energy=float(info.get("charged_energy", 0.0)),
            customer_served=served,
        )
        if isinstance(reward_calculator, TrainingRewardCalculator):
            step_info = reward_calculator.base.step_reward(before, after, **base_kwargs)
            step_info = reward_calculator.shape_step(
                step_info, before=before, after=after,
                node_type=node_type, customer_served=served, ctx=ctx or StepContext(),
            )
        else:
            step_info = reward_calculator.step_reward(before, after, **base_kwargs)

        raw_r = step_info.raw_reward
        phi_prev = phi_prev_holder[0]
        if rl_cfg is not None and rl_cfg.use_potential_shaping:
            from src.training.actor_critic import apply_potential_shaping, compute_potential

            phi_new = compute_potential(env, rl_cfg)
            shaped_r, shaping = apply_potential_shaping(raw_r, phi_prev, phi_new, rl_cfg.gamma)
            step_info.shaping_reward = shaping
            step_info.reward = shaped_r + (step_info.reward - raw_r)
            phi_prev_holder[0] = phi_new
        else:
            shaped_r = step_info.reward
            step_info.shaping_reward = 0.0
        step_infos.append(step_info)

    return raw_r, shaping, shaped_r, step_info
