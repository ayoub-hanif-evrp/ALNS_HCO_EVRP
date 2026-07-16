"""Consistency tests: masks, transitions, auto-close accounting, charge portions."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.charge_masks import feasible_charge_portions, safe_depot_return_from, simulate_recharge_departure
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.masks import customer_feasible, station_feasible
from src.env.reward import StepRewardCalculator, snapshot_env
from src.env.tw_diagnostics import customer_service_start, selected_station_is_necessary, urgent_rank_stats
from src.models.policy import TransformerEVRPTWPolicy
from src.training.actor_critic import RLConfig, actor_critic_loss
from src.training.rollout_ac import run_episode_trajectory
from src.training.rollout_transitions import record_transition_reward
from src.training.trajectory import DecisionEpochTransition, EpisodeTransition, Trajectory


def test_auto_close_included_in_objective_reconstruction(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1, use_horizon_check=True, hard_time_windows=True)
    weights = CostWeights()
    calc = StepRewardCalculator(weights, reward_scale=1000.0)
    step_infos = []

    ev = env.evs[0]
    j = inst.customer_indices[0]
    before_close = snapshot_env(env)
    env.apply_action(ev, j, None)
    before_close = snapshot_env(env)
    close_info = env.auto_close_route(ev, forced=True, return_info=True)
    record_transition_reward(
        env=env,
        before=before_close,
        info=close_info,
        reward_calculator=calc,
        train_calc=None,
        rl_cfg=None,
        step_infos=step_infos,
        phi_prev_holder=[0.0],
    )

    j_final, _, summary = env.score(weights)
    before_terminal = snapshot_env(env)
    terminal = calc.terminal_adjustment(before_terminal, summary)
    recon = sum(s.delta_objective for s in step_infos) + terminal.delta_objective
    rel = abs(recon - j_final) / max(1.0, abs(j_final))
    assert rel < 0.01


def test_customer_hard_tw_mask_matches_transition(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1, hard_time_windows=True)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    ok, _ = customer_feasible(
        inst, ev, j, env.served_mask, env.reserved_mask, False,
        env.use_depot_charging, env.charging_portions,
    )
    if ok:
        env.apply_action(ev, j, None)
        assert env.customer_lateness <= 1e-6


def test_station_requires_feasible_charge_portion(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    if not inst.charging_station_indices:
        pytest.skip("no stations")
    s = inst.charging_station_indices[0]
    ok, reason = station_feasible(
        inst, ev, s, env.served_mask, env.reserved_mask, True, True, env.charging_portions
    )
    if ok:
        mask, _ = feasible_charge_portions(
            inst, ev, s, env.served_mask, env.reserved_mask, portions=env.charging_portions
        )
        assert mask.any(), reason


def test_charge_mask_depot_return_requires_battery(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    if not inst.charging_station_indices:
        pytest.skip("no stations")
    s = inst.charging_station_indices[0]
    mask, reasons = feasible_charge_portions(
        inst, ev, s, env.served_mask, env.reserved_mask, portions=env.charging_portions,
    )
    depot = inst.depot_index
    for idx, ok in enumerate(mask):
        if not ok:
            continue
        portion = env.charging_portions[idx]
        departure, batt, _, _ = simulate_recharge_departure(inst, ev, s, portion, free=False)
        energy = float(inst.energy_matrix[s, depot])
        assert batt + 1e-6 >= energy, reasons[idx]
        ok_time, _ = safe_depot_return_from(inst, s, departure, batt)
        assert ok_time, reasons[idx]


def test_feasible_customer_has_safe_continuation(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    for j in inst.customer_indices:
        ok, reason = customer_feasible(
            inst, ev, j, env.served_mask, env.reserved_mask, True,
            env.use_depot_charging, env.charging_portions,
        )
        if not ok:
            continue
        _, _, ss, _ = customer_service_start(inst, ev, j)
        batt = ev.remaining_battery - inst.energy_matrix[ev.current_node, j]
        depot = inst.depot_index
        can_depot = (
            batt + 1e-6 >= inst.energy_matrix[j, depot]
            and ss + inst.travel_time_matrix[j, depot] <= inst.nodes[depot].due_time + 1e-6
        )
        can_station = False
        for s in inst.charging_station_indices:
            if batt + 1e-6 < inst.energy_matrix[j, s]:
                continue
            from src.env.ev_state import EVState

            sim = EVState(0, j, ss + inst.nodes[j].service_time, batt, ev.remaining_capacity)
            pm, _ = feasible_charge_portions(
                inst, sim, s, env.served_mask, env.reserved_mask, portions=env.charging_portions,
            )
            if pm.any():
                can_station = True
                break
        assert can_depot or can_station, reason


def test_autoclose_trajectory_objective_mismatch(small_instance, model_cfg):
    inst = small_instance
    policy = TransformerEVRPTWPolicy(model_cfg)
    weights = CostWeights()
    rl_cfg = RLConfig(reward_scale=1000.0, use_gae=True)
    _, traj = run_episode_trajectory(
        inst, policy, m=1, weights=weights, rl_cfg=rl_cfg,
        collect_trajectory=True, training_mode=True, mode="greedy",
    )
    assert traj is not None
    assert traj.relative_original_objective_mismatch < 0.01


def test_depot_mode_head_learnable(model_cfg):
    policy = TransformerEVRPTWPolicy(model_cfg)
    for p in policy.pointer.depot_mode_head.parameters():
        assert p.requires_grad
    d = policy.embedding_dim
    ctx = torch.randn(d, requires_grad=True)
    depot_emb = torch.randn(d)
    global_ctx = torch.randn(d)
    mask_ctx = torch.randn(d)
    logits = policy.pointer.depot_mode_logits(ctx, depot_emb, global_ctx, mask_ctx)
    assert logits.shape[-1] == 2
    lp = torch.log_softmax(logits, dim=-1)[0]
    assert lp.requires_grad
    lp.backward()
    assert ctx.grad is not None


def test_autoclose_gae_includes_env_transition(model_cfg, small_instance):
    device = "cpu"
    traj = Trajectory()
    ctx = torch.randn(8)
    log_p = torch.tensor(-0.5, requires_grad=True)
    # SMDP decision epoch aggregates the environment autoclose reward into shaped_reward.
    traj.decision_transitions.append(
        DecisionEpochTransition(
            decoder_context=ctx,
            total_log_prob=log_p,
            entropy=torch.tensor(0.1),
            shaped_reward=-5.0,
            raw_reward=-5.0,
            feasible_mask=np.ones(5, dtype=bool),
            action_node=1,
            ev_id=0,
            environment_events=[{"type": "auto_close", "shaped_reward": -5.0}],
        )
    )
    traj.terminal_info = None

    class DummyCritic(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0])

    out = actor_critic_loss(
        traj, DummyCritic(), None,
        RLConfig(use_gae=True, gamma=0.99, gae_lambda=0.95, normalize_advantages=False),
        device=device,
    )
    assert out.advantages.numel() == 1
    assert float(out.advantages.reshape(-1)[0]) != 0.0


def test_pre_action_slack_before_mutation(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    mask, _ = env.feasible_mask(ev)
    j = next((c for c in inst.customer_indices if mask[c]), None)
    if j is None:
        pytest.skip("no feasible customer")
    pre = urgent_rank_stats(env, ev, j, mask)
    pre_ctx = env.snapshot_decision_context(ev)
    env.apply_action(ev, j, None)
    post_ctx = env.snapshot_decision_context(ev)
    assert pre_ctx["time_before"] != post_ctx["time_before"] or pre_ctx["from_node"] != post_ctx["from_node"]
    assert pre.served_customer_slack >= -1e-6


def test_selected_station_necessity_is_station_specific(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    if not inst.charging_station_indices:
        pytest.skip("no stations")
    results = [
        selected_station_is_necessary(env, ev, s)[0]
        for s in inst.charging_station_indices
    ]
    assert isinstance(results[0], (bool, np.bool_))


def test_customer_and_depot_lateness_tracked_separately(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    j = inst.customer_indices[0]
    env.apply_action(ev, j, None)
    env.auto_close_route(ev, forced=True)
    assert env.customer_lateness >= 0.0
    assert env.depot_lateness >= 0.0
    assert abs(env.total_lateness - (env.customer_lateness + env.depot_lateness)) < 1e-6


def test_no_premature_autoclose_when_station_feasible(small_instance):
    """Auto-close must not trigger while any station/depot-recharge action is feasible."""
    inst = small_instance
    if not inst.charging_station_indices:
        pytest.skip("no stations")
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    ev.remaining_battery = inst.vehicle.battery_capacity * 0.2
    mask, _ = env.feasible_mask(ev)
    has_station = any(mask[s] for s in inst.charging_station_indices)
    has_depot_recharge = mask[inst.depot_index] and not env.all_customers_served()
    if not has_station and not has_depot_recharge:
        pytest.skip("no recharge escape at this battery level")
    assert env.try_auto_close_route(ev) is None
    assert not env.should_close_route(ev)


def test_depot_finish_counts_one_step(small_instance):
    """Policy depot-finish must not double-increment step_count via auto_close_route."""
    inst = small_instance
    env = EVRPTWEnv(inst, m=1)
    ev = env.evs[0]
    for j in inst.customer_indices:
        env.served_mask[j] = True
    env.num_served_customers = inst.num_customers
    if inst.customer_indices:
        ev.current_node = inst.customer_indices[0]
    mask, _ = env.feasible_mask(ev)
    if not mask[inst.depot_index]:
        pytest.skip("depot finish not feasible from this state")
    steps_before = env.step_count
    env.apply_action(ev, inst.depot_index, None, depot_finish=True)
    assert env.step_count == steps_before + 1


def test_hard_tw_customer_lateness_zero_after_feasible_service(small_instance):
    inst = small_instance
    env = EVRPTWEnv(inst, m=1, hard_time_windows=True)
    ev = env.evs[0]
    for j in inst.customer_indices:
        ok, _ = customer_feasible(
            inst, ev, j, env.served_mask, env.reserved_mask, True,
            env.use_depot_charging, env.charging_portions,
        )
        if not ok:
            continue
        env.apply_action(ev, j, None)
        assert env.customer_lateness <= 1e-6
        break
    else:
        pytest.skip("no feasible customer at start")
