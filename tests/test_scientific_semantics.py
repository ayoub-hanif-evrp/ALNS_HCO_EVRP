"""Scientific semantics: hard vs pruning, serviceability, SMDP, baselines."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.action_masks import compute_action_masks
from src.env.evrptw_env import EVRPTWEnv
from src.env.mask_layers import (
    ActionPruningConfig,
    combine_mask_layers,
    compute_hard_feasibility_mask,
    compute_optional_pruning_mask,
)
from src.env.policy_action import PolicyAction, apply_policy_action
from src.env.serviceability import compute_ev_customer_serviceability, compute_global_serviceability_recharge_aware
from src.evaluation.baselines import NearestFeasiblePolicy, RandomFeasiblePolicy, run_heuristic_episode
from src.models.critic import build_trajectory_auxiliary_targets
from src.models.dynamic_candidate import compute_dynamic_candidate_features
from src.models.policy import TransformerEVRPTWPolicy
from src.models.normalization import FeatureNormalizer


def test_hard_vs_pruning_station_separation(small_instance):
    env = EVRPTWEnv(small_instance, m=2)
    env.environment_config = {"action_pruning": {"enabled": True, "prune_dominated_stations": True}}
    ev = env.evs[0]
    gs = env.get_global_serviceability(force=True)
    hard, hard_reasons, _, _, hard_depot, _ = compute_hard_feasibility_mask(
        small_instance, env, ev,
        use_horizon_check=True, use_depot_charging=True, charging_portions=env.charging_portions,
    )
    prune, prune_reasons, _, _, _, _ = compute_optional_pruning_mask(
        small_instance, env, ev, hard, hard_reasons,
        hard_depot_mode=hard_depot,
        global_svc=gs,
        pruning=ActionPruningConfig(enabled=True, prune_dominated_stations=True),
        use_horizon_check=True,
        use_depot_charging=True,
        charging_portions=env.charging_portions,
    )
    final, _ = combine_mask_layers(hard, prune, hard_reasons, prune_reasons)
    for s in small_instance.charging_station_indices:
        if hard[s] and not prune[s]:
            assert not final[s]
            assert s in prune_reasons or s in hard_reasons


def test_depot_finish_masked_when_hard_customer_exists(small_instance):
    env = EVRPTWEnv(small_instance, m=1)
    ev = env.evs[0]
    result = compute_action_masks(env, ev)
    if result.hard_feasible_customer_count > 0:
        assert not result.feasible_depot_finish


def test_steps_since_last_customer_service_counter(small_instance):
    env = EVRPTWEnv(small_instance, m=1)
    ev = env.evs[0]
    ev.steps_since_last_customer_service = 0
    mask = compute_action_masks(env, ev)
    customers = [c for c in small_instance.customer_indices if mask.node_mask[c]]
    if not customers:
        pytest.skip("no feasible customer")
    apply_policy_action(
        env, ev,
        PolicyAction(node=customers[0], charge_portion=None, depot_finish=False),
        mask_result=mask,
    )
    assert ev.steps_since_last_customer_service == 0
    mask2 = compute_action_masks(env, ev)
    stations = [s for s in small_instance.charging_station_indices if mask2.node_mask[s]]
    if stations:
        node = stations[0]
        portion = env.charging_portions[int(np.argmax(mask2.charge_masks[node]))]
        apply_policy_action(
            env, ev,
            PolicyAction(node=node, charge_portion=portion, depot_finish=False),
            mask_result=mask2,
        )
        assert ev.steps_since_last_customer_service >= 1


def test_expired_count_updates(small_instance):
    env = EVRPTWEnv(small_instance, m=1)
    before = env.last_expired_count
    env.bump_state_version()
    assert env.last_expired_count >= 0
    assert isinstance(env.last_expired_customer_ids, list)
    assert env.last_expired_count >= before or before == 0


def test_dynamic_features_agree_with_mask(small_instance):
    env = EVRPTWEnv(small_instance, m=2)
    ev = env.evs[0]
    mask = compute_action_masks(env, ev)
    feats = compute_dynamic_candidate_features(env, ev, mask)
    for j in small_instance.customer_indices:
        if feats[j, 12] < 0.5:
            assert feats[j, 15] == 0.0 or mask.hard_feasible_station_count == 0


def test_baseline_actions_respect_masks(small_instance):
    env = EVRPTWEnv(small_instance, m=2, use_depot_charging=True)
    env.environment_config = {"action_pruning": {"enabled": True}}
    run_heuristic_episode(env, NearestFeasiblePolicy())
    run_heuristic_episode(env, RandomFeasiblePolicy(seed=1))


def test_critic_remaining_cost_targets():
    from src.training.trajectory import DecisionEpochTransition, Trajectory

    traj = Trajectory(num_customers=5, terminal_feasible=True)
    traj.decision_transitions = [
        DecisionEpochTransition(original_cost_delta=1.0),
        DecisionEpochTransition(original_cost_delta=2.0),
        DecisionEpochTransition(original_cost_delta=3.0),
    ]
    targets = build_trajectory_auxiliary_targets(
        traj, num_customers=5, terminal_feasible=True, device="cpu",
    )
    assert targets["objective_estimate"].tolist() == pytest.approx([6.0, 5.0, 3.0])


def test_normalizer_checkpoint_roundtrip():
    norm = FeatureNormalizer(4)
    norm.mean.copy_(torch.tensor([1.0, 2.0, 3.0, 4.0]))
    norm.std.copy_(torch.tensor([0.5, 0.5, 0.5, 0.5]))
    state = norm.state_dict_for_checkpoint()
    norm2 = FeatureNormalizer(4)
    norm2.load_from_checkpoint(state)
    x = torch.tensor([[2.0, 3.0, 4.0, 5.0]])
    assert torch.allclose(norm(x), norm2(x))
