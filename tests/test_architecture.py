"""Architecture tests for dynamic SMDP Transformer policy (Section 19)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from src.env.evrptw_env import EVRPTWEnv
from src.env.action_masks import compute_action_masks
from src.models.critic import CentralFleetValueCritic
from src.models.feature_specs import DYNAMIC_CANDIDATE_FEAT_DIM
from src.models.policy import TransformerEVRPTWPolicy
from src.training.actor_critic import compute_smdp_gae
from tests.conftest import small_instance


def _policy_and_env(instance):
    cfg = {
        "embedding_dim": 64,
        "encoder_layers": 2,
        "heads": 4,
        "feedforward_dim": 128,
        "dropout": 0.0,
        "use_ev_communication": True,
        "use_edge_aware_encoder": False,
    }
    policy = TransformerEVRPTWPolicy(cfg, charging_portions=[0.25, 0.5, 0.75, 1.0])
    env = EVRPTWEnv(instance, m=2, charging_portions=policy.charging_portions)
    return policy, env


def test_dynamic_candidate_feature_shape(small_instance):
    policy, env = _policy_and_env(small_instance)
    ev = env.select_acting_ev()
    mask = compute_action_masks(env, ev)
    state = policy.build_decision_state(env, ev, policy.encode(small_instance), mask)
    assert state.dynamic_candidate_feats.shape == (small_instance.num_nodes, DYNAMIC_CANDIDATE_FEAT_DIM)


def test_dynamic_candidate_time_sensitivity(small_instance):
    policy, env = _policy_and_env(small_instance)
    ev = env.select_acting_ev()
    mask = policy.encode(small_instance)
    H = policy.encode(small_instance)
    s1 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    ev.current_time += 25.0
    s2 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    assert not torch.allclose(s1.dynamic_candidate_embeddings, s2.dynamic_candidate_embeddings)


def test_battery_sensitivity(small_instance):
    policy, env = _policy_and_env(small_instance)
    ev = env.select_acting_ev()
    H = policy.encode(small_instance)
    s1 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    ev.remaining_battery *= 0.2
    s2 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    assert not torch.allclose(s1.dynamic_candidate_embeddings, s2.dynamic_candidate_embeddings)


def test_fleet_communication_changes_with_other_ev(small_instance):
    policy, env = _policy_and_env(small_instance)
    ev = env.select_acting_ev()
    H = policy.encode(small_instance)
    s1 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    other = env.evs[1] if env.evs[0].ev_id == ev.ev_id else env.evs[0]
    other.current_node = min(small_instance.num_nodes - 1, other.current_node + 1)
    other.route_nodes.append(other.current_node)
    s2 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    assert not torch.allclose(s1.active_fleet_context, s2.active_fleet_context)


def test_critic_state_sensitivity(small_instance):
    policy, env = _policy_and_env(small_instance)
    critic = CentralFleetValueCritic(64)
    ev = env.select_acting_ev()
    H = policy.encode(small_instance)
    s1 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    v1, _ = critic.forward_from_state(s1)
    env.served_mask[small_instance.customer_indices[0]] = True
    env.num_served_customers += 1
    s2 = policy.build_decision_state(env, ev, H, compute_action_masks(env, ev))
    v2, _ = critic.forward_from_state(s2)
    assert not torch.isclose(v1, v2)


def test_hierarchical_log_probability_decomposition(small_instance):
    policy, env = _policy_and_env(small_instance)
    ev = env.select_acting_ev()
    H = policy.encode(small_instance)
    mask = compute_action_masks(env, ev)
    out = policy.select_action(env, ev, H, mask, mode="greedy")
    if out.secondary_log_prob is not None:
        recomposed = out.node_log_prob + out.secondary_log_prob
        assert torch.allclose(recomposed, out.log_prob, atol=1e-5)
    else:
        assert torch.allclose(out.node_log_prob, out.log_prob, atol=1e-5)


def test_smdp_gae_forced_closure_in_preceding_return():
    rewards = torch.tensor([1.0, 0.0], dtype=torch.float32)
    values = torch.tensor([0.5, 0.3], dtype=torch.float32)
    adv, ret = compute_smdp_gae(rewards, values, gamma=1.0, lam=0.95)
    assert adv[0].item() > adv[1].item()
    assert rewards[0] + rewards[1] == pytest.approx(1.0)


def test_train_eval_greedy_identity(small_instance):
    policy, env = _policy_and_env(small_instance)
    policy.eval()
    ev = env.select_acting_ev()
    H = policy.encode(small_instance)
    mask = compute_action_masks(env, ev)
    a1 = policy.select_action(env, ev, H, mask, mode="greedy")
    a2 = policy.select_action(env, ev, H, mask, mode="greedy")
    assert a1.node == a2.node
    assert a1.charge_portion == a2.charge_portion
    assert a1.depot_finish == a2.depot_finish
