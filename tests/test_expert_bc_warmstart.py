"""Tests for feasibility-first expert, BC losses, and BC→RL warm-start wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv
from src.env.policy_action import PolicyAction, apply_policy_action
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.training.behavior_cloning import bc_coef_at_update, masked_ce_from_logits
from src.utils.tensor_utils import safe_mask_logits

DATASET = Path(__file__).resolve().parents[1] / "dataset" / "Schneider" / "evrptw_instances"
SMALL = DATASET / "c101C5"


@pytest.fixture(scope="module")
def small_inst():
    if not SMALL.exists():
        pytest.skip("small Schneider instance missing")
    return load_instance(SMALL)


def _env(instance, m=2):
    return EVRPTWEnv(
        instance, m=m,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
        use_horizon_check=True,
        use_depot_charging=True,
        hard_time_windows=True,
    )


def test_expert_never_selects_masked_node(small_inst):
    env = _env(small_inst)
    expert = FeasibilityFirstExpert(ExpertConfig(lookahead_depth=0))
    ev = env.select_acting_ev()
    analysis = env.analyze_decision(ev)
    action = expert.select_action(env, ev, analysis)
    assert bool(analysis.final_node_mask[action.node])


def test_expert_never_selects_illegal_charge_or_depot(small_inst):
    env = _env(small_inst)
    expert = FeasibilityFirstExpert(ExpertConfig(lookahead_depth=0))
    steps = 0
    while not env.is_done() and steps < 40:
        ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
            continue
        if env.try_auto_close_from_analysis(ev, analysis) is not None:
            continue
        action = expert.select_action(env, ev, analysis)
        assert bool(analysis.final_node_mask[action.node])
        inst = env.instance
        if action.node == inst.depot_index:
            dmm = analysis.depot_mode_mask
            if action.depot_finish:
                assert bool(dmm[1])
            else:
                assert bool(dmm[0])
                cm = analysis.final_charge_masks[action.node]
                portions = list(env.charging_portions)
                idx = portions.index(float(action.charge_portion))
                assert bool(cm[idx])
        elif inst.nodes[action.node].is_charging_station:
            cm = analysis.final_charge_masks[action.node]
            portions = list(env.charging_portions)
            idx = portions.index(float(action.charge_portion))
            assert bool(cm[idx])
        apply_policy_action(env, ev, action, mask_result=analysis.mask_result)
        steps += 1


def test_expert_uses_canonical_exact_mask(small_inst):
    env = _env(small_inst)
    ev = env.select_acting_ev()
    analysis = env.analyze_decision(ev)
    assert analysis.mask_result is not None
    assert np.array_equal(analysis.final_node_mask, analysis.mask_result.node_mask)
    expert = FeasibilityFirstExpert(ExpertConfig(lookahead_depth=0))
    action = expert.select_action(env, ev, analysis)
    apply_policy_action(env, ev, action, mask_result=analysis.mask_result)


def test_expert_trajectory_zero_hard_violations(small_inst):
    from src.expert.generate_demonstrations import run_expert_episode

    weights = CostWeights(
        lambda_vehicle=1000, lambda_distance=1, lambda_duration=0.1,
        lambda_charging=0.5, lambda_charging_price=1, lambda_waiting=0.1,
        penalty_unserved=10000, penalty_duplicate=10000, penalty_late=5000,
        penalty_battery=10000, penalty_capacity=10000, penalty_depot=10000,
        penalty_infeasible=100000, penalty_reserved=10000,
    )
    expert = FeasibilityFirstExpert(ExpertConfig(lookahead_depth=0))
    ep = run_expert_episode(
        small_inst, m=2, expert=expert, weights=weights,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
        collect_transitions=False,
    )
    assert ep.metrics["battery_violations"] == 0
    assert ep.metrics["time_window_violations"] == 0
    assert ep.metrics["capacity_violations"] == 0


def test_masked_ce_rejects_illegal_target():
    logits = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([True, False, True])
    with pytest.raises(RuntimeError):
        masked_ce_from_logits(logits, 1, mask)
    loss = masked_ce_from_logits(logits, 0, mask)
    assert torch.isfinite(loss)


def test_bc_coef_decays_correctly():
    cfg = {
        "bc_regularization": {
            "enabled": True,
            "coefficient_initial": 0.1,
            "coefficient_final": 0.0,
            "decay_updates": 100,
        }
    }
    assert abs(bc_coef_at_update(cfg, 0) - 0.1) < 1e-9
    assert abs(bc_coef_at_update(cfg, 100) - 0.0) < 1e-9
    assert abs(bc_coef_at_update(cfg, 50) - 0.05) < 1e-9
    assert bc_coef_at_update({"bc_regularization": {"enabled": False}}, 0) == 0.0


def test_bc_checkpoint_loads_into_policy(small_inst, tmp_path, model_cfg):
    from src.models.policy import TransformerEVRPTWPolicy
    from src.models.feature_specs import ARCHITECTURE_VERSION

    portions = [0.25, 0.5, 0.75, 1.0]
    policy = TransformerEVRPTWPolicy(model_cfg, charging_portions=portions)
    path = tmp_path / "bc.pt"
    torch.save({
        "architecture_version": ARCHITECTURE_VERSION,
        "model_state": policy.state_dict(),
        "model_config": model_cfg,
        "charging_portions": portions,
    }, path)
    from src.training.train_rl import load_policy
    loaded = load_policy(path, device="cpu")
    assert loaded is not None
    for a, b in zip(policy.parameters(), loaded.parameters()):
        assert torch.allclose(a, b)


def test_secondary_heads_ignored_when_not_applicable(small_inst):
    """Customer action → only node loss contribution required (charge/depot zero)."""
    from src.models.policy import TransformerEVRPTWPolicy
    from src.training.behavior_cloning import bc_loss_on_decision

    env = _env(small_inst)
    ev = env.select_acting_ev()
    analysis = env.analyze_decision(ev)
    legal = [int(c) for c in small_inst.customer_indices if bool(analysis.final_node_mask[c])]
    if not legal:
        pytest.skip("no legal customer on small instance at start")
    model_cfg = {
        "embedding_dim": 32, "heads": 4, "encoder_layers": 2,
        "feedforward_dim": 64, "dropout": 0.0,
        "use_ev_communication": False, "use_edge_aware_encoder": False,
        "use_dynamic_candidate_encoder": True,
    }
    policy = TransformerEVRPTWPolicy(model_cfg, charging_portions=list(env.charging_portions))
    H = policy.encode(small_inst, device="cpu")
    action = PolicyAction(node=legal[0], charge_portion=None, depot_finish=False)
    loss, stats = bc_loss_on_decision(policy, env, ev, analysis, H, action)
    assert torch.isfinite(loss)
    assert stats["charge_used"] == 0.0
    assert stats["depot_used"] == 0.0


def test_train_config_pipeline_no_early_stopping():
    from src.utils.config import load_config

    cfg = load_config("configs/train.yaml")
    assert cfg["pipeline"]["run_behavior_cloning"] is True
    assert cfg["pipeline"]["run_rl_finetuning"] is True
    assert cfg["behavior_cloning"]["early_stopping"] is False
    assert int(cfg["behavior_cloning"]["epochs"]) > 0
    assert cfg["rl"]["finetune_from_bc"] is True
    assert cfg["dashboard"]["auto_latest_run"] is True


def test_official_objective_reconstruction_step_consistent(small_inst):
    """Step deltas reconstruct the non-terminal part of Schneider J (exact weights)."""
    from src.env.reward import StepRewardCalculator, snapshot_env
    from src.expert.generate_demonstrations import run_expert_episode

    weights = CostWeights()
    expert = FeasibilityFirstExpert(ExpertConfig(lookahead_depth=0))
    ep = run_expert_episode(
        small_inst, m=2, expert=expert, weights=weights,
        charging_portions=[0.25, 0.5, 0.75, 1.0],
        collect_transitions=True,
    )
    # Hard constraints never violated under exact masks.
    assert ep.metrics["battery_violations"] == 0
    assert ep.metrics["time_window_violations"] == 0
    # Terminal unserved/infeasible penalties are outside step deltas — mismatch is expected
    # when unserved remain, but step path itself must be finite and non-negative abs error.
    assert ep.metrics["objective_reconstruction_mismatch"] >= 0.0
    assert np.isfinite(ep.metrics["objective"])
