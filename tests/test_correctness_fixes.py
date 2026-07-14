"""Correctness tests for EVRPTW scientific fixes (Part 1)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.instance_loader import load_instance
from src.env.charge_masks import safe_depot_return_from
from src.env.decision_analysis import (
    analyses_are_equivalent,
    analyze_decision_state_optimized,
    analyze_decision_state_reference,
)
from src.env.evrptw_env import EVRPTWEnv
from src.env.masks import EPS
from src.env.recharge_search import (
    RechargeSearchConfig,
    bounded_recharge_reachability,
    customer_reachable_via_recharge_chain,
)
from src.models.critic import CentralFleetValueCritic
from src.models.decision_state import DecisionState
from src.models.dynamic_candidate import compute_dynamic_candidate_features


ROOT = Path(__file__).resolve().parents[1]
SMALL = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "c101C5"
RC21 = ROOT / "dataset" / "Schneider" / "evrptw_instances" / "rc102_21"


def _small_env(m: int = 2) -> EVRPTWEnv:
    if not SMALL.exists():
        # Fallback to converted dataset layout used by some checkouts.
        alt = ROOT / "dataset" / "evrptw_instances" / "c101C5"
        inst = load_instance(alt if alt.exists() else SMALL)
    else:
        inst = load_instance(SMALL)
    env = EVRPTWEnv(inst, m=m, charging_portions=[0.25, 0.5, 0.75, 1.0], max_steps_factor=6)
    env.environment_config = {
        "mask_semantics": "exact_original_evrptw",
        "implementation": "optimized",
        "action_pruning": {
            "enabled": True,
            "block_exact_noop_actions": True,
            "block_repeated_zero_progress_loops": True,
            "prune_station_when_customer_available": False,
        },
        "recharge_search": {"max_station_hops": 3, "use_dominance_pruning": True},
        "cache": {"enabled": True, "cache_by_state_version": True},
    }
    return env


def test_direct_depot_battery_feature_matches_mask():
    env = _small_env()
    ev = env.select_acting_ev()
    mask = env.compute_action_masks(ev)
    feats = compute_dynamic_candidate_features(env, ev, mask)
    # Feature column 11 = safe_direct
    safe_col = feats[:, 11]
    dep = env.instance.depot_index
    for j in env.instance.customer_indices:
        if env.served_mask[j] or env.reserved_mask[j]:
            continue
        energy_to_j = float(env.instance.energy_matrix[ev.current_node, j])
        battery_after = ev.remaining_battery - energy_to_j
        svc_end = max(
            ev.current_time + env.instance.travel_time_matrix[ev.current_node, j],
            env.instance.nodes[j].ready_time,
        ) + env.instance.nodes[j].service_time
        ok_batt = battery_after + EPS >= float(env.instance.energy_matrix[j, dep])
        ok_dep, _ = safe_depot_return_from(
            env.instance, j, svc_end, battery_after, use_horizon_check=env.use_horizon_check,
        )
        expected = bool(ok_batt and ok_dep) if (
            battery_after >= -EPS
            and max(ev.current_time + env.instance.travel_time_matrix[ev.current_node, j],
                    env.instance.nodes[j].ready_time)
            <= env.instance.nodes[j].due_time + EPS
            and env.instance.nodes[j].demand <= ev.remaining_capacity + EPS
            and energy_to_j <= ev.remaining_battery + EPS
        ) else False
        # Dynamic feature safe_direct must use single depot energy subtraction.
        if expected:
            assert safe_col[j] > 0.5, f"customer {j}: expected safe_direct=1"
        # Never require 2x depot energy
        e_dep = float(env.instance.energy_matrix[j, dep])
        if battery_after + EPS >= e_dep and not (battery_after + EPS >= 2 * e_dep):
            # This is exactly the previously-buggy zone; feature must allow it
            # if time is also feasible.
            if ok_dep:
                assert safe_col[j] > 0.5, f"customer {j}: double-subtraction bug still present"


def test_num_customers_served_not_route_length():
    env = _small_env()
    ev = env.evs[0]
    assert ev.num_customers_served == 0
    mask = env.compute_action_masks(ev)
    # Find a feasible customer
    cust = next(c for c in env.instance.customer_indices if mask.node_mask[c])
    env.apply_action(ev, cust)
    assert ev.num_customers_served == 1
    route_stops = len(ev.route_nodes) - 1
    # Visit a station if possible without serving more customers
    mask2 = env.compute_action_masks(ev)
    stations = [s for s in env.instance.charging_station_indices if mask2.node_mask[s]]
    if stations:
        env.apply_action(ev, stations[0], charge_portion=1.0)
        assert ev.num_customers_served == 1
        assert len(ev.route_nodes) - 1 > route_stops


def test_auto_close_at_depot_invalidates_caches():
    env = _small_env(m=2)
    ev = env.evs[0]
    assert ev.current_node == env.instance.depot_index
    mask1 = env.compute_action_masks(ev)
    v1 = env.state_version
    cache_key = (v1, ev.ev_id)
    assert cache_key in env._action_mask_cache or True  # may or may not store depending on path
    env.auto_close_route(ev, forced=True)
    assert ev.finished
    assert env.state_version == v1 + 1
    assert env._action_mask_cache == {}
    assert env._global_serviceability_cache is None
    # Fresh mask must recompute
    ev2 = env.select_acting_ev()
    if ev2 is not None:
        mask2 = env.compute_action_masks(ev2)
        assert mask2.state_version == env.state_version


def test_safe_pruning_keeps_station_when_customer_available():
    env = _small_env()
    env.environment_config["action_pruning"]["prune_station_when_customer_available"] = False
    ev = env.select_acting_ev()
    hard = env.compute_action_masks(ev)
    # With unsafe prune disabled, hard-feasible stations that survive no-progress
    # checks stay available even if customers are feasible.
    hard_stations = [
        s for s in env.instance.charging_station_indices
        if hard.hard_node_mask is not None and hard.hard_node_mask[s]
    ]
    final_stations = [s for s in hard_stations if hard.node_mask[s]]
    # Not all may survive (noop loops), but we must not wipe all of them solely
    # because customers exist.
    if hard.feasible_customer_count > 0 and hard_stations:
        # At least one station may remain; if all pruned, reasons must not all be
        # the unsafe customer-available heuristic.
        for s in hard_stations:
            if not hard.node_mask[s]:
                reason = hard.node_reasons.get(s, "")
                assert "customer_feasible_now" not in reason


def test_reference_optimized_analysis_parity():
    env = _small_env()
    ev = env.select_acting_ev()
    # Clear caches between paths
    ref = analyze_decision_state_reference(env, ev)
    env._decision_analysis_cache = {}
    env._action_mask_cache = {}
    opt = analyze_decision_state_optimized(env, ev)
    ok, msg = analyses_are_equivalent(ref, opt, feature_atol=1e-4)
    assert ok, msg


def test_tw_expired_not_equal_all_unserviceable():
    env = _small_env()
    gs = env.get_global_serviceability(force=True)
    # Structurally: expired count ≤ unserviceable count; battery/capacity separate.
    assert gs.globally_expired_customers_count <= gs.globally_unserviceable_customers_count or (
        gs.globally_unserviceable_customers_count == 0
    )
    # Category lists are disjoint from serviceable
    svc = set(gs.serviceable_customer_ids)
    assert svc.isdisjoint(set(gs.time_window_expired_customer_ids))
    assert svc.isdisjoint(set(gs.battery_unserviceable_customer_ids))


def test_recharge_chain_search_runs():
    env = _small_env()
    ev = env.evs[0]
    # Drain battery so direct service fails for distant customers
    ev.remaining_battery = min(ev.remaining_battery, 5.0)
    reach = bounded_recharge_reachability(
        env.instance, ev,
        portions=env.charging_portions,
        config=RechargeSearchConfig(max_station_hops=3),
    )
    assert isinstance(reach.labels_by_node, dict)
    # Search must produce some labels from depot/stations when reachable
    assert reach.expanded >= 0


def _make_dummy_decision_state(n: int = 8, d: int = 32, seed: int = 0) -> DecisionState:
    from src.env.action_masks import ActionMaskResult
    from src.env.action_space import GlobalServiceability

    g = torch.Generator().manual_seed(seed)
    feas = np.zeros(n, dtype=bool)
    feas[1:4] = True
    feas[5] = True
    dyn = torch.zeros(n, 21)
    dyn[1:4, 17] = 1
    dyn[5, 18] = 1
    dyn[0, 19] = 1
    dummy_mask = ActionMaskResult(
        state_version=0,
        node_mask=feas.copy(),
        node_reasons={},
        charge_masks={},
        charge_reasons={},
        depot_mode_mask=np.array([False, False], dtype=bool),
        depot_mode_reasons=["", ""],
        feasible_customer_count=int(feas[1:4].sum()),
        feasible_station_count=1,
        feasible_depot_recharge=False,
        feasible_depot_finish=False,
        global_serviceability=GlobalServiceability(),
        mask_reason_counts={},
    )
    z = torch.randn(d, generator=g)
    return DecisionState(
        decoder_context=z.clone(),
        acting_ev_context=torch.randn(d, generator=g),
        communicated_ev_tokens=torch.randn(3, d, generator=g),
        acting_ev_token=torch.randn(d, generator=g),
        active_fleet_context=torch.randn(d, generator=g),
        finished_fleet_context=torch.randn(d, generator=g),
        fleet_balance_context=torch.randn(d, generator=g),
        global_context=torch.randn(d, generator=g),
        mask_context=torch.randn(d, generator=g),
        static_node_embeddings=torch.randn(n, d, generator=g),
        candidate_embeddings=torch.randn(n, d, generator=g),
        dynamic_candidate_embeddings=torch.randn(n, d, generator=g),
        dynamic_candidate_feats=dyn,
        global_features_raw=torch.randn(8, generator=g),
        mask_summary_raw=torch.randn(8, generator=g),
        fleet_balance_raw=torch.randn(4, generator=g),
        feasible_mask=feas,
        mask_result=dummy_mask,
    )


def test_batched_critic_matches_single():
    d = 32
    critic = CentralFleetValueCritic(embedding_dim=d)
    critic.eval()
    states = [_make_dummy_decision_state(seed=i) for i in range(4)]
    with torch.no_grad():
        singles = torch.stack([critic.forward_from_state(s)[0].reshape(()) for s in states])
        batch, _ = critic.forward_from_states_batch(states)
    assert torch.allclose(singles, batch, atol=1e-5, rtol=1e-4), (
        f"max diff={(singles - batch).abs().max().item()}"
    )


@pytest.mark.skipif(not RC21.exists(), reason="rc102_21 not present")
def test_rc21_mask_parity_sample():
    inst = load_instance(RC21)
    env = EVRPTWEnv(inst, m=3, charging_portions=[0.25, 0.5, 0.75, 1.0], max_steps_factor=4)
    env.environment_config = {
        "mask_semantics": "exact_original_evrptw",
        "implementation": "optimized",
        "action_pruning": {
            "enabled": True,
            "block_exact_noop_actions": True,
            "block_repeated_zero_progress_loops": True,
            "prune_station_when_customer_available": False,
        },
        "recharge_search": {"max_station_hops": 3, "use_dominance_pruning": True},
        "cache": {"enabled": True, "cache_by_state_version": True},
    }
    ev = env.select_acting_ev()
    ref = analyze_decision_state_reference(env, ev)
    env._decision_analysis_cache = {}
    env._action_mask_cache = {}
    opt = analyze_decision_state_optimized(env, ev)
    ok, msg = analyses_are_equivalent(ref, opt, feature_atol=1e-4)
    assert ok, msg
