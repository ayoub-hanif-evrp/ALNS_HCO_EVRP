"""Heuristic baselines using canonical exact masks (same as learned policy)."""
from __future__ import annotations

import random

import numpy as np

from src.schneider.env.action_masks import ActionMaskResult
from src.schneider.env.cost import CostWeights
from src.schneider.env.evrptw_env import EVRPTWEnv
from src.schneider.env.policy_action import PolicyAction, apply_policy_action


def _legal_charge_portion(
    env: EVRPTWEnv,
    mask_result: ActionMaskResult,
    node: int,
    preferred: float | None = None,
) -> float | None:
    cm = mask_result.charge_masks.get(node)
    if cm is None:
        return None
    portions = list(env.charging_portions)
    feasible_idx = [i for i, ok in enumerate(cm) if ok]
    if not feasible_idx:
        raise RuntimeError(f"No legal charge portion for node {node}")
    if preferred is not None:
        for idx in feasible_idx:
            if abs(portions[idx] - float(preferred)) < 1e-9:
                return float(portions[idx])
        try:
            pidx = portions.index(float(preferred))
            if cm[pidx]:
                return float(preferred)
        except ValueError:
            pass
    return float(portions[max(feasible_idx)])


def run_heuristic_episode(env: EVRPTWEnv, policy) -> EVRPTWEnv:
    """Drive the event-driven loop with a heuristic policy and canonical masks."""
    env.reset()
    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            break
        analysis = env.analyze_decision(ev)
        if env.try_controlled_closure_from_analysis(ev, analysis) is not None:
            continue
        if env.try_auto_close_from_analysis(ev, analysis) is not None:
            continue
        mask_result = analysis.mask_result
        action = policy.select(env, ev, mask_result)
        apply_policy_action(env, ev, action, mask_result=mask_result)
    for ev in env.evs:
        if not ev.finished:
            env.auto_close_route(ev, forced=True)
    return env


class NearestFeasiblePolicy:
    """Pick nearest feasible unserved customer; else nearest feasible recharge node."""

    def select(self, env: EVRPTWEnv, ev, mask_result: ActionMaskResult) -> PolicyAction:
        inst = env.instance
        mask = mask_result.final_node_mask
        dist = inst.distance_matrix[ev.current_node]

        feasible_customers = [c for c in inst.customer_indices if mask[c]]
        if feasible_customers:
            node = min(feasible_customers, key=lambda c: dist[c])
            return PolicyAction(node=node, charge_portion=None, depot_finish=False)

        feasible_stations = [s for s in inst.charging_station_indices if mask[s]]
        if feasible_stations:
            node = min(feasible_stations, key=lambda s: dist[s])
            portion = _legal_charge_portion(env, mask_result, node)
            return PolicyAction(node=node, charge_portion=portion, depot_finish=False)

        depot = inst.depot_index
        if mask[depot]:
            if mask_result.feasible_depot_recharge:
                portion = _legal_charge_portion(env, mask_result, depot)
                return PolicyAction(node=depot, charge_portion=portion, depot_finish=False)
            if mask_result.feasible_depot_finish:
                return PolicyAction(node=depot, charge_portion=None, depot_finish=True)

        raise RuntimeError("No feasible action for heuristic policy")


class RandomFeasiblePolicy:
    """Pick uniformly at random among final-mask feasible nodes and legal charge portions."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, env: EVRPTWEnv, ev, mask_result: ActionMaskResult) -> PolicyAction:
        inst = env.instance
        mask = mask_result.final_node_mask
        feasible = np.nonzero(mask)[0].tolist()
        node = self.rng.choice(feasible)
        n_obj = inst.nodes[node]

        if node == inst.depot_index:
            modes: list[tuple[bool, float | None]] = []
            if mask_result.feasible_depot_recharge and node in mask_result.charge_masks:
                cm = mask_result.charge_masks[node]
                portions = env.charging_portions
                for idx, p in enumerate(portions):
                    if cm[idx]:
                        modes.append((False, float(p)))
            if mask_result.feasible_depot_finish:
                modes.append((True, None))
            if not modes:
                raise RuntimeError("Depot selected but no legal depot mode")
            depot_finish, portion = self.rng.choice(modes)
            return PolicyAction(node=node, charge_portion=portion, depot_finish=depot_finish)

        is_recharge = n_obj.is_charging_station
        if is_recharge:
            cm = mask_result.charge_masks.get(node)
            if cm is None or not cm.any():
                raise RuntimeError(f"Station {node} has no legal charge mask")
            legal = [env.charging_portions[i] for i, ok in enumerate(cm) if ok]
            portion = float(self.rng.choice(legal))
            return PolicyAction(node=node, charge_portion=portion, depot_finish=False)

        return PolicyAction(node=node, charge_portion=None, depot_finish=False)


def evaluate_baseline(instance, m: int, policy, weights: CostWeights, **env_kwargs):
    """Run a baseline on one instance with fleet size m; return (J, summary, env)."""
    env = EVRPTWEnv(instance, m=m, **env_kwargs)
    run_heuristic_episode(env, policy)
    j, _, summary = env.score(weights)
    return j, summary, env


__all__ = [
    "NearestFeasiblePolicy",
    "PolicyAction",
    "RandomFeasiblePolicy",
    "evaluate_baseline",
    "run_heuristic_episode",
]
