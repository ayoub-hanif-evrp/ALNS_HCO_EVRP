"""Heuristic baselines for E-VRPTW.

These use the exact same environment and feasibility masks as the learned
policy, so comparisons are apples-to-apples. They also serve as an end-to-end
sanity check that the environment can produce feasible solutions.
"""
from __future__ import annotations

import random

import numpy as np

from src.env.cost import CostWeights
from src.env.evrptw_env import EVRPTWEnv


def run_heuristic_episode(env: EVRPTWEnv, policy) -> EVRPTWEnv:
    """Drive the event-driven loop with a heuristic node-selection policy."""
    env.reset()
    while not env.is_done():
        ev = env.select_acting_ev()
        if ev is None:
            break
        if env.try_auto_close_route(ev):
            continue
        mask, _ = env.feasible_mask(ev)
        node, portion = policy.select(env, ev, mask)
        env.apply_action(ev, node, portion)
    for ev in env.evs:
        if not ev.finished:
            env.auto_close_route(ev)
    return env


class NearestFeasiblePolicy:
    """Pick the nearest feasible unserved customer; else nearest recharge node."""

    def select(self, env: EVRPTWEnv, ev, mask: np.ndarray):
        inst = env.instance
        dist = inst.distance_matrix[ev.current_node]

        feasible_customers = [c for c in inst.customer_indices if mask[c]]
        if feasible_customers:
            node = min(feasible_customers, key=lambda c: dist[c])
            return node, None

        feasible_stations = [s for s in inst.charging_station_indices if mask[s]]
        if feasible_stations:
            node = min(feasible_stations, key=lambda s: dist[s])
            return node, 1.0

        if mask[inst.depot_index]:
            return inst.depot_index, 1.0

        raise RuntimeError("No feasible action for heuristic policy")


class RandomFeasiblePolicy:
    """Pick uniformly at random among feasible nodes."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, env: EVRPTWEnv, ev, mask: np.ndarray):
        inst = env.instance
        feasible = np.nonzero(mask)[0].tolist()
        node = self.rng.choice(feasible)
        n_obj = inst.nodes[node]
        is_recharge = n_obj.is_charging_station or (
            n_obj.is_depot and env.use_depot_charging and inst.depot_has_charging
        )
        portion = self.rng.choice([0.25, 0.5, 0.75, 1.0]) if is_recharge else None
        return node, portion


def evaluate_baseline(instance, m: int, policy, weights: CostWeights, **env_kwargs):
    """Run a baseline on one instance with fleet size m; return (J, summary, env)."""
    env = EVRPTWEnv(instance, m=m, **env_kwargs)
    run_heuristic_episode(env, policy)
    j, _, summary = env.score(weights)
    return j, summary, env
