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
        mask, _ = env.feasible_mask(ev)
        if not mask.any():
            env.finish_idle_ev(ev)
            continue
        node, portion = policy.select(env, ev, mask)
        env.apply_action(ev, node, portion)
    return env


class NearestFeasiblePolicy:
    """Pick the nearest feasible unserved customer; else nearest station; else depot."""

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
            return node, 1.0  # charge fully

        # Fall back to depot (the only remaining feasible option).
        return inst.depot_index, None


class RandomFeasiblePolicy:
    """Pick uniformly at random among feasible nodes."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def select(self, env: EVRPTWEnv, ev, mask: np.ndarray):
        inst = env.instance
        feasible = np.nonzero(mask)[0].tolist()
        node = self.rng.choice(feasible)
        portion = self.rng.choice([0.25, 0.5, 0.75, 1.0]) if inst.nodes[node].is_charging_station else None
        return node, portion


def evaluate_baseline(instance, m: int, policy, weights: CostWeights, **env_kwargs):
    """Run a baseline on one instance with fleet size m; return (J, summary, env)."""
    env = EVRPTWEnv(instance, m=m, **env_kwargs)
    run_heuristic_episode(env, policy)
    j, _, summary = env.score(weights)
    return j, summary, env
