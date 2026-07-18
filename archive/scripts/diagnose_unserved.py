"""Diagnose why customers remain unserved at episode end."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np

from src.data.instance_loader import load_instance
from src.env.evrptw_env import EVRPTWEnv
from src.env.masks import customer_feasible
from src.env.policy_action import apply_policy_action, resolve_depot_action
from src.evaluation.baselines import NearestFeasiblePolicy, run_heuristic_episode


def diagnose_instance(instance_dir: Path, m: int = 20) -> dict:
    inst = load_instance(instance_dir)
    env = EVRPTWEnv(inst, m=m, use_horizon_check=True, max_steps_factor=6)
    run_heuristic_episode(env, NearestFeasiblePolicy())

    unserved = [c for c in inst.customer_indices if not env.served_mask[c]]
    reason_counts: Counter = Counter()
    per_unserved: list[tuple[int, str]] = []

    for c in unserved:
        best_reason = "never_feasible_any_ev"
        for ev in env.evs:
            ok, reason = customer_feasible(
                inst, ev, c, env.served_mask, env.reserved_mask, use_horizon_check=True
            )
            if ok:
                best_reason = "still_feasible_but_not_chosen"
                break
            reason_counts[reason] += 1
        else:
            ev_reasons = []
            for ev in env.evs:
                _, reason = customer_feasible(
                    inst, ev, c, env.served_mask, env.reserved_mask, use_horizon_check=True
                )
                ev_reasons.append(reason)
            best_reason = Counter(ev_reasons).most_common(1)[0][0]
        per_unserved.append((c, best_reason))
        reason_counts[best_reason] += 1

    from src.env.ev_state import EVState

    fresh = EVState.initial(0, inst.depot_index, inst.vehicle.battery_capacity, inst.vehicle.load_capacity)
    fresh_reasons: Counter = Counter()
    for c in unserved:
        ok, reason = customer_feasible(
            inst, fresh, c, env.served_mask, env.reserved_mask, use_horizon_check=True
        )
        if not ok:
            fresh_reasons[reason] += 1

    return {
        "instance": inst.instance_name,
        "m": m,
        "num_customers": inst.num_customers,
        "served": env.num_served_customers,
        "unserved": len(unserved),
        "done_reason": env.get_done_reason(),
        "step_count": env.step_count,
        "max_steps": env.max_steps,
        "finished_evs": sum(1 for ev in env.evs if ev.finished),
        "unserved_reasons": dict(reason_counts),
        "fresh_depot_reasons_for_unserved": dict(fresh_reasons),
        "avg_lateness": env.total_lateness,
        "battery_viol": env.total_battery_violation,
        "capacity_viol": env.total_capacity_violation,
        "depot_viol": env.depot_return_violation,
    }


if __name__ == "__main__":
    base = Path("dataset/Schneider/evrptw_instances")
    train = [
        "rc102_21", "rc103_21", "rc105_21", "rc106_21", "rc107_21",
        "rc201_21", "rc202_21", "rc203_21", "rc204_21", "rc205_21",
        "rc206_21", "rc207_21",
    ]
    print("=== Greedy baseline diagnosis (m=20) ===")
    totals: Counter = Counter()
    for name in train:
        r = diagnose_instance(base / name)
        print(
            f"{r['instance']}: served={r['served']}/{r['num_customers']} "
            f"done={r['done_reason']} steps={r['step_count']}/{r['max_steps']} "
            f"finished_evs={r['finished_evs']}"
        )
        print(f"  unserved reasons: {r['unserved_reasons']}")
        print(f"  from fresh depot: {r['fresh_depot_reasons_for_unserved']}")
        for k, v in r["unserved_reasons"].items():
            totals[k] += v

    print("\n=== Aggregate unserved reasons (greedy) ===")
    for k, v in totals.most_common():
        print(f"  {k}: {v}")
