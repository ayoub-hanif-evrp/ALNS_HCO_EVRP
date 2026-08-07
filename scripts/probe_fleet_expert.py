"""Quick expert probe: capacity fleet vs TW-aware fleet."""
from __future__ import annotations

import time

from src.data.evrptw_dataset import split_dataset
from src.data.evrptwd_dataset import ensure_evrptwd_dataset
from src.data.instance_loader import load_capacity_from_config
from src.env.cost import CostWeights
from src.expert.feasibility_first_expert import FeasibilityFirstExpert
from src.expert.generate_demonstrations import run_expert_episode
from src.utils.config import load_config


def main() -> None:
    cfg = load_config("configs/train.yaml")
    ensure_evrptwd_dataset(cfg)
    ds_cfg = cfg["dataset"]
    train_ds, _ = split_dataset(
        ds_cfg["path"],
        train_ratio=0.8,
        seed=42,
        train_filter=[],
        eval_filter=[],
        train_instances=ds_cfg["train_instances"][:1],
        eval_instances=[],
        load_capacity=load_capacity_from_config(cfg),
    )
    inst = train_ds[0]
    weights = CostWeights.from_config(cfg["cost"])
    expert = FeasibilityFirstExpert(cfg)
    portions = list(cfg["problem"]["charging_portions"])
    env_cfg = cfg.get("environment") or {}
    print("instance", inst.instance_name, "m_cap", inst.m_min)
    for m in (2, 13):
        t0 = time.time()
        ep = run_expert_episode(
            inst,
            m,
            expert,
            weights,
            charging_portions=portions,
            use_horizon_check=True,
            use_depot_charging=True,
            max_steps_factor=6,
            environment_config=env_cfg,
            reward_scale=10000.0,
            collect_transitions=False,
        )
        print(
            f"m={m} served={ep.metrics['served_ratio']:.3f} "
            f"feas={ep.metrics['feasible']} "
            f"unserved={ep.metrics['unserved_customers']} "
            f"decisions={ep.metrics['num_decisions']} "
            f"t={time.time() - t0:.1f}s"
        )


if __name__ == "__main__":
    main()
