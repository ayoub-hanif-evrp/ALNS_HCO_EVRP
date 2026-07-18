"""Quick training-episode speed benchmark."""
from __future__ import annotations

import time

from src.data.evrptw_dataset import split_dataset
from src.data.instance_loader import load_capacity_from_config
from src.data.rc21_split import resolve_rc21_split
from src.env.cost import CostWeights
from src.models.policy import TransformerEVRPTWPolicy
from src.training.actor_critic import RLConfig
from src.training.rollout_ac import run_episode_trajectory
from src.utils.config import load_config
from src.utils.seed import resolve_device


def main() -> None:
    cfg = load_config("configs/train.yaml")
    device = resolve_device(cfg.get("device", "auto"))
    train_names, eval_names, _ = resolve_rc21_split(cfg)
    ds, _ = split_dataset(
        cfg["dataset"]["path"],
        train_ratio=0.8,
        seed=42,
        train_filter=cfg["dataset"].get("train_size_filter"),
        eval_filter=cfg["dataset"].get("eval_size_filter"),
        train_instances=train_names,
        eval_instances=eval_names,
        class_filter=cfg["dataset"].get("class_filter"),
        load_capacity=load_capacity_from_config(cfg),
    )
    inst = ds[0]
    policy = TransformerEVRPTWPolicy(
        cfg["model"], charging_portions=cfg["problem"]["charging_portions"],
    ).to(device)
    rl = RLConfig.from_config(cfg)
    weights = CostWeights.from_config(cfg["cost"])
    t0 = time.time()
    res, traj = run_episode_trajectory(
        inst, policy, m=20, weights=weights, rl_cfg=rl, mode="sampling",
        device=device, charging_portions=cfg["problem"]["charging_portions"],
        collect_trajectory=True, training_mode=True, cycle_id=1,
    )
    n_dec = len(traj.steps) if traj is not None else 0
    print(
        f"epoch_sec={time.time() - t0:.1f} decisions={n_dec} "
        f"served={res.env.num_served_customers}/{inst.num_customers} "
        f"obj={res.objective:.0f} mask_mode={getattr(res.env, 'mask_mode', '?')}"
    )


if __name__ == "__main__":
    main()
