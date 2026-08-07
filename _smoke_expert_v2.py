"""Compare old vs new expert ranking on a few Schneider instances (no lookahead)."""
from __future__ import annotations

import time
from pathlib import Path

from src.utils.config import load_config
from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.expert.feasibility_first_expert import FeasibilityFirstExpert
from src.expert.generate_demonstrations import _resolve_expert_m_and_episode

cfg = load_config("configs/train.yaml")
weights = CostWeights.from_eval_config(cfg)
expert = FeasibilityFirstExpert(cfg)
root = Path(cfg["dataset"]["path"])
cap = load_capacity_from_config(cfg)

tests = ["c101_21", "r101_21", "rc101_21", "c201_21", "r208_21", "rc201_21"]
print(f"Expert smoke lookahead={cfg['expert']['lookahead_depth']} fleet_hi={cfg['fleet_search']['customers_per_vehicle_hi']}", flush=True)
for name in tests:
    path = root / name
    if not path.is_dir():
        print(f"  {name}: MISSING", flush=True)
        continue
    inst = load_instance(path, load_capacity=cap)
    t0 = time.perf_counter()
    m, ep, _ = _resolve_expert_m_and_episode(
        inst, config=cfg, expert=expert, weights=weights, collect_transitions=False,
    )
    dt = time.perf_counter() - t0
    met = ep.metrics
    print(
        f"  {name}: m={m} served={met.get('served_ratio', 0):.2f} feas={met.get('feasible')} "
        f"J={met.get('objective', 0):.0f} unserved={met.get('unserved_customers')} t={dt:.1f}s",
        flush=True,
    )
