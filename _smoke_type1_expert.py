"""Type-1 smoke at m=45 with seed + repair burst (gate: feas>=0.5)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.expert.generate_demonstrations import run_expert_episode
from src.utils.config import load_config


def main() -> None:
    config = load_config("configs/train.yaml")
    config["expert"]["lookahead_depth"] = 0
    config["expert"]["repair_max_attempts"] = 150
    root = Path(config["dataset"]["path"])
    cap = load_capacity_from_config(config)
    weights = CostWeights.from_eval_config(config)
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(config))
    portions = list((config.get("problem") or {}).get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    problem = config.get("problem") or {}
    decoding = config.get("decoding") or {}
    rows = []
    for name, m in [("c101_21", 45), ("r101_21", 45), ("rc101_21", 45)]:
        print(f"START {name}", flush=True)
        inst = load_instance(root / name, load_capacity=cap)
        t0 = time.perf_counter()
        ep = run_expert_episode(
            inst, m=m, expert=expert, weights=weights, charging_portions=portions,
            use_horizon_check=bool(problem.get("use_horizon_check", True)),
            use_depot_charging=bool(problem.get("use_depot_charging", True)),
            max_steps_factor=int(decoding.get("max_steps_factor", 6)),
            environment_config=config.get("environment"), collect_transitions=False,
        )
        dt = time.perf_counter() - t0
        met = ep.metrics
        row = {
            "instance": name, "m": m,
            "served_ratio": float(met["served_ratio"]),
            "feasible": bool(met["feasible"]),
            "unserved": int(met["unserved_customers"]),
            "repair_served": int(met.get("repair_served", 0) or 0),
            "used_evs": int(met["used_evs"]),
            "seconds": dt,
            "objective": float(met["objective"]),
        }
        rows.append(row)
        print(
            f"{name}: served={row['served_ratio']:.3f} feas={row['feasible']} "
            f"unserved={row['unserved']} repair={row['repair_served']} ({dt:.1f}s)",
            flush=True,
        )
    feas = sum(1.0 if r["feasible"] else 0.0 for r in rows) / max(len(rows), 1)
    served = sum(r["served_ratio"] for r in rows) / max(len(rows), 1)
    out = {"mean_served": served, "feasibility_rate": feas, "gate_pass": bool(feas >= 0.5), "per_instance": rows}
    Path("outputs/expert_diagnostics").mkdir(parents=True, exist_ok=True)
    Path("outputs/expert_diagnostics/type1_smoke.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"SMOKE served={served:.3f} feas={feas:.3f} gate={out['gate_pass']}", flush=True)


if __name__ == "__main__":
    main()
