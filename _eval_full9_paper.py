"""Full-9 Schneider holdout paper eval: Expert / BC / RL-best (greedy, fixed m*)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.data.mixed_class_split import _is_type1_instance, _is_type2_instance
from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights, SolutionSummary
from src.evaluation.fleet_search import FleetCandidate
from src.expert.feasibility_first_expert import ExpertConfig, FeasibilityFirstExpert
from src.expert.generate_demonstrations import run_expert_episode
from src.training.rollout import run_episode
from src.training.train_rl import load_policy
from src.utils.config import load_config

FULL_HOLDOUTS = [
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
]

# Smoke-gate fleet for type-1; type-2 from type-2 v3-style hi probe.
HOLDOUT_M = {
    "c101_21": 45, "c104_21": 45, "c208_21": 31,
    "r101_21": 45, "r104_21": 45, "r211_21": 27,
    "rc101_21": 45, "rc104_21": 45, "rc208_21": 30,
}


def _agg(rows: list[dict], label: str, ckpt: str = "") -> dict:
    n = max(len(rows), 1)
    return {
        "label": label,
        "checkpoint": ckpt,
        "n": len(rows),
        "mean_served": sum(r["served_ratio"] for r in rows) / n,
        "feasibility_rate": sum(1.0 if r["feasible"] else 0.0 for r in rows) / n,
        "mean_objective": sum(r["objective"] for r in rows) / n,
        "per_instance": rows,
        "type2_subset": _subset(rows, TYPE2 := ["c208_21", "r211_21", "rc208_21"]),
        "type1_subset": _subset(
            rows, [n for n in FULL_HOLDOUTS if n not in TYPE2]
        ),
    }


def _subset(rows: list[dict], names: list[str]) -> dict:
    sel = [r for r in rows if r["instance"] in names]
    n = max(len(sel), 1)
    return {
        "n": len(sel),
        "mean_served": sum(r["served_ratio"] for r in sel) / n if sel else 0.0,
        "feasibility_rate": (
            sum(1.0 if r["feasible"] else 0.0 for r in sel) / n if sel else 0.0
        ),
        "mean_objective": sum(r["objective"] for r in sel) / n if sel else 0.0,
    }


def eval_expert(config, names: list[str], out_dir: Path) -> dict:
    expert = FeasibilityFirstExpert(ExpertConfig.from_config(config))
    weights = CostWeights.from_eval_config(config)
    root = Path(config["dataset"]["path"])
    cap = load_capacity_from_config(config)
    portions = list((config.get("problem") or {}).get("charging_portions") or [0.25, 0.5, 0.75, 1.0])
    problem = config.get("problem") or {}
    decoding = config.get("decoding") or {}
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        m = int(HOLDOUT_M[name])
        inst = load_instance(root / name, load_capacity=cap)
        t0 = time.perf_counter()
        ep = run_expert_episode(
            inst,
            m=m,
            expert=expert,
            weights=weights,
            charging_portions=portions,
            use_horizon_check=bool(problem.get("use_horizon_check", True)),
            use_depot_charging=bool(problem.get("use_depot_charging", True)),
            max_steps_factor=int(decoding.get("max_steps_factor", 6)),
            environment_config=config.get("environment"),
            collect_transitions=False,
        )
        met = ep.metrics
        row = {
            "instance": name,
            "m": m,
            "objective": float(met["objective"]),
            "feasible": bool(met["feasible"]),
            "served_ratio": float(met["served_ratio"]),
            "unserved": int(met["unserved_customers"]),
            "used_evs": int(met["used_evs"]),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(
            f"[Expert] {name}: served={row['served_ratio']:.3f} feas={row['feasible']} "
            f"J={row['objective']:.0f} m={m} ({row['seconds']:.1f}s)",
            flush=True,
        )
    summary = _agg(rows, "Expert")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def eval_checkpoint(config, ckpt: Path, names: list[str], out_dir: Path, label: str) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, device=device)
    weights = CostWeights.from_eval_config(config)
    root = Path(config["dataset"]["path"])
    cap = load_capacity_from_config(config)
    portions = list(policy.charging_portions)
    problem = config.get("problem") or {}
    decoding = config.get("decoding") or {}
    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        m = int(HOLDOUT_M[name])
        inst = load_instance(root / name, load_capacity=cap)
        t0 = time.perf_counter()
        res = run_episode(
            inst,
            policy,
            m,
            weights,
            mode="greedy",
            device=device,
            charging_portions=portions,
            use_horizon_check=bool(problem.get("use_horizon_check", True)),
            use_depot_charging=bool(problem.get("use_depot_charging", True)),
            max_steps_factor=int(decoding.get("max_steps_factor", 4)),
        )
        n_cust = max(int(inst.num_customers), 1)
        unserved = int(res.summary.unserved_customers)
        row = {
            "instance": name,
            "m": m,
            "objective": float(res.objective),
            "feasible": bool(res.feasible),
            "served_ratio": float(getattr(res, "served_ratio", (n_cust - unserved) / n_cust)),
            "unserved": unserved,
            "used_evs": int(res.summary.used_evs),
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(
            f"[{label}] {name}: served={row['served_ratio']:.3f} feas={row['feasible']} "
            f"J={row['objective']:.0f} m={m} ({row['seconds']:.1f}s)",
            flush=True,
        )
    summary = _agg(rows, label, str(ckpt))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    run = Path("outputs/runs/run_20260731_232435")
    config = load_config("configs/train_curriculum.yaml")
    root_out = run / "evaluation" / "holdout_full9"
    root_out.mkdir(parents=True, exist_ok=True)

    all_summaries: dict = {}
    print("=== Expert ===", flush=True)
    all_summaries["Expert"] = eval_expert(config, FULL_HOLDOUTS, root_out / "expert")

    for label, path in [
        ("BC", run / "bc" / "bc_best_policy.pt"),
        ("RL-best", run / "checkpoints" / "best.pt"),
    ]:
        if not path.is_file():
            print(f"SKIP {label}: missing {path}", flush=True)
            continue
        print(f"=== {label} ===", flush=True)
        all_summaries[label] = eval_checkpoint(
            config, path, FULL_HOLDOUTS, root_out / label.replace("-", "_").lower(), label,
        )

    (root_out / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8",
    )
    # Also write checkpoint_comparison-shaped methods for paper table builder.
    methods = {
        k: {
            "macro_served": v["mean_served"],
            "macro_feasibility": v["feasibility_rate"],
            "macro_objective": v["mean_objective"],
            "checkpoint": v.get("checkpoint"),
            "type2_subset": v.get("type2_subset"),
            "type1_subset": v.get("type1_subset"),
        }
        for k, v in all_summaries.items()
    }
    cmp = {
        "selection_split": "paper_full9_fixed_m",
        "selected_on_validation": "RL-best",
        "methods": methods,
    }
    (run / "evaluation" / "checkpoint_comparison.json").write_text(
        json.dumps(cmp, indent=2), encoding="utf-8",
    )

    print("\n=== FULL-9 AGGREGATE ===", flush=True)
    for label, s in all_summaries.items():
        print(
            f"{label}: served={s['mean_served']:.3f} feas={s['feasibility_rate']:.3f} "
            f"J={s['mean_objective']:.1f}",
            flush=True,
        )
        t2 = s.get("type2_subset") or {}
        t1 = s.get("type1_subset") or {}
        print(
            f"  type-2: served={t2.get('mean_served', 0):.3f} feas={t2.get('feasibility_rate', 0):.3f} "
            f"| type-1: served={t1.get('mean_served', 0):.3f} feas={t1.get('feasibility_rate', 0):.3f}",
            flush=True,
        )

    from _build_paper_tables import build_full9_table, build_type2_table

    print("Wrote", build_full9_table(run), flush=True)
    # Type-2 paper slice from the same summaries.
    t2_root = run / "evaluation" / "holdout_type2"
    t2_root.mkdir(parents=True, exist_ok=True)
    t2_summaries = {
        k: {
            "n": (v.get("type2_subset") or {}).get("n", 0),
            "mean_served": (v.get("type2_subset") or {}).get("mean_served", 0.0),
            "feasibility_rate": (v.get("type2_subset") or {}).get("feasibility_rate", 0.0),
            "mean_objective": (v.get("type2_subset") or {}).get("mean_objective", 0.0),
        }
        for k, v in all_summaries.items()
    }
    (t2_root / "all_summaries.json").write_text(
        json.dumps(t2_summaries, indent=2), encoding="utf-8",
    )
    print("Wrote", build_type2_table(run), flush=True)
    print("DONE", root_out, flush=True)


if __name__ == "__main__":
    main()
