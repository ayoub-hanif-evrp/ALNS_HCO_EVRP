"""Holdout eval for type-2 Schneider: bc_best + rl_best_* (expert_probes=1)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights, SolutionSummary
from src.evaluation.fleet_search import FleetCandidate, fleet_search
from src.evaluation.mstar_cache import load_mstar_map
from src.training.rollout import run_episode
from src.training.train_rl import load_policy
from src.utils.config import load_config


def _summary_from_row(row: dict) -> SolutionSummary:
    return SolutionSummary(
        used_evs=int(row.get("used_evs", 0)),
        total_distance=float(row.get("distance", 0.0)),
        total_duration=float(row.get("duration", 0.0)),
        total_charging_time=float(row.get("charging", 0.0)),
        unserved_customers=int(row.get("unserved", 0)),
    )


def eval_checkpoint(config, ckpt: Path, names: list[str], out_dir: Path, label: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = load_policy(ckpt, device=device)
    weights = CostWeights.from_eval_config(config)
    root = Path(config["dataset"]["path"])
    cap = load_capacity_from_config(config)
    mstar = load_mstar_map(config)
    fcfg = dict(config.get("fleet_search") or {})
    fcfg["expert_probes"] = 1
    fcfg["lexicographic"] = True
    portions = list(policy.charging_portions)
    problem = config.get("problem") or {}
    decoding = config.get("decoding") or {}
    use_horizon = bool(problem.get("use_horizon_check", True))
    use_depot = bool(problem.get("use_depot_charging", True))
    max_steps = int(decoding.get("max_steps_factor", 4))

    def runner(instance, m):
        res = run_episode(
            instance,
            policy,
            int(m),
            weights,
            mode="greedy",
            device=device,
            charging_portions=portions,
            use_horizon_check=use_horizon,
            use_depot_charging=use_depot,
            max_steps_factor=max_steps,
        )
        return FleetCandidate(
            m=int(m),
            feasible=bool(res.feasible),
            objective=float(res.objective),
            summary=res.summary,
            result=res,
        )

    rows = []
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in names:
        inst = load_instance(root / name, load_capacity=cap)
        t0 = time.perf_counter()
        if name in mstar:
            cand = runner(inst, int(mstar[name]))
            fleet_mode = "mstar_cache"
        else:
            cand, _ = fleet_search(
                inst,
                runner,
                extra_margin=int(fcfg.get("extra_margin", 12)),
                selection_mode=str(fcfg.get("selection_mode", "hierarchical")),
                fleet_cfg=fcfg,
            )
            fleet_mode = "lex_probe_hi"
        res = cand.result
        n_cust = max(int(inst.num_customers), 1)
        unserved = int(cand.summary.unserved_customers)
        row = {
            "instance": name,
            "m": int(cand.m),
            "objective": float(cand.objective),
            "feasible": bool(cand.feasible),
            "served_ratio": float((n_cust - unserved) / n_cust),
            "unserved": unserved,
            "used_evs": int(cand.summary.used_evs),
            "fleet_mode": fleet_mode,
            "seconds": time.perf_counter() - t0,
        }
        if res is not None and hasattr(res, "served_ratio"):
            row["served_ratio"] = float(res.served_ratio)
        rows.append(row)
        print(
            f"[{label}] {name}: served={row['served_ratio']:.3f} feas={row['feasible']} "
            f"J={row['objective']:.0f} m={row['m']} ({row['seconds']:.1f}s)",
            flush=True,
        )

    summary = {
        "label": label,
        "checkpoint": str(ckpt),
        "n": len(rows),
        "mean_served": sum(r["served_ratio"] for r in rows) / max(len(rows), 1),
        "feasibility_rate": sum(1.0 if r["feasible"] else 0.0 for r in rows) / max(len(rows), 1),
        "mean_objective": sum(r["objective"] for r in rows) / max(len(rows), 1),
        "per_instance": rows,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    run = Path("outputs/runs/run_20260729_140606")
    config = load_config(str(run / "config.yaml"))
    live = load_config("configs/train.yaml")
    config["fleet_search"] = live.get("fleet_search") or config.get("fleet_search")
    # Prefer live m* cache path if present.
    if (live.get("demonstrations") or {}).get("mstar_cache"):
        config.setdefault("demonstrations", {})["mstar_cache"] = live["demonstrations"]["mstar_cache"]
    # Also try run-local m*
    run_mstar = run / "demonstrations" / "mstar.json"
    if run_mstar.is_file():
        config.setdefault("demonstrations", {})["mstar_cache"] = str(run_mstar)
    type2_mstar = Path("outputs/demonstrations/schneider21_type2_v2_mstar.json")
    if type2_mstar.is_file() and "mstar_cache" not in (config.get("demonstrations") or {}):
        config.setdefault("demonstrations", {})["mstar_cache"] = str(type2_mstar)

    holdouts = ["c208_21", "r211_21", "rc208_21"]
    ckpts = []
    for label, path in [
        ("bc_best", run / "bc" / "bc_best_policy.pt"),
        ("rl_best", run / "rl" / "rl_best.pt"),
        ("rl_best_feasibility", run / "rl" / "rl_best_feasibility.pt"),
        ("rl_best_objective", run / "rl" / "rl_best_objective.pt"),
        ("checkpoints_best", run / "checkpoints" / "best.pt"),
    ]:
        if path.is_file():
            ckpts.append((label, path))
    if not ckpts:
        raise SystemExit("No checkpoints found")

    root_out = run / "evaluation" / "holdout_type2"
    all_summaries = {}
    for label, path in ckpts:
        print(f"\n=== {label} ===", flush=True)
        all_summaries[label] = eval_checkpoint(
            config, path, holdouts, root_out / label, label,
        )
    (root_out / "all_summaries.json").write_text(
        json.dumps(all_summaries, indent=2), encoding="utf-8",
    )
    print("\n=== AGGREGATE ===", flush=True)
    for label, s in all_summaries.items():
        print(
            f"{label}: served={s['mean_served']:.3f} feas={s['feasibility_rate']:.3f} "
            f"J={s['mean_objective']:.1f}",
            flush=True,
        )
    print("DONE", root_out)


if __name__ == "__main__":
    main()
