"""Set-partition archived Schneider routes; keep a plan only if it is lex-better."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schneider.data.instance_loader import load_instance  # noqa: E402
from src.schneider.data.schneider_txt_convert import DEFAULT_CONVERTED, LARGE_56  # noqa: E402
from src.schneider.env.cost import CostWeights  # noqa: E402
from src.schneider.evaluation.schneider_bks import compare_row  # noqa: E402
from src.schneider.evaluation.schneider_incumbent import (  # noqa: E402
    iter_archive_solutions,
    load_best_incumbent,
)
from src.schneider.search.route_pool import RoutePool  # noqa: E402
from src.schneider.search.route_replay import replay_solution  # noqa: E402
from src.schneider.search.set_partitioning import solve_set_partitioning  # noqa: E402
from src.schneider.utils.config import load_config  # noqa: E402


def _write_routes(path: Path, name: str, routes: list, used_k: int, distance: float, feasible: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "instance": name,
        "feasible": bool(feasible),
        "used_k": int(used_k),
        "routes": [
            {
                "nodes": [int(n) for n in nodes],
                "labels": [None if x is None else float(x) for x in (labels or [None] * len(nodes))],
            }
            for nodes, labels in routes
        ],
        "score": {"total_distance": float(distance), "used_evs": int(used_k)},
        "method": "CARE-FI-PR-recombine",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def main() -> int:
    cfg = dict(load_config(str(ROOT / "configs/schneider_fps.yaml")))
    problem = dict(cfg.get("problem") or {})
    problem["charging_portions"] = [0.10, 0.20, 0.35, 0.50, 0.75, 1.0]
    problem["allow_partial_charging"] = True
    problem["greedy_min_charge"] = True
    cfg["problem"] = problem
    weights = CostWeights.from_eval_config(cfg)
    inst_root = ROOT / DEFAULT_CONVERTED
    out_dir = ROOT / "outputs/schneider_recombine/pr/routes"
    out_dir.mkdir(parents=True, exist_ok=True)

    hits = 0
    for name in LARGE_56:
        inst = load_instance(inst_root / name)
        best_routes, meta = load_best_incumbent(name, "pr")
        if not best_routes or not meta.get("feasible"):
            print(f"  skip {name}: no incumbent", flush=True)
            continue
        best_k = int(meta["used_k"])
        pool = RoutePool()
        n_src = 0
        for path, routes, payload in iter_archive_solutions(name, "pr"):
            n_src += 1
            pool.add_from_solution(
                inst, routes,
                source_method=str(path.parent.parent.parent.name),
                source_fleet_size=int(payload.get("used_k") or best_k),
                config=cfg,
            )
        rebuilt, score, sp_meta = solve_set_partitioning(
            inst, pool, config=cfg, weights=weights, m=max(1, best_k - 1),
        )
        if score is None:
            rebuilt, score, sp_meta = solve_set_partitioning(
                inst, pool, config=cfg, weights=weights, m=best_k,
            )
        if score is None or not rebuilt:
            print(
                f"  {name} pool={len(pool)} src={n_src} no-partition "
                f"({sp_meta.get('rejected_reason')})",
                flush=True,
            )
            continue
        used = int(score.used_evs)
        dist = float(score.total_distance)
        check = replay_solution(inst, rebuilt, config=cfg, weights=weights, m=used)
        if not check.feasible or check.score is None:
            print(f"  {name} replay rejected", flush=True)
            continue
        used = int(check.score.used_evs)
        dist = float(check.score.total_distance)
        improved = used < best_k or (used == best_k and dist + 1e-6 < float(meta["distance"]))
        cmp_ = compare_row("pr", name, used, dist)
        flag = "IMPROVED" if improved else "same"
        if improved:
            hits += 1
            _write_routes(out_dir / f"{name}.json", name, rebuilt, used, dist, True)
        print(
            f"  [{flag}] {name} pool={len(pool)} K {best_k}->{used} "
            f"D {meta['distance']:.1f}->{dist:.1f} dK={cmp_['delta_k']:+d}",
            flush=True,
        )
    print(f"recombine wrote {hits} improved incumbents", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
