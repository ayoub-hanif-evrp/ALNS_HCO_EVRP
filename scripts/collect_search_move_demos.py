"""Collect ALNS search-move demonstrations (train ∪ val only; never holdouts)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.models.search_move_guide import OPS, OP_TO_IDX, build_global_features, build_node_features
from src.search.alns import run_alns
from src.search.fast_construct import matrix_construct
from src.search.repair import repair_solution
from src.utils.config import load_config

HOLDOUTS = {
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
}
VALIDATION6 = ["c103_21", "c201_21", "r103_21", "r201_21", "rc103_21", "rc201_21"]


def _allowed_instances(root: Path, limit: int | None) -> list[str]:
    names = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.endswith("_21"))
    allowed = [n for n in names if n not in HOLDOUTS]
    if limit is not None:
        # Prefer a balanced slice: include all val + sample of train.
        val = [n for n in VALIDATION6 if n in allowed]
        train = [n for n in allowed if n not in set(VALIDATION6)]
        # Take first K train by name for determinism.
        need = max(0, limit - len(val))
        allowed = val + train[:need]
    return allowed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_tgfps.yaml")
    ap.add_argument("--out", default="outputs/demonstrations/schneider21_search_moves.pt")
    ap.add_argument("--limit", type=int, default=18, help="Max instances (val+train, no holdouts)")
    ap.add_argument("--alns-seconds", type=float, default=25.0)
    ap.add_argument("--alns-iters", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    config = dict(load_config(str(ROOT / args.config)))
    weights = CostWeights.from_eval_config(config)
    cap = load_capacity_from_config(config)
    root = Path(config["dataset"]["path"])
    names = _allowed_instances(root, args.limit)
    assert not (set(names) & HOLDOUTS), "holdouts leaked into collection"

    samples = []
    meta_rows = []
    t_all = time.perf_counter()
    print(f"Collecting search-move demos on {len(names)} instances (no holdouts)", flush=True)

    for name in names:
        print(f">>> {name}", flush=True)
        inst = load_instance(root / name, load_capacity=cap)
        # Start from matrix construct + repair at a generous fleet (not fixed-fleet claim).
        m = max(20, int(inst.num_customers // 3) + 8)
        routes, score, feas = matrix_construct(
            inst, m, seed=args.seed, config=config, weights=weights,
        )
        routes, rep = repair_solution(
            inst, routes, config=config, weights=weights, m=m, max_rounds=8, mode="regret",
        )
        if rep.score is not None:
            score = rep.score
        node_feat = build_node_features(inst, routes)
        global_feat = build_global_features(inst, routes, score)

        def _logger(ev):
            # Store light samples only (features + labels), not full routes.
            samples.append({
                "instance": name,
                "op": ev["op"],
                "op_idx": int(OP_TO_IDX.get(ev["op"], 0)),
                "accepted": bool(ev["accepted"]),
                "lex_improved": bool(ev.get("lex_improved", False)),
                "removed": list(ev.get("removed") or []),
                "score_before": list(ev["score_before"]),
                "score_after": list(ev["score_after"]),
                "node_features": node_feat.clone(),
                "global_features": build_global_features(inst, ev["routes"], ev["score_obj"]).clone(),
                "customer_ids": [
                    int(n) for n, node in enumerate(inst.nodes) if node.is_customer
                ],
            })

        alns = run_alns(
            inst, routes, config=config, weights=weights, m=m,
            iterations=args.alns_iters,
            time_limit_seconds=args.alns_seconds,
            seed=args.seed,
            guide=None,
            move_logger=_logger,
        )
        meta_rows.append({
            "instance": name,
            "start_feasible": bool(feas or rep.feasible),
            "alns_accepted": alns.accepted,
            "alns_iters": alns.iterations,
            "final_served": int(inst.num_customers) - int(alns.score.unserved_customer_count),
            "final_used": int(alns.score.used_evs),
            "runtime": alns.elapsed_seconds,
        })
        print(
            f"  samples={len(samples)} accepted={alns.accepted} "
            f"served={meta_rows[-1]['final_served']}/{inst.num_customers} t={alns.elapsed_seconds:.1f}s",
            flush=True,
        )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "tgfps_search_move_v1",
        "ops": list(OPS),
        "n_samples": len(samples),
        "instances": names,
        "holdouts_excluded": sorted(HOLDOUTS),
        "meta": meta_rows,
        "samples": samples,
        "collection_seconds": time.perf_counter() - t_all,
    }
    torch.save(payload, out)
    summary = {
        "format": payload["format"],
        "n_samples": len(samples),
        "n_instances": len(names),
        "instances": names,
        "lex_improved": sum(1 for s in samples if s["lex_improved"]),
        "accepted": sum(1 for s in samples if s["accepted"]),
        "out": str(out),
    }
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
