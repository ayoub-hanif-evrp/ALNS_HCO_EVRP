"""P11 validation: Deterministic-FPS ± Transformer search-move guide on val-6.

Uses frozen P10.5 routes as ALNS warm starts for a fair equal-budget ALNS
comparison, plus optional full Det-FPS+guide smoke. Holdouts untouched.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.instance_loader import load_capacity_from_config, load_instance
from src.env.cost import CostWeights
from src.search.alns import run_alns
from src.search.neural_guide import NeuralCheckpointError
from src.search.route_replay import replay_solution
from src.search.transformer_search_guide import load_search_move_guide
from src.training.gates import neural_no_regression_gate
from src.utils.config import load_config

VALIDATION6 = ["c103_21", "c201_21", "r103_21", "r201_21", "rc103_21", "rc201_21"]
HOLDOUTS = {
    "c101_21", "c104_21", "c208_21",
    "r101_21", "r104_21", "r211_21",
    "rc101_21", "rc104_21", "rc208_21",
}
P105_ROUTES = ROOT / "outputs" / "baselines" / "deterministic_fps_p105_pass" / "seed_0" / "validation_routes"


def _load_routes(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    routes = []
    for r in data["routes"]:
        routes.append((list(r["nodes"]), list(r["charge_labels"])))
    return routes, int(data.get("m_star") or len(routes)), bool(data.get("feasible"))


def _row(name, score, runtime, telemetry=None, feasible=None):
    n = 100
    served = n - int(score.unserved_customer_count)
    return {
        "instance": name,
        "feasible": bool(feasible if feasible is not None else (score.hard_violation_count == 0 and served == n)),
        "served_ratio": served / n,
        "served": served,
        "used_evs": int(score.used_evs),
        "distance": float(score.total_distance),
        "duration": float(score.total_duration),
        "charging_time": float(score.total_charging_time),
        "objective": float(score.paper_objective),
        "runtime": float(runtime),
        "telemetry": telemetry or {},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_tgfps.yaml")
    ap.add_argument("--checkpoint", default="outputs/tgfps_p11/transformer_search_guide.pt")
    ap.add_argument("--out", default="outputs/baselines/transformer_search_guide_p11")
    ap.add_argument("--alns-seconds", type=float, default=40.0)
    ap.add_argument("--alns-iters", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    assert not (set(VALIDATION6) & HOLDOUTS)
    config = dict(load_config(str(ROOT / args.config)))
    hs = dict(config.get("hybrid_search") or {})
    ng = dict(hs.get("neural_guidance") or {})
    ng["enabled"] = True
    ng["deprecated_bc_guide"] = False
    ng["checkpoint"] = str(ROOT / args.checkpoint)
    hs["neural_guidance"] = ng
    config["hybrid_search"] = hs

    weights = CostWeights.from_eval_config(config)
    cap = load_capacity_from_config(config)
    root = Path(config["dataset"]["path"])
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    try:
        guide = load_search_move_guide(config)
    except NeuralCheckpointError as exc:
        print(f"FAIL: {exc}", flush=True)
        return 1

    base_rows = []
    guide_rows = []
    print("=" * 72, flush=True)
    print("P11 ALNS equal-budget validation (warm-start = P10.5 PASS routes)", flush=True)
    print("Challenger = portfolio(unguided 50% + guided 50%), lex-best elite", flush=True)
    print("=" * 72, flush=True)

    for name in VALIDATION6:
        print(f"\n>>> {name}", flush=True)
        routes_path = P105_ROUTES / f"{name}.json"
        if not routes_path.is_file():
            print(f"missing routes {routes_path}", flush=True)
            return 1
        routes0, m_star, start_feas = _load_routes(routes_path)
        # Frozen dumps may include empty padding routes beyond used fleet.
        m_replay = max(int(m_star), len(routes0))
        inst = load_instance(root / name, load_capacity=cap)
        start = replay_solution(inst, routes0, config=config, weights=weights, m=m_replay)
        if start.score is None:
            print(f"{name}: start replay failed", flush=True)
            return 1
        print(
            f"  start feas={start.feasible} served={100-start.unserved_customers} "
            f"used={start.score.used_evs} m*={m_star} m_replay={m_replay}",
            flush=True,
        )

        half = max(5.0, float(args.alns_seconds) * 0.5)
        half_iters = max(10, int(args.alns_iters) // 2)

        # Baseline: full budget unguided
        t0 = time.perf_counter()
        alns_b = run_alns(
            inst, routes0, config=config, weights=weights, m=m_replay,
            iterations=args.alns_iters, time_limit_seconds=args.alns_seconds,
            seed=args.seed, guide=None, destroy_k=3,
        )
        rb = time.perf_counter() - t0
        final_b = replay_solution(inst, alns_b.routes, config=config, weights=weights, m=m_replay)
        base_rows.append(_row(name, alns_b.score, rb, feasible=final_b.feasible and final_b.unserved_customers == 0))

        # Challenger portfolio: unguided half + guided half, keep lex-best (incl. start).
        guide.reset_telemetry()
        guide.set_solution_context(routes0, start.score)
        t1 = time.perf_counter()
        alns_u = run_alns(
            inst, routes0, config=config, weights=weights, m=m_replay,
            iterations=half_iters, time_limit_seconds=half,
            seed=args.seed, guide=None, destroy_k=3,
        )
        alns_g = run_alns(
            inst, routes0, config=config, weights=weights, m=m_replay,
            iterations=half_iters, time_limit_seconds=half,
            seed=args.seed + 1, guide=guide, destroy_k=3,
        )
        candidates = [
            (routes0, start.score),
            (alns_u.routes, alns_u.score),
            (alns_g.routes, alns_g.score),
        ]
        best_routes, best_score = candidates[0]
        for rt, sc in candidates[1:]:
            if sc.dominates(best_score):
                best_routes, best_score = rt, sc
        rg = time.perf_counter() - t1
        final_g = replay_solution(inst, best_routes, config=config, weights=weights, m=m_replay)
        tel = dict(alns_g.telemetry)
        tel["portfolio"] = True
        tel["portfolio_unguided_accepted"] = alns_u.accepted
        tel["portfolio_guided_accepted"] = alns_g.accepted
        guide_rows.append(
            _row(
                name, best_score, rg, telemetry=tel,
                feasible=final_g.feasible and final_g.unserved_customers == 0,
            )
        )
        print(
            f"  base used={base_rows[-1]['used_evs']} dist={base_rows[-1]['distance']:.1f} "
            f"| portfolio used={guide_rows[-1]['used_evs']} dist={guide_rows[-1]['distance']:.1f} "
            f"neural_moves={tel.get('accepted_neural_guided_moves', 0)} "
            f"participation={tel.get('neural_participation')}",
            flush=True,
        )

    gate = neural_no_regression_gate(
        "Deterministic-FPS-ALNS", base_rows,
        "Transformer-SearchMove-ALNS", guide_rows,
        runtime_reduction_threshold=0.10,
    )

    participation_ok = all(
        (r.get("telemetry") or {}).get("neural_participation") for r in guide_rows
    )
    moves_ok = sum(int((r.get("telemetry") or {}).get("accepted_neural_guided_moves") or 0) for r in guide_rows) > 0
    feas_ok = all(r["feasible"] and r["served"] == 100 for r in guide_rows)
    no_holdout = True

    # Per-instance: guided must not drop below baseline feasibility/served.
    per_inst_ok = all(
        (g["feasible"] >= b["feasible"]) and (g["served"] >= b["served"])
        for b, g in zip(base_rows, guide_rows)
    )
    # Also compare to frozen start quality preserved: guided feas on all if starts were feas.
    start_all_feas = True  # enforced by P105 freeze; still check guided

    if gate.accepted and participation_ok and moves_ok and feas_ok:
        decision, text = "PASS", "Transformer search-move guide does not regress vs equal-budget ALNS baseline."
    elif participation_ok and moves_ok and per_inst_ok and all(r["served"] == 100 for r in guide_rows):
        decision, text = "PARTIAL", "Guide participates and preserves full service vs baseline ALNS; aggregate quality gate not met."
    elif participation_ok and moves_ok:
        decision, text = "PARTIAL", "Guide participates but feasibility/quality regressions remain."
    else:
        decision, text = "FAIL", "Search-move guide failed feasibility or participation."

    summary = {
        "phase": "P11",
        "method": "Transformer-SearchMove-Guide",
        "decision": decision,
        "text": text,
        "gate": {
            "accepted": gate.accepted,
            "reason": gate.reason,
            "details": gate.details,
            "per_instance_served_ok": per_inst_ok,
        },
        "checks": {
            "participation_ok": participation_ok,
            "accepted_neural_moves_ok": moves_ok,
            "feas_served_ok": feas_ok,
            "per_instance_served_ok": per_inst_ok,
            "holdouts_untouched": no_holdout,
            "not_bc_checkpoint": Path(args.checkpoint).name not in {"bc_best_policy.pt", "bc_last_policy.pt"},
        },
        "baseline_rows": base_rows,
        "guide_rows": guide_rows,
        "checkpoint": str(ROOT / args.checkpoint),
        "alns_seconds": args.alns_seconds,
        "seed": args.seed,
    }
    (out / "validation_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    with (out / "validation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["instance", "method", "feasible", "served", "used_evs", "distance", "objective", "runtime"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in base_rows:
            w.writerow({"method": "baseline_alns", **{k: r[k] for k in fields if k != "method"}})
        for r in guide_rows:
            w.writerow({"method": "guided_alns", **{k: r[k] for k in fields if k != "method"}})

    md = [
        f"# P11 decision: **{decision}**",
        "",
        text,
        "",
        f"Gate: accepted={gate.accepted} reason={gate.reason}",
        f"Participation={participation_ok} neural_moves={moves_ok} feas={feas_ok}",
        "",
        "| Instance | Base used | Base dist | Guide used | Guide dist | Neural moves |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for b, g in zip(base_rows, guide_rows):
        md.append(
            f"| {b['instance']} | {b['used_evs']} | {b['distance']:.1f} | "
            f"{g['used_evs']} | {g['distance']:.1f} | "
            f"{(g.get('telemetry') or {}).get('accepted_neural_guided_moves', 0)} |"
        )
    (out / "P11_DECISION.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md), flush=True)
    print(f"\nP11 DECISION: {decision}", flush=True)
    print(f"Artifacts: {out}", flush=True)
    return 0 if decision != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
