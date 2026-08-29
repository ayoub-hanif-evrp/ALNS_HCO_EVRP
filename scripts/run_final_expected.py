"""Two-stage Classical fleet search + heterogeneous station refinement.

Frozen results live directly under outputs/{classical,proposed,figures,tables}.
Does not run CVaR/risk-aware. Does not write outputs/figures/ from a pilot.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hanif_convert import VARIANT_SPECS, ensure_hanif_variant_converted
from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.evaluation.hanif_protocols import completeness_report, evaluation_protocol
from src.evaluation.objective_reference import attach_reference, reference_from_classical_routes, reference_path, save_reference
from src.evaluation.output_paths import CLASSICAL, FROZEN_RESULT_DIRS, OUTPUTS, REFERENCES
from src.evaluation.paper_objective import audit_payload
from src.evaluation.validation_gate import decision_payload, evaluate_gate
from src.search.station_refine import exact_row_from_replay, refine_charging
from src.utils.config import load_config

FROZEN = list(FROZEN_RESULT_DIRS)
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
VALIDATION_18 = [f"{fam}_10_{i}" for fam in FAMILIES for i in (1, 2, 3)]
CLASSICAL_DEFAULT = CLASSICAL
REF_DEFAULT = REFERENCES


def _refuse(out: Path) -> None:
    resolved = out.resolve()
    if resolved == OUTPUTS.resolve():
        raise SystemExit("Refusing to overwrite the frozen outputs/ archive. Pass --out to a new directory.")
    for frozen in FROZEN:
        fr = frozen.resolve()
        if resolved == fr or fr in resolved.parents:
            raise SystemExit(f"Refusing to overwrite frozen results: {frozen}")


def _load_routes(exp_dir: Path, tw: str, name: str):
    path = Path(exp_dir) / tw / "routes" / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(list(r["nodes"]), list(r.get("labels") or [None] * len(r["nodes"]))) for r in payload["routes"]]


def _serialize_routes(routes: list) -> list[dict]:
    out = []
    for nodes, labels in routes:
        out.append({
            "nodes": [int(n) for n in nodes],
            "labels": [None if x is None else float(x) for x in (labels or [None] * len(nodes))],
        })
    return out


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def freeze_classical_references(classical_dir: Path, ref_dir: Path, names: list[str]) -> dict:
    combined: dict[str, dict] = {}
    for variant, tw in (("A", "original_tw"), ("B", "relaxed_tw")):
        root = Path(VARIANT_SPECS[variant]["output"])
        ensure_hanif_variant_converted(variant)
        combined[tw] = {}
        for name in names:
            routes = _load_routes(classical_dir, tw, name)
            if not routes:
                continue
            inst = load_instance(root / name)
            ref = reference_from_classical_routes(inst, routes)
            dest = reference_path(ref_dir, variant, name)
            save_reference(dest, ref)
            combined[tw][name] = ref.as_dict()
            print(f"ref {tw}/{name} {ref.as_dict()}", flush=True)
    (ref_dir / "objective_reference.json").write_text(
        json.dumps({"source": "frozen_classical_solution", "immutable": True, "by_variant": combined}, indent=2) + "\n",
        encoding="utf-8",
    )
    return combined


def _run_fps(*, names: list[str], out: Path, objective: str, budget: float, workers: int, reference_dir: Path | None) -> None:
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/run_hanif_fps.py"),
        "--variant", "both", "--budget", str(budget), "--workers", str(workers),
        "--objective", objective, "--out", str(out),
        "--instances", *names,
    ]
    if reference_dir is not None:
        cmd.extend(["--reference-dir", str(reference_dir)])
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def _copy_classical_tables(src: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "SOURCE.json").write_text(
        json.dumps({"copied_from": str(src).replace("\\", "/"), "routes_remain_at_source": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    for tw in ("original_tw", "relaxed_tw"):
        twd = dest / tw
        twd.mkdir(parents=True, exist_ok=True)
        for fname in ("results.csv", "summary.json"):
            sp = src / tw / fname
            if sp.is_file():
                shutil.copy2(sp, twd / fname)
        src_routes = src / tw / "routes"
        dst_routes = twd / "routes"
        if src_routes.is_dir():
            dst_routes.mkdir(parents=True, exist_ok=True)
            for route_file in src_routes.glob("*.json"):
                target = dst_routes / route_file.name
                if not target.is_file():
                    shutil.copy2(route_file, target)
    for fname in ("summary_both.json", "results_both.csv"):
        sp = src / fname
        if sp.is_file():
            shutil.copy2(sp, dest / fname)


def _classical_complete(classical_dir: Path, names: list[str]) -> bool:
    for tw in ("original_tw", "relaxed_tw"):
        rows = _read_csv(classical_dir / tw / "results.csv")
        got = {r.get("instance") for r in rows}
        if any(n not in got for n in names):
            return False
        if any(_load_routes(classical_dir, tw, n) is None for n in names):
            return False
    return True


def _refine_worker(payload: dict) -> dict:
    """Process-pool worker: inherit Classical routes, freeze K, refine stations."""
    name = payload["name"]
    variant = payload["variant"]
    tw = payload["tw"]
    classical_dir = Path(payload["classical_dir"])
    ref_dir = Path(payload["ref_dir"])
    out_tw = Path(payload["out_tw"])
    budget = float(payload["budget"])
    classical_runtime = float(payload["classical_runtime"])
    routes = _load_routes(classical_dir, tw, name)
    if routes is None:
        raise FileNotFoundError(f"missing Classical routes {tw}/{name}")
    root = Path(VARIANT_SPECS[variant]["output"])
    inst = load_instance(root / name)
    cfg = dict(load_config(str(ROOT / "configs/hanif_fps.yaml")))
    cfg["paper_objective"] = {"mode": "expected_heterogeneous", "risk_blend": 0.0, "risk_alpha": 0.95}
    cfg.setdefault("experiment", {})["objective_reference_dir"] = str(ref_dir)
    cfg.setdefault("dataset", {})["tw_variant"] = variant
    weights = CostWeights.from_eval_config(cfg)
    cfg = attach_reference(inst, cfg, weights, reference_dir=ref_dir, variant=variant)
    from src.search.exact_checks import count_nonempty_routes

    k_star = int(count_nonempty_routes(inst, routes))
    t0 = time.perf_counter()
    result = refine_charging(
        inst, routes, config=cfg, weights=weights, budget_seconds=budget, k_star=k_star,
    )
    refine_s = time.perf_counter() - t0
    row, exact = exact_row_from_replay(
        inst, result.routes, config=cfg, weights=weights, name=name, k_star=k_star,
        runtime_refine_s=refine_s, runtime_classical_s=classical_runtime,
        extra={
            "variant": VARIANT_SPECS[variant]["key"],
            "tw_variant": variant,
            "accepted_moves": result.accepted,
            "tried_moves": result.tried,
            "phi_initial": result.phi_initial,
            "phi_final": result.phi_final,
            "refine_termination": result.termination,
        },
    )
    routes_dir = out_tw / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)
    payload_out = {
        "instance": name,
        "feasible": bool(row["feasible"]),
        "used_k": row["used_k"],
        "k_star": k_star,
        "exact_check": exact,
        "stage": "station_refinement",
        "inherits_classical": True,
        "routes": _serialize_routes(result.routes),
        "method": "StationRefinement",
        "evaluation_protocol": evaluation_protocol(cfg),
        "phi_initial": result.phi_initial,
        "phi_final": result.phi_final,
        "accepted_moves": result.accepted,
        "tried_moves": result.tried,
        "runtime_refine_s": refine_s,
        "runtime_classical_s": classical_runtime,
        "runtime_s": float(classical_runtime) + refine_s,
    }
    (routes_dir / f"{name}.json").write_text(json.dumps(payload_out), encoding="utf-8")
    print(
        f"  [{tw}] {name} feas={row['feasible']} K={row['used_k']} (frozen {k_star}) "
        f"Phi {result.phi_initial:.4f}->{result.phi_final:.4f} "
        f"accepted={result.accepted}/{result.tried} t={refine_s:.1f}s",
        flush=True,
    )
    return row


def _write_proposed_variant(out_tw: Path, rows: list[dict], *, names: list[str], budget: float, wall_s: float) -> dict:
    rows = sorted(rows, key=lambda r: r["instance"])
    complete = completeness_report(expected_names=names, got_names=[r["instance"] for r in rows])
    out_tw.mkdir(parents=True, exist_ok=True)
    if rows:
        with (out_tw / "results.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    proto = evaluation_protocol({"paper_objective": {"mode": "expected_heterogeneous"}})
    summary = {
        "method": "StationRefinement",
        "stage": "heterogeneous_charging_refinement",
        "inherits_classical": True,
        "k_frozen": True,
        "evaluation_protocol": proto,
        "attributes_used": dict(proto),
        "n": len(rows),
        "expected_n": len(names),
        "complete": complete["complete"],
        "completeness": complete,
        "budget_s": budget,
        "wall_s": wall_s,
        "mean_k": sum(float(r["used_k"]) for r in rows) / len(rows) if rows and complete["complete"] else None,
        "mean_runtime_s": sum(float(r["runtime_s"]) for r in rows) / len(rows) if rows and complete["complete"] else None,
        "mean_runtime_refine_s": sum(float(r.get("runtime_refine_s") or 0) for r in rows) / len(rows) if rows else None,
        "mean_runtime_classical_s": sum(float(r.get("runtime_classical_s") or 0) for r in rows) / len(rows) if rows else None,
        "rows": rows,
    }
    (out_tw / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_refinement(
    *,
    names: list[str],
    classical_dir: Path,
    ref_dir: Path,
    proposed_dir: Path,
    budget: float,
    workers: int,
) -> None:
    classical_runtime = {}
    for tw in ("original_tw", "relaxed_tw"):
        for r in _read_csv(classical_dir / tw / "results.csv"):
            classical_runtime[(tw, r["instance"])] = float(r.get("runtime_s") or 0.0)
    jobs = []
    for variant, tw in (("A", "original_tw"), ("B", "relaxed_tw")):
        (proposed_dir / tw / "routes").mkdir(parents=True, exist_ok=True)
        for name in names:
            jobs.append({
                "name": name,
                "variant": variant,
                "tw": tw,
                "classical_dir": str(classical_dir),
                "ref_dir": str(ref_dir),
                "out_tw": str(proposed_dir / tw),
                "budget": budget,
                "classical_runtime": classical_runtime.get((tw, name), 0.0),
            })
    gathered = {"original_tw": [], "relaxed_tw": []}
    t0 = time.perf_counter()
    if workers <= 1:
        for job in jobs:
            row = _refine_worker(job)
            tw = "original_tw" if job["variant"] == "A" else "relaxed_tw"
            gathered[tw].append(row)
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futs = {pool.submit(_refine_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                job = futs[fut]
                row = fut.result()
                tw = "original_tw" if job["variant"] == "A" else "relaxed_tw"
                gathered[tw].append(row)
    wall = time.perf_counter() - t0
    summaries = []
    for tw in ("original_tw", "relaxed_tw"):
        summaries.append(_write_proposed_variant(proposed_dir / tw, gathered[tw], names=names, budget=budget, wall_s=wall))
    if len(summaries) == 2:
        combined = {
            "method": "StationRefinement",
            "evaluation_protocol": evaluation_protocol({"paper_objective": {"mode": "expected_heterogeneous"}}),
            "complete": all(s.get("complete") for s in summaries),
            "runtime_accounting": "runtime_s = Classical fleet search + station refinement",
            "variants": {tw: {k: s[k] for k in s if k != "rows"} for tw, s in zip(("original_tw", "relaxed_tw"), summaries)},
        }
        (proposed_dir / "summary_both.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        all_rows = summaries[0]["rows"] + summaries[1]["rows"]
        with (proposed_dir / "results_both.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)


def _generate_artifacts(exp: Path, *, frozen_final: bool) -> None:
    canonical = frozen_final and exp.resolve() == OUTPUTS.resolve()
    paper = OUTPUTS if canonical else exp
    cmd = [
        sys.executable, "-u", str(ROOT / "scripts/generate_final_paper.py"),
        "--exp", str(exp),
        "--paper", str(paper),
        "--validation-only",
    ]
    if canonical:
        cmd.append("--frozen-final")
    print("RUN", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=180.0, help="Classical fleet-search budget (if Classical is re-run)")
    ap.add_argument("--refine-budget", type=float, default=90.0, help="Stage-2 station refinement budget per instance")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument(
        "--out",
        required=True,
        help="New experiment directory. Frozen paper results already live in outputs/.",
    )
    ap.add_argument("--classical-dir", default=str(CLASSICAL_DEFAULT))
    ap.add_argument("--reference-dir", default=str(REF_DEFAULT))
    ap.add_argument("--skip-classical", action="store_true")
    ap.add_argument("--skip-proposed", action="store_true")
    ap.add_argument("--full", action="store_true", help="Run 60+60 only after validation PASS")
    ap.add_argument("--force-full", action="store_true", help="Run full even if validation fails (not recommended)")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    _refuse(out)
    out.mkdir(parents=True, exist_ok=True)
    names = VALIDATION_18
    if args.full and args.skip_classical:
        names = sorted(p.name for p in Path(VARIANT_SPECS["A"]["output"]).iterdir() if p.is_dir())
    classical_src = Path(args.classical_dir)
    if not classical_src.is_absolute():
        classical_src = ROOT / classical_src
    ref_dir = Path(args.reference_dir)
    if not ref_dir.is_absolute():
        ref_dir = ROOT / ref_dir

    (out / "objective_audit.json").write_text(
        json.dumps({
            "classical": audit_payload({"paper_objective": {"mode": "classical_distance"}}),
            "proposed": audit_payload({"paper_objective": {"mode": "expected_heterogeneous", "risk_blend": 0.0}}),
            "method": "Classical fleet minimization, then K-frozen heterogeneous station refinement",
            "risk_aware_not_run": True,
            "legacy_best": "removed (old wait_mu-as-minutes archive; do not cite 80.07/65.15)",
            "Phi_reference": "frozen Classical solution per instance",
            "previous_independent_search": "removed (independent Φ fleet search; scientific FAIL; not the paper method)",
        }, indent=2) + "\n",
        encoding="utf-8",
    )

    classical_dir = out / "classical"
    proposed_dir = out / "proposed"
    reuse_classical = (not args.skip_proposed) and _classical_complete(classical_src, names)
    if args.skip_classical or reuse_classical:
        if not _classical_complete(classical_src, names):
            raise SystemExit(f"Classical archive incomplete: {classical_src}")
        print(f"Reusing Classical Stage 1 from {classical_src}", flush=True)
        _copy_classical_tables(classical_src, classical_dir)
        if not (ref_dir / "objective_reference.json").is_file():
            freeze_classical_references(classical_src, ref_dir, names)
    else:
        _run_fps(
            names=names, out=classical_dir, objective="classical",
            budget=args.budget, workers=args.workers, reference_dir=None,
        )
        freeze_classical_references(classical_dir, out / "references", names)
        ref_dir = out / "references"
        classical_src = classical_dir

    if not args.skip_proposed:
        _run_refinement(
            names=names,
            classical_dir=classical_src if reuse_classical else classical_dir,
            ref_dir=ref_dir,
            proposed_dir=proposed_dir,
            budget=args.refine_budget,
            workers=args.workers,
        )

    gate = evaluate_gate(classical_dir, proposed_dir)
    (out / "VALIDATION.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    decision = decision_payload(gate, out_dir=out)
    (out / "DECISION.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)
    print(json.dumps({k: decision[k] for k in decision if k != "gate"}, indent=2), flush=True)

    if args.full and not args.skip_classical:
        if not gate["pass"] and not args.force_full:
            print("VALIDATION FAILED — not running the full 120. Keep Classical as the main solver.", flush=True)
            _generate_artifacts(out, frozen_final=False)
            return 2
        _refuse(out)
        full_ref = out / "references"
        all_names = sorted(p.name for p in Path(VARIANT_SPECS["A"]["output"]).iterdir() if p.is_dir())
        _run_fps(names=all_names, out=out / "classical", objective="classical", budget=args.budget, workers=args.workers, reference_dir=None)
        freeze_classical_references(out / "classical", full_ref, all_names)
        _run_refinement(
            names=all_names,
            classical_dir=out / "classical",
            ref_dir=full_ref,
            proposed_dir=out / "proposed",
            budget=args.refine_budget,
            workers=args.workers,
        )
        full_gate = evaluate_gate(out / "classical", out / "proposed")
        (out / "VALIDATION.json").write_text(json.dumps(full_gate, indent=2) + "\n", encoding="utf-8")
        _generate_artifacts(out, frozen_final=bool(full_gate.get("pass")))
        return 0 if full_gate.get("pass") else 2

    _generate_artifacts(out, frozen_final=False)
    return 0 if gate["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
