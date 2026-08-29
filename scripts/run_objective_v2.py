"""P13 objective_v2 driver: freeze CurrentBest, run A/B/C, write audit.

Never writes to outputs/CurrentBestDeterministicHanif/.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.hanif_convert import VARIANT_SPECS, ensure_hanif_variant_converted
from src.data.instance_loader import load_instance
from src.env.cost import CostWeights
from src.evaluation.hanif_protocols import FROZEN_HEADLINE
from src.evaluation.objective_reference import build_objective_reference, save_reference, reference_path
from src.evaluation.paper_objective import audit_payload, independent_recompute_phi, phi_from_replay
from src.search.route_replay import replay_solution
from src.utils.config import load_config

FROZEN = ROOT / "outputs/CurrentBestDeterministicHanif"
OUT = ROOT / "outputs/objective_v2"
PILOT = ["C1_10_1", "C2_10_1", "R1_10_1", "R2_10_1", "RC1_10_1", "RC2_10_1"]
MODES = [
    ("classical", "classical"),
    ("expected", "expected"),
    ("risk_aware", "risk_aware"),
]


def _load_frozen_routes(variant: str, name: str) -> list | None:
    sub = "original_tw" if variant == "A" else "relaxed_tw"
    path = FROZEN / sub / "routes" / f"{name}.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [(list(r["nodes"]), list(r.get("labels") or [None] * len(r["nodes"]))) for r in payload["routes"]]


def replay_frozen_incumbents(names: list[str]) -> dict:
    report = {
        "frozen": str(FROZEN).replace("\\", "/"),
        "note": "Routes were optimized under raw wait_mu timing; replay uses E[W]=exp(mu+0.5 sigma^2).",
        "variants": {},
    }
    for variant, key in (("A", "original_tw"), ("B", "relaxed_tw")):
        root = Path(VARIANT_SPECS[variant]["output"])
        ensure_hanif_variant_converted(variant)
        rows = []
        for name in names:
            inst = load_instance(root / name)
            routes = _load_frozen_routes(variant, name)
            if routes is None:
                rows.append({"instance": name, "status": "missing_frozen_route"})
                continue
            res = replay_solution(inst, routes, m=len(routes))
            rows.append({
                "instance": name,
                "status": "feasible_incumbent" if res.feasible else "infeasible_under_expected_wait",
                "feasible": bool(res.feasible),
                "used_k": int(res.used_evs),
                "distance": float(res.distance),
                "station_waiting_time": float(res.station_waiting_time),
                "customer_early_waiting_time": float(res.customer_early_waiting_time),
                "first_failure": res.first_failure,
                "retained_as_incumbent": bool(res.feasible),
            })
        n_ok = sum(1 for r in rows if r.get("feasible"))
        report["variants"][variant] = {
            "n": len(rows),
            "feasible": n_ok,
            "infeasible": len(rows) - n_ok,
            "rows": rows,
        }
    return report


def write_audit(out_dir: Path) -> None:
    audits = {}
    for mode, _cli in MODES:
        cfg = {
            "paper_objective": {
                "mode": {
                    "classical": "classical_distance",
                    "expected": "expected_heterogeneous",
                    "risk_aware": "risk_aware_heterogeneous",
                }[mode],
                "risk_alpha": 0.95,
                "risk_blend": 0.0 if mode != "risk_aware" else 0.25,
            }
        }
        audits[mode] = audit_payload(cfg)
    audits["_shared"] = {
        "previous_best": str(FROZEN).replace("\\", "/"),
        "new_experiment": str(out_dir).replace("\\", "/"),
        "Phi": "D/D_ref + Tstation/Tstation_ref + Crisk/Ccharge_ref",
        "Tstation": "Tchg + Wrisk",
        "effective": "(1-rho)*E[X] + rho*CVaR95(X)",
        "feasibility_wait": "E[W]=exp(wait_mu + 0.5 wait_sigma^2)",
        "not_chance_constrained": True,
        "density_in_objective": False,
        "reference_constructor": "dense_construct under classical ranking; components measured at rho=0",
        "frozen_headline": FROZEN_HEADLINE,
    }
    (out_dir / "objective_audit.json").write_text(json.dumps(audits, indent=2) + "\n", encoding="utf-8")


def build_shared_references(out_dir: Path, names: list[str]) -> None:
    cfg = dict(load_config(str(ROOT / "configs/hanif_fps.yaml")))
    weights = CostWeights.from_eval_config(cfg)
    ref_root = out_dir / "references"
    for variant in ("A", "B"):
        root = Path(VARIANT_SPECS[variant]["output"])
        ensure_hanif_variant_converted(variant)
        for name in names:
            dest = reference_path(ref_root, variant, name)
            if dest.is_file():
                continue
            inst = load_instance(root / name)
            ref = build_objective_reference(inst, cfg, weights)
            save_reference(dest, ref)
            print(f"reference {variant}/{name} -> {ref.as_dict()}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-solve", action="store_true")
    ap.add_argument("--build-references", action="store_true")
    ap.add_argument("--scenarios", type=int, default=100)
    ap.add_argument("--out", default="outputs/objective_v2")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    if out_dir.resolve() == FROZEN.resolve():
        raise SystemExit("Refusing to overwrite CurrentBestDeterministicHanif")
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_hanif_variant_converted("A")
    names = PILOT if (args.pilot or not args.all) else [
        p.name for p in Path(VARIANT_SPECS["A"]["output"]).iterdir() if p.is_dir()
    ]

    write_audit(out_dir)
    incumbents = replay_frozen_incumbents(names)
    (out_dir / "incumbent_replay.json").write_text(
        json.dumps(incumbents, indent=2) + "\n", encoding="utf-8",
    )

    checks = []
    root_a = Path(VARIANT_SPECS["A"]["output"])
    cfg0 = {
        "paper_objective": {"mode": "expected_heterogeneous", "risk_blend": 0.0, "risk_alpha": 0.95},
        "objective_reference": {"D_ref": 1.0, "Tstation_ref": 1.0, "Ccharge_ref": 1.0},
    }
    for name in names[:3]:
        inst = load_instance(root_a / name)
        routes = _load_frozen_routes("A", name)
        if not routes:
            continue
        res = replay_solution(inst, routes, m=len(routes), config=cfg0)
        phi = phi_from_replay(inst, res, cfg0)
        indep = independent_recompute_phi(inst, res, cfg0)
        checks.append({
            "instance": name,
            "feasible": res.feasible,
            "phi": phi["Phi"],
            "distance": phi["distance"],
            "tstation": phi["tstation"],
            "crisk": phi["crisk"],
            "charging_time": phi["charging_time"],
            "independent_agreement": bool(indep["agree"]),
            "independent": indep,
        })
    (out_dir / "manual_phi_checks.json").write_text(json.dumps(checks, indent=2) + "\n", encoding="utf-8")

    if args.skip_solve:
        print("skip-solve; audit and incumbent replay written", flush=True)
        print(json.dumps({
            k: {kk: vv for kk, vv in v.items() if kk != "rows"}
            for k, v in incumbents["variants"].items()
        }, indent=2))
        if args.build_references:
            build_shared_references(out_dir, names)
        return 0

    build_shared_references(out_dir, names)
    ref_dir = out_dir / "references"
    for mode, cli in MODES:
        dest = out_dir / mode
        cmd = [
            sys.executable, "-u", str(ROOT / "scripts/run_hanif_fps.py"),
            "--variant", "both", "--budget", str(args.budget),
            "--workers", str(args.workers),
            "--objective", cli,
            "--reference-dir", str(ref_dir),
            "--out", str(dest),
        ]
        if args.all:
            cmd.append("--all")
        else:
            cmd.append("--pilot")
        print("RUN", " ".join(cmd), flush=True)
        subprocess.check_call(cmd, cwd=str(ROOT))

    paper_cmd = [
        sys.executable, "-u", str(ROOT / "scripts/generate_objective_v2_paper.py"),
        "--exp", str(out_dir),
        "--scenarios", str(args.scenarios),
    ]
    print("RUN", " ".join(paper_cmd), flush=True)
    subprocess.check_call(paper_cmd, cwd=str(ROOT))
    print(f"objective_v2 -> {out_dir}", flush=True)
    print(f"previous_best -> {FROZEN}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
