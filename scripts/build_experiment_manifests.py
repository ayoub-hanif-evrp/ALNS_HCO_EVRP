"""Build Round-3 job manifests and inventory reusable results (no algorithm changes)."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.job_store import job_id_fr_pr, job_id_generic  # noqa: E402
from src.alns_hco.reviewer_experiments import BALANCED_12, reviewer_root  # noqa: E402
from src.common.config import (  # noqa: E402
    METHOD_ALNS_FR,
    METHOD_ALNS_PR,
    METHOD_FIXED_DETOUR,
    METHOD_FIXED_HET,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_HET,
    METHOD_PHASE2_NOHCO,
    TW_VARIANTS,
)
from src.hanif.data.schneider_txt_convert import LARGE_56  # noqa: E402
from src.hanif.loader import list_instances  # noqa: E402

HANIF_MAIN_METHODS = (
    METHOD_ALNS_FR,
    METHOD_ALNS_PR,
    METHOD_PHASE1_PR15,
    METHOD_PHASE2_NOHCO,
    METHOD_PHASE2_HET,
)
HANIF_ABLATION_METHODS = (METHOD_FIXED_DETOUR, METHOD_FIXED_HET)
MAIN_SEEDS = (0, 1, 2)
CONV_SEEDS = (0, 1)
CONV_BUDGETS = (30.0, 120.0, 300.0)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def inventory_solution_results() -> dict:
    path = ROOT / "outputs" / "raw" / "solution_results.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_key = {}
    for r in rows:
        key = (r["instance"], r["tw"], r["method"], int(r["seed"]))
        by_key[key] = r
    return {"path": str(path), "n_rows": len(rows), "keys": by_key}


def inventory_fr_pr_checkpoints() -> set[str]:
    ckpt = reviewer_root() / "fr_pr_convergence" / "checkpoints"
    done = set()
    if not ckpt.is_dir():
        return done
    for p in ckpt.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        row = rec.get("row") or {}
        if rec.get("error") or not row:
            continue
        if str(row.get("feasible")).lower() not in {"true", "1"}:
            continue
        if rec.get("job_id"):
            done.add(rec["job_id"])
    return done


def inventory_shared_phase1() -> dict:
    root = ROOT / "outputs" / "shared_phase1"
    keys = []
    for p in root.rglob("seed*.json"):
        tw, inst, seedf = p.relative_to(root).parts
        keys.append((inst, tw, int(seedf.replace("seed", "").replace(".json", ""))))
    return {"n": len(keys), "keys": keys}


def build_hanif_main_manifest(inv_sol: dict) -> dict:
    instances = list_instances()
    planned = []
    for tw in TW_VARIANTS:
        for name in instances:
            for method in HANIF_MAIN_METHODS + HANIF_ABLATION_METHODS:
                for seed in MAIN_SEEDS:
                    jid = f"{method}|{name}|{tw}|seed{seed}|main30"
                    key = (name, tw, method, seed)
                    reused = key in inv_sol["keys"]
                    planned.append({
                        "job_id": jid,
                        "benchmark": "Hanif",
                        "study": "hanif_main",
                        "instance": name,
                        "family": name.split("_")[0],
                        "tw": tw,
                        "method": method,
                        "seed": seed,
                        "budget_s": 15.0 if method == METHOD_PHASE1_PR15 else 30.0,
                        "status": "reusable_solution_results" if reused else "missing_solve",
                        "provenance": "outputs/raw/solution_results.csv" if reused else None,
                    })
    reusable = [j for j in planned if j["status"].startswith("reusable")]
    missing = [j for j in planned if j["status"] == "missing_solve"]
    return {
        "label": "hanif_main_120_tw",
        "n_planned": len(planned),
        "n_reusable_solution_results": len(reusable),
        "n_missing_solve": len(missing),
        "note": (
            "Original publication CSV already covers all 60×2×methods×seeds0-2. "
            "Main work is energy enrichment from routes/shared Phase I, not re-solve."
        ),
        "jobs": planned,
    }


def build_schneider_manifest(done_attempted: set[str] | None = None) -> dict:
    done_attempted = done_attempted or set()
    planned = []
    for name in LARGE_56:
        for method in (METHOD_ALNS_FR, METHOD_ALNS_PR):
            for seed in MAIN_SEEDS:
                job = {
                    "method": method,
                    "instance_name": name,
                    "tw": "schneider",
                    "seed": seed,
                    "budget_s": 30.0,
                }
                jid = job_id_generic(job)
                planned.append({
                    **job,
                    "job_id": jid,
                    "benchmark": "Schneider",
                    "study": "schneider_main",
                    "family": name.split("_")[0].upper() if not name.startswith("rc") else ("RC" + name[2]),
                    "status": "done_or_attempted" if jid in done_attempted else "missing",
                })
    return {
        "label": "schneider_main_56",
        "n_planned": len(planned),
        "n_missing": sum(1 for j in planned if j["status"] == "missing"),
        "jobs": planned,
    }


def build_convergence_288_manifest(done_ids: set[str]) -> dict:
    planned = []
    for budget in CONV_BUDGETS:
        for tw in TW_VARIANTS:
            for name in BALANCED_12:
                for method in (METHOD_ALNS_FR, METHOD_ALNS_PR):
                    for seed in CONV_SEEDS:
                        job = {
                            "method": method,
                            "instance_name": name,
                            "tw": tw,
                            "seed": seed,
                            "budget_s": float(budget),
                        }
                        jid = job_id_fr_pr(job)
                        planned.append({
                            **job,
                            "job_id": jid,
                            "benchmark": "Hanif",
                            "study": "fr_pr_convergence_288",
                            "status": "done" if jid in done_ids else "missing",
                        })
    missing = [j for j in planned if j["status"] == "missing"]
    return {
        "label": "convergence_288_supplementary",
        "n_planned": len(planned),
        "n_done": len(planned) - len(missing),
        "n_missing": len(missing),
        "missing_by_budget": dict(Counter(j["budget_s"] for j in missing)),
        "missing_by_tw": dict(Counter(j["tw"] for j in missing)),
        "jobs": planned,
    }


def main() -> int:
    out = reviewer_root() / "manifests"
    inv_sol = inventory_solution_results()
    # don't serialize huge keys into json
    inv_sol_meta = {
        "path": inv_sol["path"],
        "n_rows": inv_sol["n_rows"],
        "n_keys": len(inv_sol["keys"]),
    }
    shared = inventory_shared_phase1()
    frpr_done = inventory_fr_pr_checkpoints()

    # Schneider attempted from checkpoints
    sch_ckpt = reviewer_root() / "schneider" / "checkpoints"
    sch_done = set()
    if sch_ckpt.is_dir():
        for p in sch_ckpt.glob("*.json"):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            if rec.get("job_id") and rec.get("row") is not None and not rec.get("error"):
                sch_done.add(rec["job_id"])

    hanif = build_hanif_main_manifest(inv_sol)
    sch = build_schneider_manifest(sch_done)
    conv = build_convergence_288_manifest(frpr_done)

    _write_json(out / "inventory.json", {
        "solution_results": inv_sol_meta,
        "shared_phase1_n": shared["n"],
        "fr_pr_checkpoints_done": len(frpr_done),
        "schneider_attempted": len(sch_done),
    })
    _write_json(out / "hanif_main_manifest.json", {k: v for k, v in hanif.items() if k != "jobs"})
    _write_csv(out / "hanif_main_jobs.csv", hanif["jobs"])
    _write_json(out / "schneider_main_manifest.json", {k: v for k, v in sch.items() if k != "jobs"})
    _write_csv(out / "schneider_main_jobs.csv", sch["jobs"])
    _write_json(out / "convergence_288_manifest.json", {k: v for k, v in conv.items() if k != "jobs"})
    _write_csv(out / "convergence_288_jobs.csv", conv["jobs"])
    _write_csv(out / "convergence_288_missing.csv", [j for j in conv["jobs"] if j["status"] == "missing"])

    summary = {
        "hanif_main": {k: hanif[k] for k in hanif if k != "jobs"},
        "schneider_main": {k: sch[k] for k in sch if k != "jobs"},
        "convergence_288": {k: conv[k] for k in conv if k != "jobs"},
        "compute_estimate_hours_rough": {
            "hanif_main_resolves_needed": 0,
            "schneider_missing_30s_2workers": round(sch["n_missing"] * 30 / 2 / 3600, 2),
            "convergence_missing_avg150s_2workers": round(conv["n_missing"] * 150 / 2 / 3600, 2),
        },
    }
    _write_json(out / "SUMMARY.json", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
