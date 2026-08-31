"""Compare exact CARE-FI (v2) against the kept v1 solver and the frozen archive.

Does not overwrite outputs/classical or outputs/proposed.

  python scripts/compare_hanif_carefi.py --representative --tw both --budget 180 --workers 2
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hanif.loader import REPRESENTATIVE, load_hanif  # noqa: E402
from src.hanif.paths import FROZEN_HANIF_CLASSICAL, HANIF_OUT  # noqa: E402
from src.hanif.solver import load_archived_classical  # noqa: E402


def _frozen_k(name: str, tw: str) -> int | None:
    routes = load_archived_classical(name, tw)
    if routes is None:
        return None
    payload_path = (
        FROZEN_HANIF_CLASSICAL
        / ("relaxed_tw" if tw.startswith("relax") else "original_tw")
        / "routes" / f"{name}.json"
    )
    if payload_path.is_file():
        row = json.loads(payload_path.read_text(encoding="utf-8"))
        if "used_k" in row:
            return int(row["used_k"])
    from src.hanif.replay import replay_solution

    inst = load_hanif(name, tw)
    return int(replay_solution(inst, routes).used_k)


def _worker(job: dict) -> dict:
    inst = load_hanif(job["name"], job["tw"])
    if job["solver"] == "current":
        from src.hanif.current.solver import solve_carefi
    else:
        from src.hanif.solver import solve_carefi
    s1, _s2 = solve_carefi(inst, budget_s=job["budget"], stage2=False)
    return {
        "instance": job["name"],
        "tw": job["tw"],
        "solver": job["solver"],
        "feasible": bool(s1.metrics.feasible),
        "used_k": int(s1.metrics.used_k),
        "distance": float(s1.metrics.distance),
        "unserved": int(s1.metrics.unserved),
        "hard_violations": int(s1.metrics.hard_violations),
        "runtime_s": float(s1.runtime_s),
        "eliminations": int((s1.stages.get("fleet_reduction") or {}).get("eliminations") or 0),
        "construction_k": (s1.stages.get("construction") or {}).get("k"),
    }


def _md_table(rows: list[dict], tw: str) -> str:
    subset = [r for r in rows if r["tw"] == tw]
    names = sorted({r["instance"] for r in subset})
    lines = [
        f"## {tw.title()} TW",
        "",
        "| Instance | Frozen K | Current v1 K | Exact v2 K | v2-v1 | v2-frozen | v2 feas |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    f_sum = c_sum = e_sum = n = 0
    for name in names:
        frozen = next((r["frozen_k"] for r in subset if r["instance"] == name), None)
        cur = next((r for r in subset if r["instance"] == name and r["solver"] == "current"), None)
        exact = next((r for r in subset if r["instance"] == name and r["solver"] == "exact"), None)
        ck = cur["used_k"] if cur else None
        ek = exact["used_k"] if exact else None
        feas = "yes" if exact and exact["feasible"] else "no"
        d1 = "" if ck is None or ek is None else f"{ek - ck:+d}"
        d0 = "" if frozen is None or ek is None else f"{ek - frozen:+d}"
        lines.append(
            f"| {name} | {frozen if frozen is not None else ''} | "
            f"{ck if ck is not None else ''} | {ek if ek is not None else ''} | {d1} | {d0} | {feas} |"
        )
        if frozen is not None and ck is not None and ek is not None:
            f_sum += frozen
            c_sum += ck
            e_sum += ek
            n += 1
    if n:
        lines.append(
            f"| **mean ({n})** | **{f_sum / n:.1f}** | **{c_sum / n:.1f}** | "
            f"**{e_sum / n:.1f}** | **{(e_sum - c_sum) / n:+.1f}** | **{(e_sum - f_sum) / n:+.1f}** | |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", choices=["original", "relaxed", "both"], default="both")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--skip-current", action="store_true", help="Reuse outputs/hanif/summaries/latest_run.csv for v1")
    args = ap.parse_args()

    names = list(args.instances) if args.instances else list(REPRESENTATIVE)
    if args.representative or not args.instances:
        names = list(REPRESENTATIVE)
    tws = ["original", "relaxed"] if args.tw == "both" else [args.tw]

    jobs = []
    for tw in tws:
        for name in names:
            jobs.append({"name": name, "tw": tw, "budget": args.budget, "solver": "exact"})
            if not args.skip_current:
                jobs.append({"name": name, "tw": tw, "budget": args.budget, "solver": "current"})

    print(f"=== CARE-FI K comparison ({len(jobs)} solves, budget={args.budget}s) ===", flush=True)
    solved: list[dict] = []
    if args.workers <= 1:
        for job in jobs:
            row = _worker(job)
            print(f"  [{row['tw']}/{row['solver']}] {row['instance']} K={row['used_k']} feas={row['feasible']}", flush=True)
            solved.append(row)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                row = fut.result()
                print(f"  [{row['tw']}/{row['solver']}] {row['instance']} K={row['used_k']} feas={row['feasible']}", flush=True)
                solved.append(row)

    if args.skip_current:
        prev = HANIF_OUT / "summaries" / "latest_run.csv"
        if prev.is_file():
            with prev.open(encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    tw = "relaxed" if "relax" in str(row.get("tw_variant", "")).lower() else "original"
                    if tw in tws and row.get("instance") in names:
                        solved.append({
                            "instance": row["instance"],
                            "tw": tw,
                            "solver": "current",
                            "feasible": str(row.get("feasible", "")).lower() in {"true", "1", "yes"},
                            "used_k": int(float(row["used_k"])),
                            "distance": float(row.get("distance") or 0),
                            "unserved": 0,
                            "hard_violations": int(float(row.get("hard_violations") or 0)),
                            "runtime_s": float(row.get("runtime_s") or 0),
                            "eliminations": 0,
                            "construction_k": None,
                        })

    for row in solved:
        row["frozen_k"] = _frozen_k(row["instance"], row["tw"])

    out_dir = HANIF_OUT / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "exact_vs_current.csv"
    keys = ["instance", "tw", "solver", "feasible", "used_k", "frozen_k", "distance",
            "unserved", "hard_violations", "runtime_s", "eliminations", "construction_k"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(solved, key=lambda r: (r["tw"], r["instance"], r["solver"])))

    md = [
        "# Exact CARE-FI (v2) vs current (v1) vs frozen archive",
        "",
        "Frozen 120-instance cite remains `outputs/classical/` (Original mean K 78.82, Relaxed 65.20).",
        "v1 is `src/hanif/current/`. v2 is the paper method in `src/hanif/`.",
        f"Budget {args.budget:.0f}s per solve. Stage 1 only (K comparison).",
        "",
    ]
    for tw in tws:
        md.append(_md_table(solved, tw))
    md_path = out_dir / "exact_vs_current.md"
    md_path.write_text("\n".join(md).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {md_path}", flush=True)
    print(md_path.read_text(encoding="utf-8"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
