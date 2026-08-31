"""Run CARE-FI on the Hanif 1000-customer set and write a compact paper archive.

  python scripts/run_hanif_carefi.py --replace --workers 8 --budget 180
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hanif.loader import FAMILIES, REPRESENTATIVE, list_instances, load_hanif  # noqa: E402
from src.hanif.paper_figures import generate_all  # noqa: E402
from src.hanif.paths import HANIF_OUT  # noqa: E402
from src.hanif.solver import solve_carefi  # noqa: E402

MAP_INSTANCES = {"C1_10_1", "R1_10_1"}
CSV_FIELDS = [
    "instance", "family", "tw", "feasible", "served", "unserved", "hard_violations",
    "k", "distance", "duration", "charging_visits", "charged_energy",
    "charging_time", "charging_expense", "station_waiting", "customer_waiting",
    "k_s2", "distance_s2", "duration_s2", "charging_visits_s2", "charged_energy_s2",
    "charging_time_s2", "charging_expense_s2", "station_waiting_s2", "customer_waiting_s2",
    "phi", "runtime_s", "eliminations",
]


def _family(name: str) -> str:
    return name.split("_")[0]


def _wait(metrics) -> float:
    return float(sum(getattr(r, "customer_waiting", 0.0) or 0.0 for r in (metrics.routes or [])))


def _map_payload(instance, routes) -> dict:
    nodes = []
    for n in instance.nodes:
        kind = "depot" if n.is_depot else "station" if n.is_charging_station else "customer"
        nodes.append((float(n.x), float(n.y), kind))
    paths = []
    for seq, _labs in routes:
        if not any(instance.nodes[i].is_customer for i in seq):
            continue
        paths.append([(float(instance.nodes[i].x), float(instance.nodes[i].y)) for i in seq])
    return {"nodes": nodes, "paths": paths}


def _metrics(m) -> dict:
    return {
        "feasible": bool(m.feasible),
        "served": int(len(m.customers)),
        "unserved": int(m.unserved),
        "hard_violations": int(m.hard_violations),
        "k": int(m.used_k),
        "distance": float(m.distance),
        "duration": float(m.duration),
        "charging_visits": int(m.charging_visits),
        "charged_energy": float(m.charged_energy),
        "charging_time": float(m.charging_time),
        "charging_expense": float(m.charging_expense),
        "station_waiting": float(m.station_waiting),
        "customer_waiting": _wait(m),
    }


def _worker(job: dict) -> dict:
    inst = load_hanif(job["name"], job["tw"])
    s1, s2 = solve_carefi(inst, budget_s=job["budget"], stage2=True)
    a = _metrics(s1.metrics)
    b = _metrics(s2.metrics) if s2 is not None else {}
    row = {
        "instance": job["name"],
        "family": _family(job["name"]),
        "tw": job["tw"],
        "feasible": bool(a["feasible"] and (not b or b["feasible"])),
        "served": a["served"],
        "unserved": a["unserved"],
        "hard_violations": a["hard_violations"],
        "k": a["k"],
        "distance": a["distance"],
        "duration": a["duration"],
        "charging_visits": a["charging_visits"],
        "charged_energy": a["charged_energy"],
        "charging_time": a["charging_time"],
        "charging_expense": a["charging_expense"],
        "station_waiting": a["station_waiting"],
        "customer_waiting": a["customer_waiting"],
        "k_s2": b.get("k"),
        "distance_s2": b.get("distance"),
        "duration_s2": b.get("duration"),
        "charging_visits_s2": b.get("charging_visits"),
        "charged_energy_s2": b.get("charged_energy"),
        "charging_time_s2": b.get("charging_time"),
        "charging_expense_s2": b.get("charging_expense"),
        "station_waiting_s2": b.get("station_waiting"),
        "customer_waiting_s2": b.get("customer_waiting"),
        "phi": None if s2 is None else (s2.stages or {}).get("phi_proposed"),
        "runtime_s": float(s1.runtime_s) + float(s2.runtime_s if s2 else 0.0),
        "eliminations": int((s1.stages.get("fleet_reduction") or {}).get("eliminations") or 0),
    }
    if job["name"] in MAP_INSTANCES and job["tw"] == "original" and s2 is not None:
        row["_map"] = _map_payload(inst, s2.routes)
    print(
        f"  [{job['tw']}] {job['name']} feas={row['feasible']} "
        f"K={row['k']} D={row['distance']:.0f} t={row['runtime_s']:.1f}s",
        flush=True,
    )
    return row


def _mean(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return sum(vals) / len(vals) if vals else None


def _summary(rows: list[dict]) -> dict:
    out: dict = {"n": len(rows), "feasible": sum(1 for r in rows if r["feasible"])}
    for tw in ("original", "relaxed"):
        sub = [r for r in rows if r["tw"] == tw]
        out[tw] = {
            "n": len(sub),
            "feasible": sum(1 for r in sub if r["feasible"]),
            "mean_k": _mean(sub, "k"),
            "mean_distance": _mean(sub, "distance"),
            "mean_duration": _mean(sub, "duration_s2") or _mean(sub, "duration"),
            "mean_charging_expense_s1": _mean(sub, "charging_expense"),
            "mean_charging_expense_s2": _mean(sub, "charging_expense_s2"),
            "mean_charging_time_s1": _mean(sub, "charging_time"),
            "mean_charging_time_s2": _mean(sub, "charging_time_s2"),
            "mean_station_waiting_s1": _mean(sub, "station_waiting"),
            "mean_station_waiting_s2": _mean(sub, "station_waiting_s2"),
            "mean_phi": _mean(sub, "phi"),
            "mean_runtime_s": _mean(sub, "runtime_s"),
            "by_family": {
                fam: {"n": sum(1 for r in sub if r["family"] == fam), "mean_k": _mean([r for r in sub if r["family"] == fam], "k")}
                for fam in FAMILIES
            },
        }
    return out


def _clear(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tw", choices=["original", "relaxed", "both"], default="both")
    ap.add_argument("--budget", type=float, default=180.0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(HANIF_OUT))
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--representative", action="store_true")
    ap.add_argument("--replace", action="store_true", help="Delete existing outputs/hanif contents first")
    args = ap.parse_args()

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    if out.resolve() == HANIF_OUT.resolve() and not args.replace:
        raise SystemExit("Pass --replace to write the paper archive under outputs/hanif/")
    if args.replace and out.resolve() == HANIF_OUT.resolve():
        _clear(out)
    else:
        out.mkdir(parents=True, exist_ok=True)

    names = list(args.instances) if args.instances else list(list_instances() or REPRESENTATIVE)
    if args.representative:
        names = list(REPRESENTATIVE)
    tws = ["original", "relaxed"] if args.tw == "both" else [args.tw]
    jobs = [{"name": n, "tw": tw, "budget": args.budget} for tw in tws for n in names]

    print(f"=== CARE-FI Hanif ({len(jobs)} jobs, budget={args.budget}s, workers={args.workers}) ===", flush=True)
    rows: list[dict] = []
    if args.workers <= 1:
        for job in jobs:
            rows.append(_worker(job))
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futs = {pool.submit(_worker, job): job for job in jobs}
            for fut in as_completed(futs):
                rows.append(fut.result())
    rows.sort(key=lambda r: (r["tw"], r["instance"]))

    maps = {}
    for r in rows:
        payload = r.pop("_map", None)
        if payload is not None:
            maps[r["instance"]] = payload

    with (out / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    summary = _summary(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    fig_dir = out / "figures"
    saved = generate_all(rows, maps, fig_dir)
    (out / "README.md").write_text(
        "# Hanif CARE-FI results\n\n"
        f"- Instances: {summary['n']} ({summary['original']['n']} original TW + {summary['relaxed']['n']} relaxed TW)\n"
        f"- Feasible: {summary['feasible']}/{summary['n']}\n"
        f"- Mean K original TW: {summary['original']['mean_k']:.2f}\n"
        f"- Mean K relaxed TW: {summary['relaxed']['mean_k']:.2f}\n\n"
        "Files: `results.csv`, `summary.json`, and 10 PNG figures in `figures/`.\n",
        encoding="utf-8",
    )
    print(json.dumps({k: summary[k] for k in ("n", "feasible", "original", "relaxed") if k in summary}, indent=2), flush=True)
    print(f"wrote {out / 'results.csv'} and {len(saved)} figures", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
