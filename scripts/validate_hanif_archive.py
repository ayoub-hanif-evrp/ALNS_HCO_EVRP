"""Exact-replay saved Hanif routes under EXTENDED and/or NO_WAIT_REPLAY.

NO_WAIT_REPLAY keeps the saved customer and charging sequences and only
disables station waits. It does not re-optimize.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hanif.data.instance_loader import load_instance
from src.hanif.env.cost import CostWeights
from src.hanif.evaluation.hanif_protocols import (
    PROTOCOL_EXTENDED,
    PROTOCOL_NO_WAIT_REPLAY,
    instance_without_station_wait,
)
from src.hanif.search.exact_checks import exact_accept_solution
from src.hanif.utils.config import load_config

VARIANT_DIRS = (
    ("A", "original_tw", "converted_hanif_original_tw"),
    ("B", "relaxed_tw", "converted_hanif_relaxed_tw"),
)


def _load_routes(payload: dict) -> list:
    return [
        (list(r["nodes"]), list(r.get("labels") or [None] * len(r["nodes"])))
        for r in payload["routes"]
    ]


def _replay_one(instance, routes, cfg, weights, claimed_k: int) -> dict:
    exact, _replay, missing = exact_accept_solution(
        instance,
        routes,
        config=cfg,
        weights=weights,
        m=int(claimed_k),
        fail_loud=False,
    )
    ok = bool(exact.ok and exact.replay_feasible and not missing)
    return {
        "ok": ok,
        "feasible": bool(exact.replay_feasible and not missing and exact.hard_violations == 0),
        "used_k": int(exact.used_vehicles),
        "served": len(exact.served_customers),
        "missing_customers": len(missing),
        "distance": float(exact.distance),
        "duration": float(exact.duration),
        "waiting_time": float(exact.waiting_time),
        "charging_visits": int(exact.charging_visits),
        "errors": list(exact.errors),
    }


def _mean(rows: list[dict], key: str):
    vals = [float(r[key]) for r in rows]
    return sum(vals) / len(vals) if vals else None


def validate_archive(
    archive: Path,
    config_path: Path,
    *,
    protocols: list[str] | None = None,
) -> dict:
    cfg = dict(load_config(str(config_path)))
    weights = CostWeights.from_eval_config(cfg)
    wanted = [p.upper() for p in (protocols or ["EXTENDED", "NO_WAIT_REPLAY"])]
    report = {
        "archive": archive.as_posix(),
        "ok": True,
        "protocols": {},
        "protocol_definitions": {
            "EXTENDED": PROTOCOL_EXTENDED,
            "NO_WAIT_REPLAY": PROTOCOL_NO_WAIT_REPLAY,
        },
    }
    for protocol in wanted:
        proto_meta = PROTOCOL_EXTENDED if protocol == "EXTENDED" else PROTOCOL_NO_WAIT_REPLAY
        variants = {}
        proto_ok = True
        for variant, sub, data_sub in VARIANT_DIRS:
            route_dir = archive / sub / "routes"
            rows = []
            if not route_dir.is_dir():
                variants[variant] = {"n": 0, "n_valid": 0, "n_feasible": 0, "rows": []}
                continue
            for path in sorted(route_dir.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                instance = load_instance(ROOT / "dataset" / "Hanif" / data_sub / path.stem)
                if protocol == "NO_WAIT_REPLAY":
                    instance = instance_without_station_wait(instance)
                routes = _load_routes(payload)
                row = _replay_one(instance, routes, cfg, weights, int(payload["used_k"]))
                row["instance"] = path.stem
                row["claimed_k"] = int(payload["used_k"])
                rows.append(row)
                proto_ok = proto_ok and bool(row["ok"])
            variants[variant] = {
                "n": len(rows),
                "n_valid": sum(1 for row in rows if row["ok"]),
                "n_feasible": sum(1 for row in rows if row["feasible"]),
                "mean_k": _mean(rows, "used_k"),
                "mean_distance": _mean(rows, "distance"),
                "mean_duration": _mean(rows, "duration"),
                "mean_waiting_time": _mean(rows, "waiting_time"),
                "rows": rows,
            }
        report["protocols"][protocol] = {
            "definition": proto_meta,
            "ok": proto_ok,
            "variants": variants,
        }
        report["ok"] = report["ok"] and proto_ok
    if "EXTENDED" in report["protocols"] and "NO_WAIT_REPLAY" in report["protocols"]:
        comparison = {}
        for variant, _sub, _data in VARIANT_DIRS:
            ext = {
                r["instance"]: r
                for r in report["protocols"]["EXTENDED"]["variants"].get(variant, {}).get("rows", [])
            }
            now = {
                r["instance"]: r
                for r in report["protocols"]["NO_WAIT_REPLAY"]["variants"].get(variant, {}).get("rows", [])
            }
            rows = []
            for name in sorted(set(ext) & set(now)):
                a, b = ext[name], now[name]
                rows.append({
                    "instance": name,
                    "extended_feasible": a["feasible"],
                    "no_wait_feasible": b["feasible"],
                    "extended_k": a["used_k"],
                    "no_wait_k": b["used_k"],
                    "extended_distance": a["distance"],
                    "no_wait_distance": b["distance"],
                    "extended_duration": a["duration"],
                    "no_wait_duration": b["duration"],
                    "extended_waiting_time": a["waiting_time"],
                    "no_wait_waiting_time": b["waiting_time"],
                    "k_unchanged": a["used_k"] == b["used_k"],
                })
            comparison[variant] = {
                "n": len(rows),
                "k_unchanged": sum(1 for r in rows if r["k_unchanged"]),
                "mean_wait_extended": _mean(rows, "extended_waiting_time"),
                "mean_wait_no_wait": _mean(rows, "no_wait_waiting_time"),
                "rows": rows,
            }
        report["semantic_comparison"] = comparison
    # Backward-compatible top-level variants = EXTENDED (or the only protocol).
    primary = "EXTENDED" if "EXTENDED" in report["protocols"] else wanted[0]
    report["variants"] = report["protocols"][primary]["variants"]
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archive")
    ap.add_argument("--config", default="configs/hanif_fps.yaml")
    ap.add_argument(
        "--protocol",
        default="both",
        choices=["extended", "no-wait", "both"],
        help="EXTENDED, NO_WAIT_REPLAY, or both",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    mapping = {
        "extended": ["EXTENDED"],
        "no-wait": ["NO_WAIT_REPLAY"],
        "both": ["EXTENDED", "NO_WAIT_REPLAY"],
    }
    report = validate_archive(
        (ROOT / args.archive).resolve(),
        (ROOT / args.config).resolve(),
        protocols=mapping[args.protocol],
    )
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        path = (ROOT / args.out).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
