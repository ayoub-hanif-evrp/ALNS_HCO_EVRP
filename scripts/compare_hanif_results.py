"""Compare a candidate Hanif run with an immutable frozen incumbent.

The comparison is paired by instance and refuses to mix evaluation protocols
when both summaries declare one.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hanif.evaluation.hanif_protocols import completeness_report, protocols_compatible


def _rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["used_k"] = int(float(row["used_k"]))
        row["runtime_s"] = float(row.get("runtime_s") or 0.0)
        row["waiting_time"] = float(row.get("waiting_time") or 0.0)
        row["feasible"] = str(row["feasible"]).strip().lower() in {"1", "true", "yes"}
    return rows


def _protocol(root: Path) -> str | None:
    path = root / "summary_both.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    protocol = data.get("evaluation_protocol")
    if isinstance(protocol, dict):
        return str(protocol.get("name") or "") or None
    return str(protocol) if protocol else None


def compare(candidate: Path, frozen: Path) -> dict[str, Any]:
    cand_protocol, frozen_protocol = _protocol(candidate), _protocol(frozen)
    if not protocols_compatible(cand_protocol, frozen_protocol):
        raise ValueError(
            f"Protocol mismatch: candidate={cand_protocol}, frozen={frozen_protocol}"
        )

    report: dict[str, Any] = {
        "candidate": candidate.as_posix(),
        "frozen": frozen.as_posix(),
        "evaluation_protocol": cand_protocol or frozen_protocol or "EXTENDED",
        "complete": True,
        "variants": {},
    }
    paired: dict[str, dict[str, int]] = {}
    for variant, sub in (("A", "original_tw"), ("B", "relaxed_tw")):
        cand_rows = _rows(candidate / sub / "results.csv")
        base_rows = {r["instance"]: r for r in _rows(frozen / sub / "results.csv")}
        deltas: list[int] = []
        family_delta: dict[str, list[int]] = defaultdict(list)
        wins = ties = losses = feasible = 0
        for row in cand_rows:
            base = base_rows.get(row["instance"])
            if base is None:
                raise ValueError(f"Frozen result missing {variant}/{row['instance']}")
            if row["feasible"]:
                feasible += 1
            delta = int(row["used_k"]) - int(base["used_k"])
            deltas.append(delta)
            family_delta[row["instance"].split("_")[0]].append(delta)
            wins += delta < 0
            ties += delta == 0
            losses += delta > 0
            paired.setdefault(row["instance"], {})[variant] = int(row["used_k"])

        completeness = completeness_report(
            expected_names=[r["instance"] for r in base_rows.values()],
            got_names=[r["instance"] for r in cand_rows],
        )
        variant_complete = completeness["complete"] and len(cand_rows) == len(base_rows)
        report["complete"] = report["complete"] and variant_complete
        n = len(cand_rows)
        stats = {
            "n": n,
            "expected_n": len(base_rows),
            "complete": variant_complete,
            "completeness": completeness,
            "n_feasible": feasible,
        }
        if variant_complete and n:
            stats.update({
                "candidate_mean_k": sum(r["used_k"] for r in cand_rows) / n,
                "frozen_mean_k": sum(base_rows[r["instance"]]["used_k"] for r in cand_rows) / n,
                "mean_delta_k": sum(deltas) / n,
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mean_runtime_s": sum(r["runtime_s"] for r in cand_rows) / n,
                "mean_waiting_time": sum(r["waiting_time"] for r in cand_rows) / n,
                "family_mean_delta_k": {
                    family: sum(values) / len(values)
                    for family, values in sorted(family_delta.items())
                },
            })
        else:
            stats["partial_statistics_do_not_cite"] = {
                "n": n,
                "candidate_mean_k": sum(r["used_k"] for r in cand_rows) / max(n, 1),
                "wins": wins,
                "ties": ties,
                "losses": losses,
            }
        report["variants"][variant] = stats

    b_worse = [
        {"instance": name, "k_a": vals["A"], "k_b": vals["B"]}
        for name, vals in sorted(paired.items())
        if "A" in vals and "B" in vals and vals["B"] > vals["A"]
    ]
    report["b_worse_than_a"] = b_worse
    report["b_worse_than_a_count"] = len(b_worse)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", help="Candidate run root containing original_tw/relaxed_tw")
    ap.add_argument(
        "--frozen",
        default="outputs/hanif/classical",
        help="Immutable control root",
    )
    ap.add_argument("--out", default=None, help="Optional JSON output path")
    args = ap.parse_args()
    candidate = (ROOT / args.candidate).resolve()
    frozen = (ROOT / args.frozen).resolve()
    report = compare(candidate, frozen)
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        out = (ROOT / args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
