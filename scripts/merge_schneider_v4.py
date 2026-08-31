"""Combine v4 closest-25 and v4-rest into one 56-row PR table."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schneider.evaluation.schneider_bks import summarize  # noqa: E402


def _load(folder: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with (folder / "results.csv").open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows[row["instance"]] = row
    return rows


def _norm(row: dict) -> dict:
    out = dict(row)
    out["feasible"] = str(out.get("feasible", "")).lower() in ("true", "1")
    out["used_k"] = int(float(out["used_k"]))
    out["distance"] = float(out["distance"])
    raw_dk = out.get("delta_k")
    out["delta_k"] = int(float(raw_dk)) if raw_dk not in (None, "") else None
    raw_gap = out.get("distance_gap_%")
    out["distance_gap_%"] = None if raw_gap in (None, "") else float(raw_gap)
    return out


def main() -> int:
    folders = [
        ROOT / "outputs/schneider_v4/pr",
        ROOT / "outputs/schneider_v4_rest/pr",
    ]
    prev_path = ROOT / "outputs/schneider_v3_all/pr"
    dest = ROOT / "outputs/schneider_v4_all/pr"
    (dest / "routes").mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    for folder in folders:
        rows.update(_load(folder))
        src = folder / "routes"
        if src.is_dir():
            for path in src.glob("*.json"):
                shutil.copy2(path, dest / "routes" / path.name)

    combined = [_norm(r) for r in sorted(rows.values(), key=lambda r: r["instance"])]
    summary = summarize(combined)
    summary["n"] = len(combined)
    print("=== v4 all 56 ===")
    print(json.dumps({k: summary[k] for k in summary if k != "citation"}, indent=2))

    prev = _load(prev_path) if (prev_path / "results.csv").is_file() else {}
    print("\n=== vs v3_all ===")
    better = worse = same = 0
    for name, raw in sorted(rows.items()):
        now = _norm(raw)
        old = prev.get(name)
        if not old:
            continue
        old_k = int(float(old["used_k"]))
        old_d = float(old["distance"])
        old_dk = int(float(old["delta_k"])) if old.get("delta_k") not in (None, "") else None
        if now["delta_k"] == 0 and old_dk not in (0, None) and now["used_k"] < old_k:
            better += 1
            print(f"  NEW TIE {name} K {old_k}->{now['used_k']}")
        elif now["used_k"] < old_k or (now["used_k"] == old_k and now["distance"] + 1e-6 < old_d):
            better += 1
            print(
                f"  better {name} K {old_k}->{now['used_k']} "
                f"D {old_d:.1f}->{now['distance']:.1f} dK={now['delta_k']}"
            )
        elif now["used_k"] > old_k:
            worse += 1
            print(f"  WORSE {name} K {old_k}->{now['used_k']}")
        else:
            same += 1
    print(f"improved={better} same={same} worse={worse}")

    keys: list[str] = []
    for row in combined:
        for key in row:
            if key not in keys:
                keys.append(key)
    with (dest / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(combined)
    (dest / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("wrote", dest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
