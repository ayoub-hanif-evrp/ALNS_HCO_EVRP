"""Combine type-1 and type-2 v3 PR tables; compare to the first frozen PR run."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schneider.evaluation.schneider_bks import summarize  # noqa: E402


def _load_csv(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
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
    v2 = ROOT / "outputs/schneider_v3/pr"
    v1 = ROOT / "outputs/schneider_v3_type1/pr"
    old = ROOT / "outputs/schneider/pr"
    dest = ROOT / "outputs/schneider_v3_all/pr"
    (dest / "routes").mkdir(parents=True, exist_ok=True)

    rows: dict[str, dict] = {}
    for folder in (v2, v1):
        csv_path = folder / "results.csv"
        if csv_path.is_file():
            rows.update(_load_csv(csv_path))
        src = folder / "routes"
        if src.is_dir():
            for path in src.glob("*.json"):
                shutil.copy2(path, dest / "routes" / path.name)

    old_rows = _load_csv(old / "results.csv") if (old / "results.csv").is_file() else {}
    combined = [_norm(r) for r in sorted(rows.values(), key=lambda r: r["instance"])]
    summary = summarize(combined)
    summary["n"] = len(combined)
    print("=== v3 combined (type1+type2) ===")
    print(json.dumps({k: summary[k] for k in summary if k != "citation"}, indent=2))

    print("\n=== vs first outputs/schneider/pr ===")
    better = worse = same = feas_gain = 0
    for name, raw in sorted(rows.items()):
        now = _norm(raw)
        prev = old_rows.get(name)
        if not prev:
            continue
        was_feas = str(prev.get("feasible", "")).lower() in ("true", "1")
        if now["feasible"] and not was_feas:
            feas_gain += 1
            print(f"  FEAS now {name} K={now['used_k']}")
            continue
        if not now["feasible"]:
            continue
        old_k = int(float(prev["used_k"]))
        old_d = float(prev["distance"])
        if now["used_k"] < old_k or (now["used_k"] == old_k and now["distance"] + 1e-6 < old_d):
            better += 1
            print(
                f"  better {name} K {old_k}->{now['used_k']}  "
                f"D {old_d:.1f}->{now['distance']:.1f}  dK={now['delta_k']}"
            )
        elif now["used_k"] > old_k:
            worse += 1
            print(f"  WORSE {name} K {old_k}->{now['used_k']}")
        else:
            same += 1
    print(f"improved={better} sameK={same} worse={worse} new_feasible={feas_gain}")

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
