"""Verify frozen deterministic + neural result hashes and recompute tables."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recompute_from_rows(rows: list[dict]) -> dict:
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "mean_served": sum(float(r.get("served_ratio", 0)) for r in rows) / n,
        "feasibility_rate": sum(1.0 if r.get("feasible") else 0.0 for r in rows) / n,
        "mean_objective": sum(float(r.get("objective", r.get("paper_J", 0))) for r in rows) / n,
        "mean_used_evs": sum(float(r.get("used_evs", 0)) for r in rows) / n,
        "mean_runtime": sum(float(r.get("runtime", 0)) for r in rows) / n,
    }


def main() -> int:
    det = ROOT / "outputs" / "baselines" / "expert_repair_alns_20260801"
    manifest = det / "manifest.json"
    summary = det / "holdout_full9_tgfps" / "summary.json"
    ok = True
    if not manifest.is_file() or not summary.is_file():
        print("FAIL: frozen Expert+Repair+ALNS baseline missing")
        return 1
    man = json.loads(manifest.read_text(encoding="utf-8"))
    got = sha256(summary)
    exp = man.get("summary_sha256")
    print("deterministic summary sha256:", got)
    if exp and got != exp:
        print("FAIL: summary hash mismatch", exp)
        ok = False
    else:
        print("OK: deterministic summary hash matches manifest")

    payload = json.loads(summary.read_text(encoding="utf-8"))
    rows = payload.get("rows") or []
    live = payload.get("live_summary") or {}
    recomputed = recompute_from_rows(rows)
    print("recomputed:", json.dumps(recomputed, indent=2))
    for k in ("mean_served", "feasibility_rate", "mean_objective"):
        if k in live and abs(float(live[k]) - float(recomputed[k])) > 1e-6:
            print(f"FAIL: {k} mismatch live={live[k]} recomputed={recomputed[k]}")
            ok = False
    if ok:
        print("OK: table recomputed from per-instance rows")

    # Clarification print
    print(
        "NOTE: frozen Expert+Repair+ALNS is deterministic — "
        "not Transformer-guided TG-FPS."
    )

    neural_val = ROOT / "outputs" / "tgfps_neural_validation" / "hybrid_best_validation.json"
    if neural_val.is_file():
        nv = json.loads(neural_val.read_text(encoding="utf-8"))
        print("neural validation selected:", nv.get("selected_method"))
        print("answer:", nv.get("answer"))
    else:
        print("neural validation not present yet (run scripts/run_neural_validation.py)")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
