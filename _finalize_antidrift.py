"""Wait for anti-drift early-stop, then write final best-vs-last summary."""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path


def summarize(run: Path) -> dict:
    log = run / "logs" / "train_log.jsonl"
    by: dict[int, list] = defaultdict(list)
    for line in log.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        c = int(r.get("cycle", -1))
        if 1 <= c <= 8:
            by[c].append(r)
    rows = []
    for c in sorted(by):
        xs = by[c]
        rows.append({
            "cycle": c,
            "n": len(xs),
            "served": sum(x.get("served_rate", 0) for x in xs) / len(xs),
            "feas": sum(x.get("feasibility_rate", 0) for x in xs) / len(xs),
            "J": sum(x.get("final_objective", 0) for x in xs) / len(xs),
            "reward": sum(x.get("reward", 0) for x in xs) / len(xs),
        })
    best = None
    best_key = None
    for r in rows:
        key = (-r["feas"], -r["served"], r["J"])
        if best_key is None or key < best_key:
            best_key = key
            best = r
    last = rows[-1] if rows else None
    out = {
        "cycles": rows,
        "best_cycle": best,
        "last_cycle": last,
        "success_bar_train": {
            "served_ge_0_98": bool(best and best["served"] >= 0.98),
            "feas_ge_0_55": bool(best and best["feas"] >= 0.55),
        },
        "checkpoints": {
            "best": str(run / "checkpoints" / "best.pt"),
            "last": str(run / "checkpoints" / "last.pt"),
            "best_mtime": (run / "checkpoints" / "best.pt").stat().st_mtime
            if (run / "checkpoints" / "best.pt").is_file() else None,
            "last_mtime": (run / "checkpoints" / "last.pt").stat().st_mtime
            if (run / "checkpoints" / "last.pt").is_file() else None,
        },
    }
    path = run / "evaluation" / "antidrift_cycle_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    md = run / "evaluation" / "antidrift_cycle_summary.md"
    lines = [
        "# Anti-drift RL cycle summary",
        "",
        "| Cycle | n | Served | Feas | Mean J | Reward |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        mark = " *" if best and r["cycle"] == best["cycle"] else ""
        lines.append(
            f"| {r['cycle']}{mark} | {r['n']} | {r['served']:.4f} | {r['feas']:.4f} | "
            f"{r['J']:.1f} | {r['reward']:.3f} |"
        )
    lines.append("")
    if best and last:
        lines.append(
            f"Best hierarchical cycle **{best['cycle']}** vs last cycle **{last['cycle']}**: "
            f"served Δ={best['served']-last['served']:+.4f}, "
            f"feas Δ={best['feas']-last['feas']:+.4f}, "
            f"J Δ={best['J']-last['J']:+.1f}."
        )
        lines.append("")
        lines.append(
            f"Train success bar: served≥0.98 → {best['served']>=0.98}, "
            f"feas≥0.55 → {best['feas']>=0.55}."
        )
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def main() -> None:
    run = Path("outputs/runs/run_20260729_140606")
    patience = 2
    for i in range(40):
        out = summarize(run)
        cycles = out["cycles"]
        complete = [r for r in cycles if r["n"] >= 30]
        print(
            f"[{i}] complete_cycles={[r['cycle'] for r in complete]} "
            f"best={None if not out['best_cycle'] else out['best_cycle']['cycle']} "
            f"last_partial={None if not cycles else (cycles[-1]['cycle'], cycles[-1]['n'])}",
            flush=True,
        )
        if len(complete) >= 2:
            best_c = out["best_cycle"]["cycle"]
            # Count trailing complete cycles that did not improve on best.
            trailing = 0
            for r in reversed(complete):
                if r["cycle"] == best_c:
                    break
                trailing += 1
            # Early-stop fires after patience non-improving complete cycles.
            if trailing >= patience and complete[-1]["n"] >= 30:
                print("Early-stop condition met (or passed).", flush=True)
                break
            # Also stop if RL summary exists.
            if (run / "rl" / "rl_summary.json").is_file():
                # Only if updated after anti-drift start (best.pt today).
                print("rl_summary present", flush=True)
                break
        time.sleep(60)
    out = summarize(run)
    print(json.dumps(out["best_cycle"], indent=2))
    print(json.dumps(out["last_cycle"], indent=2))


if __name__ == "__main__":
    main()
