from pathlib import Path
import json
from collections import defaultdict
from datetime import datetime

run = Path("outputs/runs/run_20260729_140606")
p = run / "logs" / "train_log.jsonl"
by = defaultdict(list)
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    c = int(r.get("cycle", -1))
    if 1 <= c <= 8:
        by[c].append(r)

rows = []
for c in sorted(by):
    xs = by[c]
    if len(xs) < 30:
        continue
    rows.append({
        "cycle": c,
        "n": len(xs),
        "served": sum(x.get("served_rate", 0) for x in xs) / len(xs),
        "feas": sum(x.get("feasibility_rate", 0) for x in xs) / len(xs),
        "J": sum(x.get("final_objective", 0) for x in xs) / len(xs),
        "reward": sum(x.get("reward", 0) for x in xs) / len(xs),
    })

best = min(rows, key=lambda r: (-r["feas"], -r["served"], r["J"]))
last = rows[-1]
out = {
    "generated_at": datetime.now().isoformat(timespec="seconds"),
    "cycles": rows,
    "best_cycle": best,
    "last_cycle": last,
    "restored_best_vs_last": {
        "best_cycle": best["cycle"],
        "last_cycle": last["cycle"],
        "served_delta": best["served"] - last["served"],
        "feas_delta": best["feas"] - last["feas"],
        "J_delta": best["J"] - last["J"],
        "reward_delta": best["reward"] - last["reward"],
    },
    "success_bar_train": {
        "served_ge_0_98": best["served"] >= 0.98,
        "feas_ge_0_55": best["feas"] >= 0.55,
        "passed": best["served"] >= 0.98 and best["feas"] >= 0.55,
    },
    "early_stop": {
        "patience": 2,
        "note": "Best hierarchical cycle retained in checkpoints/best.pt; use that (not last) for paper.",
    },
}
(out_dir := run / "evaluation").mkdir(parents=True, exist_ok=True)
(out_dir / "antidrift_cycle_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
lines = [
    "# Anti-drift RL cycle summary",
    "",
    "| Cycle | n | Served | Feas | Mean J | Reward |",
    "|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    mark = " *" if r["cycle"] == best["cycle"] else ""
    lines.append(
        f"| {r['cycle']}{mark} | {r['n']} | {r['served']:.4f} | {r['feas']:.4f} | "
        f"{r['J']:.1f} | {r['reward']:.3f} |"
    )
lines += [
    "",
    f"Best hierarchical cycle **{best['cycle']}** vs last cycle **{last['cycle']}**: "
    f"served Δ={best['served']-last['served']:+.4f}, "
    f"feas Δ={best['feas']-last['feas']:+.4f}, "
    f"J Δ={best['J']-last['J']:+.1f}, "
    f"reward Δ={best['reward']-last['reward']:+.3f}.",
    "",
    f"Train success bar: served≥0.98 → {best['served']>=0.98}, feas≥0.55 → {best['feas']>=0.55}.",
    "",
    "Paper eval must use `checkpoints/best.pt` (or holdout `checkpoints_best`), never `last.pt`.",
]
(out_dir / "antidrift_cycle_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(out["best_cycle"], indent=2))
print(json.dumps(out["restored_best_vs_last"], indent=2))
