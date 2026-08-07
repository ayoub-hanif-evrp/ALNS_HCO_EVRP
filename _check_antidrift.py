from pathlib import Path
import json
from collections import defaultdict

p = Path("outputs/runs/run_20260729_140606/logs/train_log.jsonl")
print("mtime", p.stat().st_mtime, "bytes", p.stat().st_size)
by = defaultdict(list)
for line in p.read_text(encoding="utf-8").splitlines():
    r = json.loads(line)
    c = int(r.get("cycle", -1))
    if 1 <= c <= 8:
        by[c].append(r)
for c in sorted(by):
    xs = by[c]
    srv = sum(x.get("served_rate", 0) for x in xs) / len(xs)
    feas = sum(x.get("feasibility_rate", 0) for x in xs) / len(xs)
    j = sum(x.get("final_objective", 0) for x in xs) / len(xs)
    print(
        f"c{c}: n={len(xs)} served={srv:.4f} feas={feas:.4f} "
        f"J={j:.1f} last_ep={xs[-1].get('epoch')}"
    )
