"""Investigate Schneider R1/RC1 single-customer unserved (no algorithm changes)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import reviewer_root, write_json  # noqa: E402
from src.hanif.benchmarks.greedy import solve_greedy  # noqa: E402
from src.hanif.data.instance_loader import load_instance  # noqa: E402
from src.hanif.data.schneider_txt_convert import (  # noqa: E402
    DEFAULT_CONVERTED,
    DEFAULT_RAW,
    convert_one_txt,
    parse_schneider_txt,
)
from src.hanif.replay import replay_solution  # noqa: E402


def missing_customers(inst, routes):
    covered = set()
    for item in routes:
        nodes = item[0] if isinstance(item, (tuple, list)) and item and isinstance(item[0], (list, tuple)) else item
        for n in nodes:
            if int(n) in set(map(int, inst.customer_indices)):
                covered.add(int(n))
    return sorted(set(map(int, inst.customer_indices)) - covered)


def try_solo_route(inst, cid: int) -> dict:
    """Feasibility of a depot–customer–depot route (full charge at depot)."""
    routes = [([0, cid, 0], [None, None, None])]
    m = replay_solution(inst, routes)
    return {
        "customer": cid,
        "solo_feasible": bool(m.feasible),
        "solo_unserved": int(m.unserved),
        "solo_hard_violations": int(m.hard_violations),
        "solo_distance": float(m.distance),
        "solo_duration": float(m.duration),
    }


def diagnose(name: str) -> dict:
    raw = Path(DEFAULT_RAW) / f"{name}.txt"
    out = Path(DEFAULT_CONVERTED)
    convert_one_txt(raw, out)
    parsed = parse_schneider_txt(raw)
    inst = load_instance(out / name)
    gi = solve_greedy(inst, deadline=10.0)
    miss = missing_customers(inst, gi["routes"])
    details = []
    for cid in miss:
        node = inst.nodes[cid]
        solo = try_solo_route(inst, cid)
        d = float(inst.distance_matrix[0, cid])
        e = float(inst.energy_matrix[0, cid])
        details.append({
            "customer_index": cid,
            "x": node.x, "y": node.y,
            "demand": node.demand,
            "ready": node.ready_time,
            "due": node.due_time,
            "service": node.service_time,
            "dist_depot": d,
            "energy_depot": e,
            "Q": parsed["Q"],
            "v": parsed["v"],
            "travel_time_depot": d / float(parsed["v"]) if parsed["v"] else None,
            "depot_due": parsed["depot"]["due"],
            **solo,
        })
    return {
        "instance": name,
        "gi_feasible": bool(gi["metrics"].feasible),
        "gi_unserved": int(gi["metrics"].unserved),
        "gi_k": int(gi["metrics"].used_k),
        "missing_customers": miss,
        "details": details,
        "interpretation": (
            "If solo_feasible is True but GI/ALNS leave the customer unserved, this is a "
            "search/construction failure under the current operators/time limit — NOT a proof "
            "of mathematical infeasibility of the Schneider instance. "
            "If solo_feasible is False, the unrounded Euclidean conversion and/or TW/battery "
            "semantics make that customer unreachable alone (converter/model interaction)."
        ),
    }


def main() -> int:
    names = ["r101_21", "rc101_21", "r102_21", "rc102_21", "r201_21", "rc201_21"]
    findings = []
    for n in names:
        txt = Path(DEFAULT_RAW) / f"{n}.txt"
        if not txt.is_file():
            continue
        findings.append(diagnose(n))
        print(n, findings[-1]["gi_feasible"], findings[-1]["missing_customers"],
              [(d["customer_index"], d["solo_feasible"]) for d in findings[-1]["details"]], flush=True)
    out = reviewer_root() / "schneider" / "infeasibility_investigation.json"
    write_json(out, findings)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
