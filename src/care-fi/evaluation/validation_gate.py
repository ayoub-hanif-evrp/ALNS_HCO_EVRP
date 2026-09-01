"""Scientific validation gate for Classical vs Stage-2 station refinement.

The automatic PASS must match the scientific decision. A 0.5% blip on a single
station metric (with waiting/price worse) is not a pass.
"""
from __future__ import annotations

import csv
from pathlib import Path


IMPROVE_PCT = 2.0
WORSEN_PCT = 1.0
STATION_KEYS = (
    "station_waiting_time",
    "charging_time",
    "expected_charging_cost",
)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _mean(rows: list[dict], key: str) -> float:
    vals = [_f(r, key) for r in rows]
    return sum(vals) / len(vals) if vals else float("nan")


def _pct(classical: float, proposed: float) -> float:
    if abs(classical) < 1e-9:
        return 0.0 if abs(proposed) < 1e-9 else 100.0
    return 100.0 * (proposed - classical) / classical


def evaluate_gate(classical_dir: Path, proposed_dir: Path) -> dict:
    """Return a gate report whose ``pass`` equals the scientific decision."""
    report: dict = {
        "tw": {},
        "pass": True,
        "scientific_pass": True,
        "reasons": [],
        "improve_threshold_pct": IMPROVE_PCT,
        "worsen_threshold_pct": WORSEN_PCT,
        "rule": (
            "100% feasible and served; K_proposed <= K_classical on every instance; "
            f"on each TW at least one of wait/charging-time/expense improves by >={IMPROVE_PCT:g}%; "
            f"none of those three worsens by >={WORSEN_PCT:g}%."
        ),
    }
    for tw in ("original_tw", "relaxed_tw"):
        c_rows = _read_csv(Path(classical_dir) / tw / "results.csv")
        p_rows = _read_csv(Path(proposed_dir) / tw / "results.csv")
        by_c = {r["instance"]: r for r in c_rows}
        paired = [(by_c[r["instance"]], r) for r in p_rows if r.get("instance") in by_c]
        n = len(paired)
        feas_c = sum(1 for a, _ in paired if str(a.get("feasible")).lower() == "true")
        feas_p = sum(1 for _, b in paired if str(b.get("feasible")).lower() == "true")
        served_p = all(
            _f(b, "served_ratio") >= 1.0 - 1e-9 or _f(b, "served") >= _f(b, "n_customers")
            for _, b in paired
        )
        k_worse = [
            str(b.get("instance"))
            for a, b in paired
            if _f(b, "used_k") > _f(a, "used_k") + 1e-9
        ]
        deltas = {key: _pct(_mean([a for a, _ in paired], key), _mean([b for _, b in paired], key)) for key in STATION_KEYS}
        d_k = _mean([b for _, b in paired], "used_k") - _mean([a for a, _ in paired], "used_k")
        improved = [key for key, d in deltas.items() if d <= -IMPROVE_PCT]
        worsened = [key for key, d in deltas.items() if d >= WORSEN_PCT]
        station_ok = bool(improved) and not worsened
        block = {
            "n": n,
            "feasible_classical": feas_c,
            "feasible_proposed": feas_p,
            "served_proposed_all": bool(served_p),
            "mean_k_classical": _mean([a for a, _ in paired], "used_k"),
            "mean_k_proposed": _mean([b for _, b in paired], "used_k"),
            "Delta_K": d_k,
            "Delta_distance_%": _pct(
                _mean([a for a, _ in paired], "distance"),
                _mean([b for _, b in paired], "distance"),
            ),
            "Delta_station_wait_%": deltas["station_waiting_time"],
            "Delta_charging_time_%": deltas["charging_time"],
            "Delta_charging_expense_%": deltas["expected_charging_cost"],
            "k_worse_instances": k_worse,
            "station_metrics_improved": improved,
            "station_metrics_worsened": worsened,
            "station_related_improved": station_ok,
        }
        report["tw"][tw] = block
        if n == 0 or feas_p < n or not served_p:
            report["pass"] = False
            report["reasons"].append(f"{tw}: feasibility or service not preserved")
        if k_worse:
            report["pass"] = False
            report["reasons"].append(
                f"{tw}: fleet increased on {len(k_worse)} instance(s): {k_worse}"
            )
        if d_k > 1e-9:
            report["pass"] = False
            report["reasons"].append(f"{tw}: mean fleet worse (Delta_K={d_k:.3f})")
        if not station_ok:
            report["pass"] = False
            if not improved:
                report["reasons"].append(
                    f"{tw}: no station metric improved by >={IMPROVE_PCT:g}% "
                    f"(wait {deltas['station_waiting_time']:+.2f}%, "
                    f"charge-time {deltas['charging_time']:+.2f}%, "
                    f"expense {deltas['expected_charging_cost']:+.2f}%)"
                )
            if worsened:
                report["reasons"].append(
                    f"{tw}: station metric(s) worsened by >={WORSEN_PCT:g}%: {worsened}"
                )
    report["scientific_pass"] = bool(report["pass"])
    report["overall_Delta_K"] = (
        (report["tw"]["original_tw"]["Delta_K"] + report["tw"]["relaxed_tw"]["Delta_K"]) / 2.0
        if report["tw"] else 0.0
    )
    return report


def decision_payload(gate: dict, *, out_dir: str) -> dict:
    ok = bool(gate.get("scientific_pass") and gate.get("pass"))
    if ok:
        decision = "PASS — station refinement improved operational charging burden without increasing fleet. Eligible for the full 60+60."
    else:
        decision = "PARTIAL — do not run the full 60+60; keep Classical Fleet-Distance as the main solver."
    return {
        "automatic_threshold_pass": ok,
        "scientific_pass": ok,
        "decision": decision,
        "rationale": "; ".join(gate.get("reasons") or []) or "Station refinement meets the scientific gate.",
        "full_120_not_run": True,
        "risk_aware_not_run": True,
        "legacy_best": "removed (old wait_mu-as-minutes archive; do not cite 80.07/65.15)",
        "new_results": str(out_dir).replace("\\", "/"),
        "gate": {k: gate[k] for k in gate if k != "tw"} | {"tw": gate.get("tw")},
    }
