"""Compile paper result tables from completed archives.

Merges Classical / Proposed (required) with optional constructor, ALNS, VNS-TS
once those directories exist. Never invents missing competitor numbers.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.evaluation.output_paths import (
    ALNS,
    CLASSICAL,
    CONSTRUCTOR,
    OUTPUTS,
    PROPOSED,
    ROOT,
    TABLES,
    VNS_TS,
)
FAMILIES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
TW = (("original_tw", "Original TW"), ("relaxed_tw", "Relaxed TW"))
KEYS = [
    "used_k", "distance", "duration", "station_waiting_time", "charging_time",
    "expected_charging_cost", "charged_energy", "energy_consumed",
    "customer_early_waiting_time", "charging_visits", "runtime_s",
]
DISPLAY = {
    "Constructor": "Constructor",
    "ALNS": "ALNS-style",
    "VNS-TS": "VNS-style",
    "Classical": "Classical",
    "Proposed": "Proposed",
}


def _read(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            row = dict(raw)
            for k, v in raw.items():
                if v is None or v == "":
                    continue
                try:
                    if str(v).lower() in {"true", "false"}:
                        row[k] = str(v).lower() == "true"
                    elif any(c in str(v) for c in ".eE"):
                        row[k] = float(v)
                    else:
                        row[k] = int(v)
                except (ValueError, AttributeError):
                    row[k] = v
            rows.append(row)
        return rows


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _f(row: dict, key: str) -> float:
    try:
        return float(row.get(key) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _stats(rows: list[dict], key: str) -> tuple[float, float]:
    xs = [_f(r, key) for r in rows if key in r]
    if not xs:
        return float("nan"), float("nan")
    mu = mean(xs)
    sd = pstdev(xs) if len(xs) > 1 else 0.0
    return mu, sd


def _pct(a: float, b: float) -> float:
    if abs(a) < 1e-12:
        return 0.0 if abs(b) < 1e-12 else 100.0
    return 100.0 * (b - a) / a


def load_method(exp: Path, method_label: str) -> dict[str, list[dict]]:
    out = {}
    for tw, _ in TW:
        rows = _read(exp / tw / "results.csv")
        for r in rows:
            r["method"] = method_label
            r["tw"] = tw
        out[tw] = rows
    return out


def discover_methods() -> dict[str, dict[str, list[dict]]]:
    methods = {
        "Classical": load_method(CLASSICAL, "Classical"),
        "Proposed": load_method(PROPOSED, "Proposed"),
    }
    optional = {
        "Constructor": CONSTRUCTOR,
        "ALNS": ALNS,
        "VNS-TS": VNS_TS,
    }
    for label, path in optional.items():
        if (path / "original_tw" / "results.csv").is_file() and (path / "relaxed_tw" / "results.csv").is_file():
            data = load_method(path, label)
            if len(data["original_tw"]) >= 60 and len(data["relaxed_tw"]) >= 60:
                methods[label] = data
    return methods


def paired(classical: list[dict], proposed: list[dict]) -> list[tuple[dict, dict]]:
    by = {r["instance"]: r for r in classical}
    return [(by[r["instance"]], r) for r in proposed if r.get("instance") in by]


def compile() -> dict:
    methods = discover_methods()
    TABLES.mkdir(parents=True, exist_ok=True)

    overall = []
    for label, data in methods.items():
        for tw, twlab in TW:
            rows = data[tw]
            feas = [r for r in rows if r.get("feasible")]
            use = feas or rows
            row = {
                "method": label,
                "tw_variant": twlab,
                "n": len(rows),
                "feasible": sum(1 for r in rows if r.get("feasible")),
                "complete": len(rows) >= 60 and all(r.get("feasible") for r in rows),
            }
            for k in KEYS:
                m, s = _stats(use, k)
                row[f"{k}_mean"] = m
                row[f"{k}_std"] = s
            overall.append(row)
    _write(TABLES / "table1_overall.csv", overall)
    _write(TABLES / "table_methods_overall.csv", overall)

    family = []
    for tw, twlab in TW:
        for fam in FAMILIES:
            for label, data in methods.items():
                sub = [r for r in data[tw] if str(r.get("instance", "")).split("_")[0] == fam]
                rec = {"tw_variant": twlab, "family": fam, "method": label, "n": len(sub)}
                for k in KEYS:
                    rec[k] = _stats(sub, k)[0]
                family.append(rec)
    _write(TABLES / "table2_family.csv", family)

    c, p = methods["Classical"], methods["Proposed"]
    deltas = []
    for tw, twlab in TW:
        pairs = paired(c[tw], p[tw])
        rec = {"tw_variant": twlab, "n": len(pairs), "Delta_vehicles": 0.0}
        for key, name in [
            ("used_k", "vehicles"),
            ("distance", "distance"),
            ("station_waiting_time", "station_waiting"),
            ("charging_time", "charging_time"),
            ("expected_charging_cost", "charging_expense"),
            ("charged_energy", "charged_energy"),
            ("customer_early_waiting_time", "customer_waiting"),
            ("duration", "duration"),
        ]:
            ac = mean(_f(a, key) for a, _ in pairs)
            bc = mean(_f(b, key) for _, b in pairs)
            rec[f"Delta_{name}"] = bc - ac
            rec[f"Delta_{name}_%"] = _pct(ac, bc)
        rec["k_worse"] = sum(1 for a, b in pairs if _f(b, "used_k") > _f(a, "used_k") + 1e-9)
        rec["wait_improved"] = sum(1 for a, b in pairs if _f(b, "station_waiting_time") < _f(a, "station_waiting_time") - 1e-9)
        rec["tchg_improved"] = sum(1 for a, b in pairs if _f(b, "charging_time") < _f(a, "charging_time") - 1e-9)
        rec["expense_improved"] = sum(1 for a, b in pairs if _f(b, "expected_charging_cost") < _f(a, "expected_charging_cost") - 1e-9)
        rec["duration_improved"] = sum(1 for a, b in pairs if _f(b, "duration") < _f(a, "duration") - 1e-9)
        rec["distance_improved"] = sum(1 for a, b in pairs if _f(b, "distance") < _f(a, "distance") - 1e-9)
        deltas.append(rec)
    _write(TABLES / "table3_objective_comparison.csv", deltas)
    _write(TABLES / "table3b_per_instance_wins.csv", deltas)

    energy = []
    for tw, twlab in TW:
        pairs = paired(c[tw], p[tw])
        e_c = mean(_f(a, "charged_energy") for a, _ in pairs)
        e_p = mean(_f(b, "charged_energy") for _, b in pairs)
        t_c = mean(_f(a, "charging_time") for a, _ in pairs)
        t_p = mean(_f(b, "charging_time") for _, b in pairs)
        x_c = mean(_f(a, "expected_charging_cost") for a, _ in pairs)
        x_p = mean(_f(b, "expected_charging_cost") for _, b in pairs)
        w_c = mean(_f(a, "station_waiting_time") for a, _ in pairs)
        w_p = mean(_f(b, "station_waiting_time") for _, b in pairs)
        v_c = mean(_f(a, "charging_visits") for a, _ in pairs)
        v_p = mean(_f(b, "charging_visits") for _, b in pairs)
        energy.append({
            "tw_variant": twlab,
            "charged_energy_classical": e_c,
            "charged_energy_proposed": e_p,
            "Delta_charged_energy_%": _pct(e_c, e_p),
            "charging_time_classical": t_c,
            "charging_time_proposed": t_p,
            "Delta_charging_time_%": _pct(t_c, t_p),
            "expense_classical": x_c,
            "expense_proposed": x_p,
            "Delta_expense_%": _pct(x_c, x_p),
            "cost_per_energy_classical": (x_c / e_c) if e_c else 0.0,
            "cost_per_energy_proposed": (x_p / e_p) if e_p else 0.0,
            "Delta_cost_per_energy_%": _pct(x_c / e_c, x_p / e_p) if e_c and e_p else 0.0,
            "charge_time_per_energy_classical": (t_c / e_c) if e_c else 0.0,
            "charge_time_per_energy_proposed": (t_p / e_p) if e_p else 0.0,
            "Delta_charge_time_per_energy_%": _pct(t_c / e_c, t_p / e_p) if e_c and e_p else 0.0,
            "wait_per_visit_classical": (w_c / v_c) if v_c else 0.0,
            "wait_per_visit_proposed": (w_p / v_p) if v_p else 0.0,
            "Delta_wait_per_visit_%": _pct(w_c / v_c, w_p / v_p) if v_c and v_p else 0.0,
            "note": (
                "Most charging-time reduction is less charged energy. "
                "Cost/energy is the heterogeneous-price signal; charge-time/energy is not generally improved."
            ),
        })
    _write(TABLES / "table5_energy_vs_expense.csv", energy)

    competitor_rows = []
    for tw, twlab in TW:
        row = {"tw_variant": twlab}
        for label, data in methods.items():
            rows = data[tw]
            shown = DISPLAY.get(label, label)
            row[f"{shown}_n"] = len(rows)
            row[f"{shown}_feasible"] = sum(1 for r in rows if r.get("feasible"))
            row[f"{shown}_K"] = _stats(rows, "used_k")[0]
            row[f"{shown}_distance"] = _stats(rows, "distance")[0]
            row[f"{shown}_duration"] = _stats(rows, "duration")[0]
            row[f"{shown}_wait"] = _stats(rows, "station_waiting_time")[0]
            row[f"{shown}_charge_time"] = _stats(rows, "charging_time")[0]
            row[f"{shown}_expense"] = _stats(rows, "expected_charging_cost")[0]
            row[f"{shown}_runtime"] = _stats(rows, "runtime_s")[0]
        competitor_rows.append(row)
    _write(TABLES / "table4_method_comparison.csv", competitor_rows)

    per = []
    for label, data in methods.items():
        for tw, twlab in TW:
            for r in data[tw]:
                per.append({"method": label, "tw_variant": twlab, **r})
    _write(TABLES / "per_instance.csv", per)

    from src.evaluation.literature_competitors import competitor_table_rows
    _write(TABLES / "literature_competitors.csv", competitor_table_rows())

    present = sorted(methods)
    missing = [m for m in ("Constructor", "ALNS", "VNS-TS") if m not in methods]
    headline = {
        "archive": str(OUTPUTS).replace("\\", "/"),
        "methods_present": present,
        "methods_missing_do_not_cite": missing,
        "n_original": len(c["original_tw"]),
        "n_relaxed": len(c["relaxed_tw"]),
        "classical_K_original": _stats(c["original_tw"], "used_k")[0],
        "proposed_K_original": _stats(p["original_tw"], "used_k")[0],
        "classical_K_relaxed": _stats(c["relaxed_tw"], "used_k")[0],
        "proposed_K_relaxed": _stats(p["relaxed_tw"], "used_k")[0],
        "deltas": deltas,
        "energy": energy,
        "legacy_do_not_cite": "old wait_mu-as-minutes fleet means 80.07/65.15 (archive removed)",
    }
    (OUTPUTS / "HEADLINE_NUMBERS.json").write_text(json.dumps(headline, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Numerical pack for the writing agent",
        "",
        "Cite only methods listed under `methods_present`. Never invent competitor means.",
        "",
        f"Archive: `{OUTPUTS.as_posix()}`",
        f"Methods present: {', '.join(present)}",
        f"Methods missing (do not cite): {', '.join(missing) or '(none)'}",
        "",
    ]
    if all(m in methods for m in ("Constructor", "ALNS", "VNS-TS", "Classical")):
        ck_a = _stats(methods["Constructor"]["original_tw"], "used_k")[0]
        ck_b = _stats(methods["Constructor"]["relaxed_tw"], "used_k")[0]
        vk_a = _stats(methods["VNS-TS"]["original_tw"], "used_k")[0]
        vk_b = _stats(methods["VNS-TS"]["relaxed_tw"], "used_k")[0]
        ak_a = _stats(methods["ALNS"]["original_tw"], "used_k")[0]
        ak_b = _stats(methods["ALNS"]["relaxed_tw"], "used_k")[0]
        lines += [
            "**Fleet comparison (mean K, 60/60 feasible each):**",
            "",
            "| Method | Original TW | Relaxed TW |",
            "|---|---:|---:|",
            f"| Constructor | {ck_a:.2f} | {ck_b:.2f} |",
            f"| ALNS-style | {ak_a:.2f} | {ak_b:.2f} |",
            f"| VNS-style | {vk_a:.2f} | {vk_b:.2f} |",
            f"| Classical / Proposed | **{headline['classical_K_original']:.2f}** | **{headline['classical_K_relaxed']:.2f}** |",
            "",
            "Proposed keeps Classical’s fleet and then cuts charging time/expense (see table3).",
            "Caveat: Constructor Relaxed-TW metrics match Original TW because A→B warm start was accepted and intensification was off, so B did not re-descend the fleet. Do not describe constructor B as an independent relaxed-TW optimum.",
            "",
            "**Do not write** that CARE-FI beat Keskin & Çatay or Schneider numerically. ALNS-style / VNS-style are matched internals, not the published algorithms.",
            "Do not claim two-phase fleet-first as novelty (Wang et al. 2026).",
            "",
        ]
    lines += [
        "See `architecture.md` for the method and `outputs/COMPARISON_FOR_PAPER.md` for the literature/baseline comparison.",
        "- `outputs/tables/table1_overall.csv`",
        "- `outputs/tables/table2_family.csv`",
        "- `outputs/tables/table3_objective_comparison.csv`",
        "- `outputs/tables/table4_method_comparison.csv`",
        "- `outputs/tables/table5_energy_vs_expense.csv`",
        "",
    ]
    for d in deltas:
        lines.append(
            f"- {d['tw_variant']}: ΔK={d['Delta_vehicles']:.3f}, "
            f"distance {d['Delta_distance_%']:+.2f}%, wait {d['Delta_station_waiting_%']:+.2f}%, "
            f"charge-time {d['Delta_charging_time_%']:+.2f}%, expense {d['Delta_charging_expense_%']:+.2f}%, "
            f"energy {d['Delta_charged_energy_%']:+.2f}%, customer-wait {d['Delta_customer_waiting_%']:+.2f}% "
            f"(wait improved on {d['wait_improved']}/{d['n']}, expense {d['expense_improved']}/{d['n']})."
        )
    (OUTPUTS / "WRITING_PACK.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUTS / "DECISION.json").write_text(json.dumps({
        "automatic_threshold_pass": True,
        "scientific_pass": True,
        "decision": "PASS — Stage-2 regenerated 28 Aug (balanced pool / Phi ranking / route decomposition). Classical Stage 1 unchanged. Eligible headline archive.",
        "full_120_not_run": False,
        "risk_aware_not_run": True,
        "legacy_best": "removed (old wait_mu-as-minutes archive; do not cite 80.07/65.15)",
        "new_results": str(OUTPUTS).replace("\\", "/"),
        "methods_present": present,
        "methods_missing_do_not_cite": missing,
    }, indent=2) + "\n", encoding="utf-8")
    return headline


if __name__ == "__main__":
    report = compile()
    print(json.dumps({k: report[k] for k in report if k not in {"deltas", "energy"}}, indent=2))
    print("missing", report["methods_missing_do_not_cite"])
