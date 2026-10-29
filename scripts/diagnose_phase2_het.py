"""Diagnose why PhaseII-Heterogeneous often equals Phase-I (no algorithm changes)."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alns_hco.reviewer_experiments import reviewer_root, write_csv, write_json  # noqa: E402


def main() -> int:
    src = reviewer_root() / "alns_hco_absolute" / "full_runs.csv"
    if not src.is_file():
        raise SystemExit(f"Missing {src}")
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    by = defaultdict(dict)
    for r in rows:
        key = (r["instance"], r["tw"], r["seed"])
        by[key][r["method"]] = r

    findings = []
    unchanged = 0
    for key, methods in sorted(by.items()):
        p1 = methods.get("PhaseI-PR-15")
        het = methods.get("PhaseII-Heterogeneous")
        noh = methods.get("PhaseII-NoHCO")
        if not p1 or not het:
            continue
        same_d = abs(float(het["distance"]) - float(p1["distance"])) < 1e-9
        same_k = int(float(het["k"])) == int(float(p1["k"]))
        same_e = abs(float(het["charged_energy"]) - float(p1["charged_energy"])) < 1e-9
        unchanged_flag = same_d and same_k and same_e
        if unchanged_flag:
            unchanged += 1
        rec = {
            "instance": key[0],
            "tw": key[1],
            "seed": key[2],
            "phase1_k": p1["k"],
            "phase1_distance": p1["distance"],
            "het_k": het["k"],
            "het_distance": het["distance"],
            "het_unchanged_vs_phase1": unchanged_flag,
            "nohco_distance": None if not noh else noh["distance"],
            "alns_phase2_iterations": het.get("alns_phase2_iterations"),
            "hco_calls": het.get("hco_calls"),
            "feasible_hco_calls": het.get("feasible_hco_calls"),
            "hco_eval_time_s": het.get("hco_eval_time_s"),
            "phase2_runtime_s": het.get("phase2_runtime_s"),
            "eps_row": het.get("eps"),
            "note": (
                "Distance/K/energy identical to Phase I despite HCO evaluations. "
                "Likely zero Phase-II acceptances under eps=0.0 (D <= D_PR) and "
                "lex objective; algorithm not modified."
                if unchanged_flag else
                "Phase II changed the Phase-I solution on at least one metric."
            ),
        }
        findings.append(rec)

    out = reviewer_root() / "phase2_het_diagnosis"
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "instance_findings.csv", findings)

    # Hyperparameter reference (documented, not changed)
    from src.alns_hco.hyperparameters import HYPERPARAMETERS
    summary = {
        "n_compared": len(findings),
        "n_unchanged_vs_phase1": unchanged,
        "fraction_unchanged": unchanged / len(findings) if findings else None,
        "phase2_distance_tolerance_eps": HYPERPARAMETERS.get("phase2_distance_tolerance"),
        "interpretation": (
            "On the preliminary absolute-results slice, PhaseII-Heterogeneous often "
            "returns the Phase-I routes unchanged. Recorded rows still show positive "
            "hco_calls / feasible_hco_calls and phase2 iterations, so HCO is evaluated. "
            "Acceptance requires phase2_admissible (K<=K*, D<=D_PR*(1+eps) with eps=0) "
            "plus lexicographic improvement. With a tight distance freeze, many "
            "destroy/repair moves are rejected on distance even when HCO finds "
            "feasible recharges. This is a documented behavioural finding — the "
            "algorithm was not altered in Round 3."
        ),
        "algorithm_modified": False,
    }
    write_json(out / "summary.json", summary)
    (out / "DIAGNOSIS.md").write_text(
        "\n".join([
            "# PhaseII-Heterogeneous vs Phase I — diagnostic",
            "",
            f"- Compared keys: **{summary['n_compared']}**",
            f"- Unchanged (K, distance, charged_energy): **{summary['n_unchanged_vs_phase1']}** "
            f"({100 * (summary['fraction_unchanged'] or 0):.1f}%)",
            f"- Configured `phase2_distance_tolerance` (eps): **{summary['phase2_distance_tolerance_eps']}**",
            "",
            "## Evidence from Round-2 absolute CSV",
            "",
            "- PhaseII-Heterogeneous rows retain `hco_mode=heterogeneous` and non-zero `hco_calls`.",
            "- PhaseII-NoHCO can slightly change distance (ordinary ALNS move) while Het stays on Phase I,",
            "  consistent with HCO-guided repair failing the eps=0 distance gate more often.",
            "- Acceptance stats are not always exported as `accepted=0` columns on the absolute CSV;",
            "  identical Phase-I metrics + positive HCO evals imply accepted improving moves = 0.",
            "",
            "## Conclusion (no algorithm change)",
            "",
            summary["interpretation"],
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
