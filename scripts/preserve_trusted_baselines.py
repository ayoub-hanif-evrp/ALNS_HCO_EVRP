"""Extract trusted baseline rows and purge invalid HCO/Phase-II results before the corrected rerun."""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
RAW = OUT / "raw"
BASELINE_DIR = OUT / "baselines_trusted"
TRUSTED = {
    "GI", "VNS", "SA", "ALNS-FR", "ALNS-PR", "PhaseI-PR-15",
}


def _load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("solution_results.csv", "master_results.csv", "route_results.csv", "station_visits.csv"):
        src = RAW / name
        rows = _load(src)
        trusted = [r for r in rows if r.get("method") in TRUSTED]
        _write(BASELINE_DIR / name, trusted)
        print(f"{name}: kept {len(trusted)} / {len(rows)} trusted baseline rows")

    # Rebuild raw with baselines only (HCO rows will be appended by the corrected rerun)
    for name in ("solution_results.csv", "master_results.csv", "route_results.csv", "station_visits.csv"):
        shutil.copy2(BASELINE_DIR / name, RAW / name)

    # HCO activity must be fully regenerated
    for p in (RAW / "hco_activity.csv",):
        if p.is_file():
            p.unlink()
            print(f"deleted {p}")

    # Delete invalid HCO figures / tables / analyses
    victims = [
        OUT / "legacy_pre_revision",
        OUT / "figures" / "figB_fixed_route_heterogeneity.png",
        OUT / "figures" / "figC_station_characteristics.png",
        OUT / "figures" / "figD_phase1_to_phase2.png",
        OUT / "figures" / "figE_phase2_heterogeneity.png",
        OUT / "figures" / "figF_family_heterogeneous_effect.png",
        OUT / "tables" / "table_fixed_route_heterogeneity.csv",
        OUT / "tables" / "table_phase1_to_phase2.csv",
        OUT / "tables" / "table_phase1_to_phase2_plain.csv",
        OUT / "tables" / "table_phase2_heterogeneity.csv",
        OUT / "tables" / "table_phase2_plain_vs_heterogeneous.csv",
        OUT / "tables" / "table_station_selection.csv",
        OUT / "tables" / "table_family_results.csv",
        OUT / "analysis" / "hco_activation_analysis.csv",
        OUT / "analysis" / "research_questions.md",
        OUT / "analysis" / "time_allocation_sensitivity.csv",
        OUT / "config" / "epsilon_selection_dev.csv",
        OUT / "processed" / "charging_indicators.csv",
        OUT / "statistics" / "paired_comparisons.csv",
        OUT / "figure_candidates_manifest.md",
        OUT / "figure_manifest.md",
        OUT / "final_experiment_report.md",
        OUT / "code_revision_notes.md",
    ]
    for p in victims:
        if p.is_dir():
            shutil.rmtree(p)
            print(f"removed dir {p}")
        elif p.is_file():
            p.unlink()
            print(f"deleted {p}")

    # Keep figA and partial-recharging / overall tables
    print("Preserved: shared_phase1/, figA, table_partial_recharging, table_overall_methods (if present)")
    print("Trusted baselines in", BASELINE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
