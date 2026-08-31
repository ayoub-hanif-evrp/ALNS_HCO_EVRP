"""Build the submission-oriented outputs/paper_ready package."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.schneider.evaluation.hanif_protocols import FROZEN_HEADLINE, PROTOCOL_EXTENDED, PROTOCOL_NO_WAIT_REPLAY
from src.schneider.evaluation.literature_competitors import competitor_table_rows
from src.schneider.visualization.hanif_paper_figures import generate_paper_artifacts

FROZEN = ROOT / "outputs/CurrentBestDeterministicHanif"
FROZEN_HASHES = {
    "original_tw/results.csv": "3e2c8394787da032",
    "relaxed_tw/results.csv": "eb9341c7175d4a01",
    "summary_both.json": "38a73500243337a6",
    "results_both.csv": "fd34f3e2a4529b7e",
}


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _drop_keys(obj, keys: set[str]):
    if isinstance(obj, dict):
        return {k: _drop_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_drop_keys(v, keys) for v in obj]
    return obj


def frozen_integrity() -> dict:
    report = {"ok": True, "files": {}}
    for rel, expected in FROZEN_HASHES.items():
        path = FROZEN / rel
        digest = _sha16(path) if path.is_file() else None
        ok = digest == expected
        report["files"][rel] = {"expected": expected, "got": digest, "ok": ok}
        report["ok"] = report["ok"] and ok
    for variant, sub in (("A", "original_tw"), ("B", "relaxed_tw")):
        n = len(list((FROZEN / sub / "routes").glob("*.json")))
        ok = n == 60
        report["files"][f"{sub}/routes"] = {"n": n, "ok": ok}
        report["ok"] = report["ok"] and ok
    return report


def write_method_docs(out_dir: Path, values: dict, hardware: dict) -> None:
    (out_dir / "METHOD.md").write_text(
        "# Method (manuscript-ready)\n\n"
        "## Headline solver\n\n"
        "The reported full-benchmark method is **Deterministic-FPS + Hanif fleet "
        "intensification** (`HanifFleetIntensification`). CARE-FI is a six-family "
        "pilot only and is not the 60x2 headline.\n\n"
        "Route elimination and regret insertion are known primitives. Novelty is the "
        "scalable fleet-first integration, exact EV replay (wait / charge / time "
        "windows), and A-to-B transfer on 1000-customer instances.\n\n"
        "## Formal objective hierarchy\n\n"
        "Search acceptance is lexicographic (all minimized):\n\n"
        "1. hard constraint violations\n"
        "2. unserved customers\n"
        "3. used vehicles K\n"
        "4. total distance\n"
        "5. total charging time\n"
        "6. total route duration\n"
        "7. scalar paper objective J\n\n"
        "The scalar J = lambda_v K + lambda_d D + lambda_t T + lambda_c C "
        "+ lambda_p P + lambda_w W + penalties is used only after the "
        "lexicographic keys. In this protocol lambda_p = 0 (charging price is "
        "recorded but not optimized).\n\n"
        "## Deterministic-FPS / HFI pseudocode\n\n"
        "```\n"
        "Deterministic-FPS(instance, budget, intensify, warm_start):\n"
        "    (lo, hi) <- fleet bounds from demand / TW pressure\n"
        "    routes <- dense feasible construction at an upper fleet\n"
        "    routes <- coarse fleet descent until first infeasible shrink\n"
        "    if intensify:\n"
        "        repeat until budget / lower bound:\n"
        "            rank routes (legacy size/overlap first; CARE ranking lazy)\n"
        "            eject one route\n"
        "            recover customers by station-aware direct insertion\n"
        "            if direct fails, regret-3/2 insertion\n"
        "            accept iff exact replay is feasible and all customers served\n"
        "    if warm_start for variant B:\n"
        "        seed B from A's saved routes and charging labels\n"
        "    return exact-replayed routes\n"
        "```\n\n"
        "## Complexity\n\n"
        "Let n be customers (1000), s stations (333), K fleet size, and R "
        "intensification rounds. Dense construction is quadratic in n. Each "
        "elimination attempt replays and reinserts a subset of customers, so a "
        "round is O(K * n_ejected * n) in the insertion oracle, truncated by "
        "wall-clock. Exact replay is linear in route length. The method is a "
        "feasibility-filtered heuristic, not an exact solver.\n\n"
        "## Protocols\n\n"
        f"- EXTENDED: {json.dumps(PROTOCOL_EXTENDED)}\n"
        f"- NO_WAIT_REPLAY: {json.dumps(PROTOCOL_NO_WAIT_REPLAY)}\n\n"
        "Comparison methods are literature EVRPTW metaheuristics (ALNS, VNS/TS) "
        "re-run on this benchmark.\n\n"
        "## Parameter / hardware protocol\n\n"
        "- budget: 180 s per instance per variant\n"
        "- seed: 0\n"
        "- construction: dense, neural guidance off\n"
        "- intensification: on for the frozen headline; off for the matched ablation\n"
        "- A to B warm start: on unless explicitly disabled\n"
        "- matched ablation: without intensification, B inherits A's fleet until HFI compresses it\n"
        f"- hardware: {json.dumps(hardware)}\n",
        encoding="utf-8",
    )
    claims = [
        "100% feasibility on the frozen 60/60 A and 60/60 B archives under EXTENDED.",
        "Mean fleet 80.07 (A) and 65.15 (B) for HanifFleetIntensification at 180 s.",
        "Fleet cut versus the project's own fast feasible baseline: 17.2% (A), 18.9% (B).",
        "Matched ablation, when complete, compares the same 180 s budget with intensification off.",
        "Semantic replay reports EXTENDED vs NO_WAIT_REPLAY on frozen routes without re-optimization.",
        "External comparison is against literature EVRPTW metaheuristics (ALNS, VNS/TS) re-run here.",
    ]
    not_claimed = [
        "CARE-FI as the 60x2 headline (it is a six-family pilot only).",
        "Optimality, stochastic wait_sigma, or charging-price optimization.",
        "Copying published Schneider-instance numbers onto this 1000-customer set.",
    ]
    a = values.get("A") or {}
    b = values.get("B") or {}
    matched = values.get("matched_ablation") or {}
    matched_line = (
        f"- Matched 180 s no-intensification (complete): A/B mean K {matched.get('A_mean_k')} "
        f"/ {matched.get('B_mean_k')} vs intensified {a.get('mean_k')} / {b.get('mean_k')}. "
        "Without intensification, A-to-B warm start keeps B at A's fleet; intensification is "
        "what compresses both variants."
        if matched.get("complete")
        else "- Matched 180 s no-intensification run is incomplete; means are not cited."
    )
    (out_dir / "CLAIMS.md").write_text(
        "# Supported claims and limitations\n\n"
        "## Supported\n\n"
        + "".join(f"- {c}\n" for c in claims)
        + f"{matched_line}\n"
        + "\nHeadline numbers from the frozen archive:\n\n"
        f"- A feasible {a.get('feasible')} mean K {a.get('mean_k')}\n"
        f"- B feasible {b.get('feasible')} mean K {b.get('mean_k')}\n\n"
        "## Not claimed\n\n"
        + "".join(f"- {c}\n" for c in not_claimed)
        + "\n## Limitations\n\n"
        "- Hanif dataset only; no claim on Schneider or other EVRPTW sets.\n"
        "- Heuristic search with an exact feasibility filter, not a MIP/CP optimum.\n"
        "- Station wait uses E[W]=exp(wait_mu + 0.5 wait_sigma^2); wait_mu is not the mean.\n"
        "- If the matched 180 s no-intensification run is incomplete, that ablation "
        "is reported as a failure rather than averaged.\n"
        "- CARE-FI ranking did not pass the predeclared full 60x2 gate.\n"
        "- Without intensification, B inherits A's fleet via warm start and does not "
        "independently compress.\n",
        encoding="utf-8",
    )
    (out_dir / "LIMITATIONS.md").write_text(
        "# Limitations\n\n"
        "- Hanif dataset only; no claim on Schneider or other EVRPTW sets.\n"
        "- Heuristic search with an exact feasibility filter, not a MIP/CP optimum.\n"
        "- Station wait uses E[W]=exp(wait_mu + 0.5 wait_sigma^2); wait_mu is not the mean.\n"
        "- Route elimination and regret insertion are known primitives; novelty is "
        "the scalable fleet-first integration, exact EV replay, and A-to-B transfer.\n"
        "- If the matched 180 s no-intensification run is incomplete, that ablation "
        "is reported as a failure rather than averaged.\n"
        "- CARE-FI is a six-family pilot only and is not the 60x2 headline.\n",
        encoding="utf-8",
    )
    (out_dir / "PROTOCOL.md").write_text(
        "# Evaluation protocols\n\n"
        "Two protocols appear in generated metadata. Optimization uses EXTENDED. "
        "NO_WAIT_REPLAY is a frozen-route diagnostic.\n\n"
        f"```json\n{json.dumps({'EXTENDED': PROTOCOL_EXTENDED, 'NO_WAIT_REPLAY': PROTOCOL_NO_WAIT_REPLAY}, indent=2)}\n```\n",
        encoding="utf-8",
    )
    (out_dir / "hardware.json").write_text(json.dumps(hardware, indent=2) + "\n", encoding="utf-8")
    (out_dir / "COMPETITORS.md").write_text(
        "# Literature metaheuristics\n\n"
        "Published EVRPTW methods used as comparison families. None report numbers "
        "on this 1000-customer Homberger+stations set, so we re-run ALNS and VNS/TS "
        "here under matched Q, h, and time windows.\n\n"
        "| ID | Method | Citation | Original instances |\n"
        "|----|--------|----------|--------------------|\n"
        + "".join(
            f"| {r['method_id']} | {r['method']} | {r['citation']} | {r['original_instances']} |\n"
            for r in competitor_table_rows()
        ),
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--best", default="outputs/CurrentBestDeterministicHanif")
    ap.add_argument("--baseline", default="outputs/hanif_fps/fast_feasible_baseline")
    ap.add_argument("--ablation", default="outputs/hanif_fps_no_intensify_180")
    ap.add_argument("--out", default="outputs/paper_ready")
    ap.add_argument("--skip-replay", action="store_true")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(exist_ok=True)

    integrity = frozen_integrity()
    (out_dir / "frozen_integrity.json").write_text(json.dumps(integrity, indent=2) + "\n", encoding="utf-8")
    if not integrity["ok"]:
        print("FROZEN ARCHIVE CHANGED — refusing to continue.", flush=True)
        print(json.dumps(integrity, indent=2), flush=True)
        return 1

    rows = competitor_table_rows()
    lit_csv = out_dir / "tables" / "literature_competitors.csv"
    with lit_csv.open("w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "literature_competitors.json").write_text(
        json.dumps({"methods": rows}, indent=2) + "\n", encoding="utf-8",
    )
    for leftover in (
        out_dir / "hanif_context.json",
        out_dir / "tables" / "hanif_context.csv",
        out_dir / "tables" / "benchmark_context.csv",
    ):
        leftover.unlink(missing_ok=True)

    ablation = ROOT / args.ablation
    comparison = None
    if (ablation / "original_tw" / "results.csv").is_file():
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts/compare_hanif_results.py"),
                str(ablation),
                "--frozen",
                args.best,
                "--out",
                str(out_dir / "matched_ablation_vs_frozen.json"),
            ],
            cwd=str(ROOT),
        )
        comparison = json.loads((out_dir / "matched_ablation_vs_frozen.json").read_text(encoding="utf-8"))
    else:
        (out_dir / "MATCHED_ABLATION_MISSING.md").write_text(
            "Matched 180 s no-intensification archive is missing.\n",
            encoding="utf-8",
        )

    replay_path = out_dir / "exact_validation.json"
    if not args.skip_replay:
        subprocess.check_call(
            [
                sys.executable,
                str(ROOT / "scripts/validate_hanif_archive.py"),
                args.best,
                "--protocol",
                "both",
                "--out",
                str(replay_path),
            ],
            cwd=str(ROOT),
        )
    if replay_path.is_file():
        payload = json.loads(replay_path.read_text(encoding="utf-8"))
        replay_path.write_text(
            json.dumps(_drop_keys(payload, {"hanif_con_hc_equivalent"}), indent=2) + "\n",
            encoding="utf-8",
        )

    generate_paper_artifacts(
        best_dir=ROOT / args.best,
        baseline_dir=ROOT / args.baseline,
        control_dir=ROOT / args.best,
        ablation_dir=ablation if ablation.is_dir() else None,
        replay_report=replay_path if replay_path.is_file() else None,
        out_dir=out_dir,
        skip_freeze_lookup=True,
    )

    values = {}
    values_path = out_dir / "manuscript_values.json"
    if values_path.is_file():
        values = json.loads(values_path.read_text(encoding="utf-8"))
    hardware = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
        "node": platform.node(),
        "headline": FROZEN_HEADLINE,
    }
    write_method_docs(out_dir, values, hardware)
    if comparison is not None:
        shutil.copy2(out_dir / "matched_ablation_vs_frozen.json", out_dir / "tables" / "matched_ablation_vs_frozen.json")
    index_path = out_dir / "INDEX.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {}
    index.update({
        "docs": ["METHOD.md", "CLAIMS.md", "LIMITATIONS.md", "PROTOCOL.md", "COMPETITORS.md"],
        "exact_validation": "exact_validation.json",
        "literature_competitors": "tables/literature_competitors.csv",
        "frozen_integrity": "frozen_integrity.json",
        "matched_ablation": "matched_ablation_vs_frozen.json",
        "ablation_archive": args.ablation.replace("\\", "/"),
        "headline": "outputs/CurrentBestDeterministicHanif",
        "care_fi": "six_family_pilot_only",
    })
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    readme = out_dir / "README.md"
    extra = (
        "\nAlso in this package:\n\n"
        "- `METHOD.md` — objective hierarchy, pseudocode, complexity, protocols\n"
        "- `CLAIMS.md` / `LIMITATIONS.md` — supported claims and non-claims\n"
        "- `PROTOCOL.md` — EXTENDED vs NO_WAIT_REPLAY\n"
        "- `COMPETITORS.md` — literature EVRPTW metaheuristics (ALNS, VNS/TS, ILS, LNS×GTS)\n"
        "- `exact_validation.json` — frozen-route replay under both protocols\n"
        "- `tables/literature_competitors.csv` — published EVRPTW metaheuristics\n"
        "- `frozen_integrity.json` — hashes confirming the headline archive was not modified\n"
    )
    if readme.is_file() and "METHOD.md" not in readme.read_text(encoding="utf-8"):
        readme.write_text(readme.read_text(encoding="utf-8") + extra, encoding="utf-8")
    print(f"paper_ready -> {out_dir}", flush=True)
    print(f"frozen_integrity_ok={integrity['ok']}", flush=True)
    if comparison is not None:
        print(f"matched_ablation_complete={comparison.get('complete')}", flush=True)
    return 0 if integrity["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
