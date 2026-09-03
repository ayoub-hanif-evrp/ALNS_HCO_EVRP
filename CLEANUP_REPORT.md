# Hanif-paper repository cleanup

Date: 2026-09-03. File organization only; Hanif numerical results were copied, not recomputed.

## Backup

Created **before** any moves:

| Path | What |
|---|---|
| `archive/repo_before_hanif_cleanup.bundle` | Full git history (`git clone archive/repo_before_hanif_cleanup.bundle restored`) |
| `archive/removed_trees.zip` | Working-tree snapshot of outputs, Schneider trees, scripts, configs, legacy |
| `archive/README.txt` | Restore notes |

`archive/` is gitignored so the bundle is not pushed.

## Deleted from the Hanif working tree

Not present under `outputs/` after cleanup:

- Duplicate `outputs/hanif/paper/data/master_results.csv` (same rows as `outputs/raw/master_results.csv`)
- Duplicate `outputs/hanif/runs/latest_rows.json` (same runs as the master CSV)
- Placeholder `figS01_representative_route.md` and `figS02_soc_profile.md` (no saved Hanif route/SoC data)
- `outputs/README.md` (old multi-benchmark layout)

Gitignored scratch still on disk if present: `dataset/_tmp_evrptw/`, `dataset/_tmp_schneider/`, empty `outputs/hanif/` leftovers.

## Retained (Hanif paper)

**Results** under `outputs/`:

- `raw/master_results.csv` — 3720 runs, 120 instances (60 original TW + 60 relaxed), methods GI, VNS, SA, ALNS-FR, ALNS-PR, ALNS-HCO, ALNS-PR+HCO, seeds 0–4
- Metrics in that CSV: feasible, served, unserved, hard_violations, K, distance, duration, customer_waiting, charging_visits, charged_energy, station_waiting, charging_time, charging_expense, runtime, phase times, K*
- `processed/` — figure CSVs, captions, METHOD, RESULTS_SUMMARY, experiment_config
- `ablation/` — Table 4 (FR / PR / PR+HCO / HCO)
- `statistics/` — paired % , bootstrap 95% CI, Wilcoxon p, win/tie/loss
- `tables/` — Tables 1, 2, 3, 5, A01
- `figures/` — Figs 1–6; `figures/supplement/` Fig S3 runtime
- `diagnostics/` — quality checks

**Code (unchanged solvers):** `src/alns-hco/`, `src/care-fi/` (Hanif loader/replay/GI/VNS/SA), `src/common/`, `src/schneider/` (kept because it is solver source, not a result folder).

**Data:** `dataset/Hanif/`

**Scripts kept in `scripts/`:** `run_all_paper_experiments.py`, `run_all_paper_experiments.sh`

**Tests kept:** `tests/test_alns_hco.py`, `test_hanif_dataset.py`, `test_care_fi.py`, `test_hanif_carefi.py`, fixtures

## Final directory tree (paper-relevant)

```
outputs/
  raw/master_results.csv
  processed/          captions, METHOD, RESULTS_SUMMARY, experiment_config, fig*.csv, table*.csv
  ablation/           table04_ablation.{csv,md,tex}
  statistics/         statistical_tests.csv, table03_paired_alns_pr_vs_hco.csv
  tables/             table01, table02, table03, table05, tableA01
  figures/            fig01–fig06 .{png,svg,pdf}
  figures/supplement/ figS03_runtime_distribution
  diagnostics/        quality_checks.txt, legacy_vs_paper.md
scripts/
  run_all_paper_experiments.py
  run_all_paper_experiments.sh
src/alns-hco/         solver, HCO, paper.py
src/care-fi/          loader, replay, GI/VNS/SA
src/common/           config, stats, paths
dataset/Hanif/
review_before_delete/ see README there
archive/              local backup (gitignored)
```

## Manuscript mapping

| Manuscript item | File |
|---|---|
| Table 1 experimental setup | `outputs/tables/table01_experimental_setup.md` |
| Table 2 overall method comparison | `outputs/tables/table02_algorithm_comparison.md` |
| Table 3 ALNS-PR vs ALNS-HCO | `outputs/tables/table03_main_hco_effect.md` and `outputs/statistics/statistical_tests.csv` |
| Table 4 ablation | `outputs/ablation/table04_ablation.md` |
| Table 5 family analysis | `outputs/tables/table05_family_analysis.md` |
| Table A01 reproducibility | `outputs/tables/tableA01_reproducibility.md` |
| Fig 1 framework | `outputs/figures/fig01_alns_hco_framework.*` + `processed/fig01_alns_hco_framework.csv` |
| Fig 2 paired improvements | `outputs/figures/fig02_paired_improvements.*` + `processed/fig02_paired_improvements.csv` |
| Fig 3 charging-cost scatter | `outputs/figures/fig03_charging_cost_scatter.*` + `processed/fig03_charging_cost_scatter.csv` |
| Fig 4 charging-mechanism decomposition | `outputs/figures/fig04_charging_mechanisms.*` + `processed/fig04_charging_mechanisms.csv` |
| Fig 5 family heatmap | `outputs/figures/fig05_family_heatmap.*` + `processed/fig05_family_heatmap.csv` |
| Fig 6 runtime (supplement-ready) | `outputs/figures/fig06_runtime_quality.*` + `processed/fig06_runtime_quality.csv` |
| Fig S3 runtime distribution | `outputs/figures/supplement/figS03_runtime_distribution.*` |
| Representative route / SoC | **not retained** (no saved Hanif route/SoC files) |

Regenerate tables/figures from the same master CSV:

`python scripts/run_all_paper_experiments.py --compile-only`

## Not deleted (uncertain or source)

Put in `review_before_delete/` rather than discarded:

- Schneider dataset and all Schneider experimental outputs
- EVRPTWD dataset
- CARE-FI and pre-paper ALNS-HCO result archives
- `legacy/` and `legacy_reference/`
- Extra scripts (Schneider, CARE-FI, FPS, old paper generators)
- Extra tests (Schneider, FPS, old paper)

Left in place on purpose:

- `src/schneider/` — Schneider solver source, not experimental CSVs
- `src/alns-hco/schneider_paper.py` — compiler for the old Schneider table
- `configs/` — configuration files
- `METHOD.md` (CARE-FI method note at repo root, if still present)
- `main_train.py`, architecture docs — not Hanif paper results; still in the repo unless already moved

Station-level \(\bar P_s\), \(\bar W_s\), \(r_s\) time series were **not** in the paper master CSV, so nothing extra was kept for that.

Derived ratios \(C_{ch}/E_{ch}\), \(T_{wait}/N_{ch}\), \(T_{ch}/E_{ch}\) are computed in Fig 4 from the master CSV columns (energy, expense, wait, visits, charge time); they are in `processed/fig04_charging_mechanisms.csv`.
