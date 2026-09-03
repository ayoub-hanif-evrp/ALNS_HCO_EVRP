# Hanif paper results

Source of truth: `raw/master_results.csv` (3720 rows). Numbers were not
recomputed; tables and Figures 2–5 are compiled from this file.

| Path | Contents |
|---|---|
| `raw/master_results.csv` | Every (instance, TW, method, seed) row |
| `processed/` | Figure CSVs, captions, METHOD, RESULTS_SUMMARY |
| `ablation/` | Table 4 |
| `statistics/` | Paired tests |
| `tables/` | Tables 1–3, 5, A01 |
| `figures/` | Fig 2–5 (png/svg/pdf). Fig 1 is the author framework drawing |
| `figures/archive/` | Discarded dashboard-style figures |
| `figure_manifest.md` | Mapping from each figure to source and script |

Regenerate Figures 2–5 only (does not re-run the solver, does not overwrite Fig 1):

```powershell
python scripts/generate_hanif_figures.py
```
