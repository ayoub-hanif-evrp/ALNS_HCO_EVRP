# Commands — Deterministic Hanif two-stage EVRPTW

Frozen paper results live directly under `outputs/` (do not overwrite those folders).

Method details: `architecture.md`.

## Re-run Classical fleet search into a new folder

```powershell
python scripts/run_hanif_fps.py --all --variant both --budget 180 --workers 8 --out outputs/hanif_fps_run_new
```

## Two-stage Proposed (write to a new folder; do not overwrite `outputs/classical` or `outputs/proposed`)

```powershell
python scripts/run_final_expected.py --skip-classical --full --out outputs/refinement_rerun
```

## Matched baselines (already present)

- `outputs/constructor`
- `outputs/alns`
- `outputs/vns_ts`

Re-run into a **new** folder if needed. `scripts/run_paper_baselines.py` skips folders that are already complete.

## Compile paper tables

```powershell
python scripts/compile_paper_results.py
```

## Semantic replay of the headline archive

```powershell
python scripts/validate_hanif_archive.py outputs/proposed --protocol both
```
