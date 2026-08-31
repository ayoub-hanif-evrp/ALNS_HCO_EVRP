# CARE-FI

Deterministic two-stage EVRPTW solver.

**CARE-FI** is the method. The **Hanif 1000-customer dataset** is one benchmark it is applied to. The **Schneider EVRPTW set** is a separate published-BKS benchmark. Do not call CARE-FI “the Hanif method.”

Paper method: `METHOD.md`.

## Run

```powershell
python scripts/run_hanif_carefi.py --representative --tw both --budget 180 --workers 2
python scripts/run_schneider_benchmark.py --mode pr --budget 180 --workers 2
```

Scratch files go to `outputs/_tmp/` (gitignored). Delete that folder when you are done. Never write into `outputs/hanif/` or `outputs/schneider/`.

## Citeable archives (do not overwrite)

| Benchmark | Archive | Headline |
|---|---|---|
| Hanif Stage 1 / Stage 2 | `outputs/hanif/classical/`, `outputs/hanif/proposed/` | K Original 78.82, Relaxed 65.20 |
| Schneider PR | `outputs/schneider/pr/` | 56/56 feasible, 5 fleet ties, mean ΔK +2.64 |

## Layout

```
src/hanif/        CARE-FI for the Hanif 1000-customer set
src/schneider/    Schneider BKS branch
outputs/hanif/    citeable Hanif results
outputs/schneider/  citeable Schneider results
legacy/           old training / search, not the paper method
```

```powershell
pytest tests/test_hanif_carefi.py -q
```
