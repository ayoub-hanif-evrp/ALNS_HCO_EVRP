# CARE-FI and ALNS-HCO

**CARE-FI** is a deterministic two-stage solver. **ALNS-HCO** is Adaptive Large
Neighborhood Search with a Heterogeneous Charging Oracle. The **Hanif**
1000-customer set and the **Schneider** EVRPTW set are separate benchmarks.
Do not call either algorithm “the Hanif method.”

ALNS-HCO method: `src/alns-hco/METHOD.md`. CARE-FI method: `METHOD.md`.

## Run ALNS-HCO paper experiments (Hanif)

```powershell
python scripts/run_all_paper_experiments.py --representative --seeds 0 --workers 8
python scripts/run_all_paper_experiments.py --workers 8
python scripts/run_all_paper_experiments.py --compile-only
```

Equal budget: ALNS-PR 30 s vs ALNS-HCO 15 s + 15 s. Seeds `{0,1,2,3,4}` (`n=5`).

## Layout

```
src/common/       shared experiment/paper utilities
src/alns-hco/     ALNS-HCO (Hanif experiments import as src.alns_hco)
src/care-fi/      CARE-FI
src/schneider/    Schneider BKS branch
outputs/hanif/    ALNS-HCO paper package (Hanif only)
outputs/alns-hco/ frozen pre-paper ALNS-HCO snapshot
outputs/care-fi/  CARE-FI Hanif archive
outputs/schneider/ Schneider archive (never mixed with Hanif)
legacy_reference/ frozen copy of the pre-paper ALNS-HCO solver and CSVs
```
