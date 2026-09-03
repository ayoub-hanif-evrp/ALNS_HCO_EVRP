# CARE-FI and ALNS-HCO (Hanif paper)

**ALNS-HCO** is Adaptive Large Neighborhood Search with a Heterogeneous Charging
Oracle. **Hanif** is the 1000-customer EVRPTW benchmark (60 original TW + 60
relaxed TW; families C1, C2, R1, R2, RC1, RC2).

Method: `src/alns-hco/METHOD.md` and `METHOD_ALNS_HCO.md`.

## Reproduce Hanif paper tables and figures

```powershell
python scripts/generate_hanif_figures.py
python scripts/run_all_paper_experiments.py --compile-only
python scripts/run_all_paper_experiments.py --workers 8
```

Equal budget: ALNS-PR 30 s vs ALNS-HCO 15 s + 15 s. Seeds `{0,1,2,3,4}`.

## Layout

```
dataset/Hanif/           120 converted instances
src/alns-hco/            ALNS-HCO solver, HCO, paper compilers
src/care-fi/             instance loader, replay, GI/VNS/SA/ALNS-FR
src/common/              paper config and stats
scripts/                 Hanif experiment + table/figure regeneration
outputs/                 Hanif paper numbers, tables, figures
```

See `outputs/README.md` and `outputs/figure_manifest.md`.
